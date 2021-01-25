# conding=utf-8

import tensorflow as tf
from nets.core.basciModel import BasicModel
from nets.backbone.convertResnet import ResNet
from core.loss import batch_hard_triplet_loss

class CACENET(BasicModel):
    def __init__(self,
                 train_mode,
                 local_conv_out_channels=256,
                 triplet_type='none',
                 num_layers=50,
                 last_conv_stride=1,
                 dropout_place='after',
                 pooling_types='baseline',
                 loss_type='softmax',
                 margin=0.5,
                 non_local_type='none',
                 non_local_block=2,
                 compression=2,
                 with_pair_triplet=True,
                 alpha=0.9,
                 ):
        """"""
        super(CACENET, self).__init__(train_mode)
        self.backbone = ResNet(train_mode,
                               num_layers=num_layers,
                               last_conv_stride=last_conv_stride,
                               non_local_type=non_local_type,
                               non_local_block=non_local_block,
                               compression=compression
                               )

        self.local_conv_out_channels = local_conv_out_channels

        assert triplet_type in ['none', 'whole', 'stripe']
        self.triplet_type = triplet_type
        self.dropout_place = dropout_place
        self.loss_type = loss_type
        self.pooling_types = pooling_types
        self.margin = margin
        self.with_pair_triplet = with_pair_triplet
        self.alpha = alpha

    def get_pair_feature_repeat(self, global_feat):
        """"""
        feat_shape = global_feat.get_shape().as_list()
        if feat_shape[0] is None:
            feat_shape[0] = tf.shape(global_feat)[0]

        global_feat0 = tf.tile(global_feat, [feat_shape[0], 1, 1, 1])
        global_feat1 = tf.expand_dims(global_feat, 1)
        global_feat1 = tf.tile(global_feat1, [1, feat_shape[0], 1, 1, 1])
        new_shape = feat_shape.copy()
        new_shape[0] = feat_shape[0] * feat_shape[0]
        global_feat1 = tf.reshape(global_feat1, new_shape)

        global_feat_pair = tf.concat([global_feat0, global_feat1], axis=2)

        return global_feat_pair

    def get_pair_label_repeat(self, samples_label):
        label_shape = samples_label.get_shape().as_list()
        if label_shape[0] is None:
            label_shape[0] = tf.shape(samples_label)[0]
        samples_label0 = tf.tile(samples_label, [label_shape[0]])
        samples_label1 = tf.expand_dims(samples_label, 1)
        samples_label1 = tf.tile(samples_label1, [1, label_shape[0]])
        samples_label1 = tf.reshape(samples_label1, [label_shape[0] * label_shape[0]])
        return samples_label0, samples_label1

    def forward(self, x):
        """"""
        backbone_endpoints = self.backbone.forward(x)
        self.net = backbone_endpoints['stage_4']
        global_feat = self.net

        feat_shape = global_feat.get_shape().as_list()
        print(feat_shape)

        if feat_shape[0] is None:
            feat_shape[0] = tf.shape(global_feat)[0]

        global_feat_pair = self.get_pair_feature_repeat(global_feat)

        global_feat_pair = self._pair_graph_(global_feat_pair)

        global_feat0 = global_feat_pair[:, :, 0:feat_shape[2], :]
        global_feat1 = global_feat_pair[:, :, feat_shape[2]:, :]

        with tf.variable_scope("head", reuse=tf.AUTO_REUSE) as scope:
            triplet_logits, softmax_logits = self.head(global_feat)
            triplet_logits0, softmax_logits0 = self.head(global_feat0)
            triplet_logits1, softmax_logits1 = self.head(global_feat1)

        if self.train_mode:
            return softmax_logits, triplet_logits, softmax_logits0, softmax_logits1, triplet_logits0, triplet_logits1
        else:
            return softmax_logits, triplet_logits

    def head(self, global_feat):
        triplet_logits = []
        local_softmax_logits = []
        with tf.variable_scope('local_conv_list'):
            if self.pooling_types == 'baseline':
                feat_avg = tf.reduce_mean(global_feat, axis=[1, 2], keepdims=True)
                feat_max = tf.reduce_max(global_feat, axis=[1, 2], keepdims=True)
                local_feat = tf.concat([feat_avg, feat_max], 3)
            elif self.pooling_types == 'gemm':
                local_feat = tf.pow(
                    tf.reduce_mean(
                        tf.pow(tf.maximum(global_feat, 1e-6), 3),
                        [1, 2], keepdims=True), 1. / 3)
            # if self.local_conv_out_channels:
            #     local_feat = self.batch_norm(local_feat, scope='pool')
            if self.dropout_place == 'before':
                local_feat = self.drop_out(local_feat, 0.5)
            if self.local_conv_out_channels:
                local_feat = self.convolve(local_feat,
                                           channel_out=self.local_conv_out_channels,
                                           ksize=1,
                                           stride=1,
                                           scope='conv')
            local_feat = self.batch_norm(local_feat, scope='bn')
            local_feat = tf.layers.flatten(local_feat)
            if self.triplet_type == 'stripe':
                triplet_logits.append(tf.nn.l2_normalize(local_feat, axis=1))
            if self.train_mode:
                local_feat = tf.nn.relu(local_feat)
            else:
                local_feat = tf.nn.l2_normalize(local_feat, axis=1)
            local_softmax_logits.append(local_feat)

        return triplet_logits, local_softmax_logits

    def loss(self, images, labels, num_classes, scope=None):
        """"""
        labels = labels['label']
        if isinstance(images, list):
            samples = tf.concat(images, 0)
            samples_label = tf.concat(labels, 0)
            samples_label = tf.cast(samples_label, tf.int64)
        else:
            samples = images
            samples_label = labels

        samples_label0, samples_label1 = self.get_pair_label_repeat(samples_label)

        all_class = sum(num_classes)
        print('all_class: ', all_class)
        softmax_logits, triplet_logits, softmax_logits0, softmax_logits1, triplet_logits0, triplet_logits1 = self.forward(samples)

        # softmax
        losses = []
        losses_names = []
        global_logits = []
        local_logits = []

        with tf.variable_scope('fc_list'):
            if self.dropout_place == 'after':
                logits = self.drop_out(softmax_logits[0], 0.5)
            else:
                logits = softmax_logits[0]
            kernel_initializer = tf.random_normal_initializer(stddev=0.001)
            logits = self.fullconnect(logits,
                                      all_class,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=False
                                      )
            local_logits.append(logits)
            loss = tf.losses.sparse_softmax_cross_entropy(samples_label, logits)
            losses_names.append('softmax_0_loss')
            losses.append(loss)

            # tf.add_to_collectiontion('loss_value', loss)

        with tf.variable_scope("fc_pair", reuse=tf.AUTO_REUSE) as scope:
            losses0, losses_name0 = self.fc_pair(softmax_logits0, samples_label0, samples_label1, all_class)
            losses1, losses_name1 = self.fc_pair(softmax_logits1, samples_label1, samples_label0, all_class)
            losses += losses0
            losses += losses1
            losses_names += losses_name0
            losses_names += losses_name1

        if self.triplet_type != 'none':
            loss = batch_hard_triplet_loss(triplet_logits[0], samples_label, self.margin)
            losses_names.append('triplet_loss')
            losses.append(loss)

            if self.with_pair_triplet:
                loss = self.pairwise_hard_triplet_loss(triplet_logits0[0], triplet_logits1[0], samples_label, self.margin)
                losses_names.append('pairwise_triplet_loss')
                losses.append(loss)

        if self.train_mode:
            return losses, losses_names
        else:
            return global_logits, local_logits

    def fc_pair(self, softmax_logits, samples_label0, samples_label1, all_class):
        losses_names = []
        losses = []
        with tf.variable_scope('pair_fc_list'):
            if self.dropout_place == 'after':
                logits = self.drop_out(softmax_logits[0], 0.5)
            else:
                logits = softmax_logits[0]
            kernel_initializer = tf.random_normal_initializer(stddev=0.001)
            logits = self.fullconnect(logits,
                                      all_class,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=False
                                      )
            loss0 = tf.losses.sparse_softmax_cross_entropy(samples_label0, logits)
            loss1 = tf.losses.sparse_softmax_cross_entropy(samples_label1, logits)
            loss = self.alpha * loss0 + (1 - self.alpha) * loss1
            losses_names.append('pair_softmax_loss')

            losses.append(loss)
            # tf.add_to_collection('loss_value', loss)
        return losses, losses_names


    def pairwise_hard_triplet_loss(self, embeddings0, embeddings1, labels, margin=None):
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = tf.sqrt(tf.reduce_sum(tf.square(embeddings0 - embeddings1), axis=-1) + 1e-12)
        pairwise_dist = tf.reshape(pairwise_dist, [labels.get_shape().as_list()[0], labels.get_shape().as_list()[0]])
        loss = batch_hard_triplet_loss(pairwise_dist, labels, margin=margin, compute_distance=False)

        return loss


