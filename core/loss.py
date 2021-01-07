#conding=utf-8

import tensorflow as tf

def _pairwise_distances(embeddings, type="euclid"):
    """计算特征距离
    embeddings: tensor of shape (batch_size, embed_dim) L2范数归一化过的
    type："euclid" 计算方式， 有euclid 和 cosine 两种
    对于L2范数归一化过的embeddings， euclid 和 cosine等价
    """
    assert type in ['euclid', 'cosine']
    if type == 'euclid':
        a = tf.expand_dims(embeddings, axis=1)  # (b, 1, dim)
        b = tf.expand_dims(embeddings, axis=0)  # (1, b, dim)
        dist = tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + 1e-12)
    else:
        dist = 1- tf.matmul(embeddings, embeddings, transpose_b=True)
    return dist


def batch_hard_triplet_loss(embeddings_or_dist, labels, margin=None, compute_distance=True):
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
    if compute_distance:
        pairwise_dist = _pairwise_distances(embeddings_or_dist)  # batch_size, batch_size
    else:
        pairwise_dist = embeddings_or_dist
    print('pairwise_dist shape: ', pairwise_dist.get_shape().as_list())
    # labels = tf.Print(labels, [labels], message='debug test labels', summarize=64)

    same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                  tf.expand_dims(labels, axis=0)
                                  )
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    furthest_positive = tf.reduce_max(pairwise_dist * tf.cast(positive_mask, tf.float32), axis=1)
    # L2归一化的 pairwise_dist的最大值的2， 最小值是0
    closest_negative = tf.reduce_min(pairwise_dist*tf.cast(negative_mask, tf.float32) + 1e6 * tf.cast(same_identity_mask, tf.float32), axis=1)

    # furthest_positive = tf.Print(furthest_positive, [furthest_positive], summarize=64)
    # closest_negative = tf.Print(closest_negative, [closest_negative], summarize=64)
    if margin and margin > 0:
        loss = tf.maximum(0.0, furthest_positive + margin - closest_negative)
    else:
        loss = tf.nn.softplus(furthest_positive - closest_negative)
    loss = tf.reduce_mean(loss)

    return loss



