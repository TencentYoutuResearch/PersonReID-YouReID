#conding=utf-8

import tensorflow as tf
from nets.core.basciModel import BasicModel

blocks = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}

class ResNet(BasicModel):
    def __init__(self, train_mode, num_layers=101, last_conv_stride=2,
                 ngroup=1,
                 non_local_type='none', non_local_block=2, compression=2):
        super().__init__(train_mode)

        assert num_layers in [50, 101, 152]
        self.last_conv_stride = last_conv_stride

        self.strides = [1, 2, 2, last_conv_stride]
        self.dims = [256, 512, 1024, 2048]
        self.num_blocks = blocks[num_layers]
        self.non_local_type = non_local_type
        self.non_local_block = non_local_block
        self.compression = compression
        self.ngroup = ngroup

    def forward(self, x):
        endpoints = {}
        endpoints['input'] = x
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]])
        x = self.convolve(x,
                          channel_out=64,
                          ksize=7,
                          stride=2,
                          withbias=False,
                          withpad=False,
                          scope='conv1')
        endpoints['conv1'] = x
        x = self.batch_norm(x, scope='bn1')
        endpoints['bn1'] = x
        x = tf.nn.relu(x)
        endpoints['relu'] = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.max_pool(x, ksize=3, stride=2, withpad=False)

        for idx, (num_layer, stride, dim) in enumerate(zip(self.num_blocks, self.strides, self.dims)):
            if self.ngroup == 1 or idx < 3:
                bottleneck_fn = self.bottleneck_v1
                scope = 'layer%d' % (idx+1) #if idx <=2 else 'new_layer%d' % (idx+1)
            else:
                bottleneck_fn = self.bottleneck_v2
                scope = 'next_layer%d' % (idx + 1)
            x = self._make_layer(x, num_layer, stride, dim,
                                 bottleneck_fn=bottleneck_fn,
                                 scope=scope)
            endpoints['stage_%d' % (idx+1)] = x
        return endpoints

    def _make_layer(self, x, num_layers, block_stride,
                    channel_out, bottleneck_fn, scope=None):
        with tf.variable_scope(scope):
            for i in range(num_layers):
                x = bottleneck_fn(x, block_stride if i == 0 else 1, channel_out,
                                       scope=str(i), only_one_block_each_stage_non_local= (i == num_layers-2))
        return x

    def bottleneck_v1(self, x, block_stride, channel_out, scope=None,
                      only_one_block_each_stage_non_local=False):
        channel_in = x.get_shape().dims[-1].value
        chn_bottleneck = int(channel_out // 4)
        shortcut = x
        with tf.variable_scope(scope):
            if block_stride != 1 or channel_in != channel_out:
                with tf.variable_scope('downsample'):
                    shortcut = self.convolve(shortcut,
                                      channel_out=channel_out,
                                      ksize=1,
                                      stride=block_stride,
                                      withbias=False,
                                      withpad=False,
                                      scope='0')
                    shortcut = self.batch_norm(shortcut, scope='1')

            x = self.convolve(x,
                              channel_out=chn_bottleneck,
                              ksize=1,
                              stride=1,
                              withbias=False,
                              scope='conv1')
            x = self.batch_norm(x, scope='bn1')
            x = tf.nn.relu(x)
            # 2
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
            x = self.convolve(x,
                              channel_out=chn_bottleneck,
                              ksize=3,
                              stride=block_stride,
                              withbias=False,
                              withpad=False,
                              scope='conv2')
            x = self.batch_norm(x, scope='bn2')
            x = tf.nn.relu(x)
            # 3
            x = self.convolve(x,
                              channel_out=channel_out,
                              ksize=1,
                              stride=1,
                              withbias=False,
                              scope='conv3')
            x = self.batch_norm(x, scope='bn3')

            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)

            if self.non_local_type in ['single', 'double'] and only_one_block_each_stage_non_local:
                if self.non_local_block == 2:
                    channel_f = 1024
                elif self.non_local_block == 3:
                    channel_f = 512
                if channel_out < channel_f:
                    return x
                if self.non_local_type == 'single':
                    x = self._non_local_block(x, compression=self.compression)
                else:
                    b, h, w, c = x.get_shape().as_list()
                    x = tf.reshape(x, shape=(b //2, 2 * h, w, c))
                    # x = tf.concat([x[:, 0, :, :, :], x[:, 1, :, :, :]], axis=1)
                    x = self._non_local_block(x, compression=self.compression)
                    x = tf.reshape(x, shape=(b, h, w, c))
        return x




