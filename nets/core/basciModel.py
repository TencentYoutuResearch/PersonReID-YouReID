#conding=utf-8

import tensorflow as tf
from nets.core.cnn import BasicCNN

class BasicModel(BasicCNN):
    def __init__(self, is_train, data_format='NHWC'):
        super(BasicModel, self).__init__(is_train, data_format)

    def _non_local_block(self, x, intermediate_dim=None, compression=16, mode='embedded'):
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        b, h, w, c = x.get_shape().as_list()
        if b is None:
            b = tf.shape(x)[0]
        if intermediate_dim is None:
            intermediate_dim = c // compression

        with tf.variable_scope('non_local_embedded'):
            if mode == 'gaussian':
                x1 = tf.reshape(x, shape=(b, -1, c))
                x2 = tf.reshape(x, shape=(b, -1, c))
                x2 = tf.transpose(x2, perm=(0, 2, 1))
                f = tf.matmul(x1, x2)
                f = tf.nn.softmax(f)
            elif mode == 'embedded':
                x1 = self.convolve(x, channel_out=intermediate_dim, ksize=1, scope='x1')
                x2 = self.convolve(x, channel_out=intermediate_dim, ksize=1, scope='x2')
                x1 = tf.reshape(x1, shape=(b, -1, intermediate_dim))
                x2 = tf.reshape(x2, shape=(b, -1, intermediate_dim))
                x2 = tf.transpose(x2, perm=(0, 2, 1))
                f = tf.matmul(x1, x2)
                f = tf.nn.softmax(f)
            g = self.convolve(x, channel_out=intermediate_dim, ksize=1, scope='g')
            g = tf.reshape(g, shape=(b, -1, intermediate_dim))
            o = tf.matmul(f, g)
            o = tf.reshape(o, shape=(b, h, w, intermediate_dim))
            o = self.convolve(o, channel_out=c, ksize=1, kernel_init=tf.constant_initializer(0), scope='o')

        return x + o

    def _pair_graph_(self, x, intermediate_dim=None, compression=16):
        b, h, w, c = x.get_shape().as_list()
        if b is None:
            b = tf.shape(x)[0]
        if intermediate_dim is None:
            intermediate_dim = c // compression

        with tf.variable_scope('pair_graph'):
            x1 = self.convolve(x, channel_out=intermediate_dim, ksize=1, scope='x1')
            x1 = tf.nn.l2_normalize(x1, axis=-1)
            x1 = tf.reshape(x1, shape=(b, -1, intermediate_dim))
            x2 = tf.transpose(x1, perm=(0, 2, 1))
            coef = tf.matmul(x1, x2)
            coef = tf.nn.relu(coef)

            adja = coef * (1. - tf.eye(h*w, batch_shape=[b]))
            degree = tf.reduce_sum(adja, axis=-1, keepdims=True) * tf.eye(h*w, batch_shape=[b])
            norm = tf.stop_gradient(1. / (tf.sqrt(tf.where(degree > 1e-3, degree, tf.ones_like(degree))))) * tf.eye(h*w, batch_shape=[b])

            # f = degree - adja
            f = tf.matmul(tf.matmul(norm, degree - adja), norm) # b, n, n
            g = self.convolve(x, channel_out=intermediate_dim, ksize=1, scope='g')
            g = tf.reshape(g, shape=(b, h*w, intermediate_dim))
            o = tf.matmul(f, g)
            o = tf.reshape(o, shape=(b, h, w, intermediate_dim))
            o = self.convolve(o, channel_out=c, ksize=1, kernel_init=tf.constant_initializer(0), scope='o')

            return x + o


