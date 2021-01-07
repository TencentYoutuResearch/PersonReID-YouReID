from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from core.config import config

class BaseCNN:
    """Base class for CNN model.

    """
    def __init__(self, is_train, data_format='NHWC'):
        """Create a ``BaseCNN`` object.

        Args:
            is_train (``bool``): Is training or not. 

        """
        self.train_mode = is_train
        assert data_format in ['NHWC', 'NCHW']
        self.data_format = 'channels_last' if data_format == 'NHWC' else 'channels_first'
        self.conv_regular = self.get_conv_regular()
        self.fc_regular = self.get_fc_regular()

    def get_conv_regular(self):
        """Create regularizer for all convolution layers.

        Subclasses can override this method to use customized regularizer.

        Return:
            A regularizer. Defaults to l2_regularizer with weight_decay = 0.001.
            Return ``None`` to disable regularization.
        """
        weight_decay = 0.0005
        if config.get('weight_decay'):
            weight_decay = config.get('weight_decay')
        return layers.l2_regularizer(weight_decay)

    def get_fc_regular(self):
        """Create regularizer for all fullconnect layers.

        Subclasses can override this method to use customized regularizer.

        Return:
            A regularizer. Defaults to l2_regularizer with weight_decay = 0.001.
            Return ``None`` to disable regularization.
        """
        weight_decay = 0.0005
        if config.get('weight_decay'):
            weight_decay = config.get('weight_decay')
        return layers.l2_regularizer(weight_decay)

    def convolve(self, x, channel_out, ksize, stride=1, withpad=True, withbias=False,
                 bias_init=None, kernel_init=None,
                 scope=None):
        """Create convolution layer.
        """
        pad_method = 'SAME' if withpad else 'VALID'
        if withbias:
            if bias_init:
                bias_init = bias_init
            else:
                bias_init = tf.zeros_initializer
        else:
            bias_init = None

        with tf.variable_scope(scope, 'conv', custom_getter=self.custom_getter) as sc:
            y = tf.layers.conv2d(x,
                        filters=channel_out,
                        kernel_size=ksize,
                        strides=stride,
                        padding=pad_method,
                        data_format=self.data_format,
                        kernel_regularizer = self.conv_regular,
                        use_bias = withbias,
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        name=sc) #scope
        return y

    def fullconnect(self, x, num_units_out,
                    kernel_initializer=None,
                    use_bias=True,
                    scope=None):
        """Create fullconnect layer.
        """
        if x.get_shape().ndims > 2:
            x = tf.layers.flatten(x)

        y = tf.layers.dense(x,
                units=num_units_out,
                activation=None,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=self.fc_regular,
                name=scope)

        return y

    def batch_norm(self, x, center=True, scope=None):
        """Create batch normalization layer.
        """
        #print('cnn.py bn')
        with tf.variable_scope(scope, 'bn', custom_getter=self.custom_getter) as sc:
            y = tf.layers.batch_normalization(x,
                    axis=-1 if self.data_format == 'channels_last' else 1,
                    momentum=0.997,
                    center=center,
                    scale=True,
                    epsilon=1e-5,
                    training=self.train_mode,
                    fused=self.train_mode,
                    name=sc)

        return y

    def max_pool(self, x, ksize, stride, withpad=True):
        """Create max pooling layer.
        """
        pad_method = 'SAME' if withpad else 'VALID'
        return tf.layers.max_pooling2d(x,
                                       data_format=self.data_format,
                                       pool_size=ksize,
                                       strides=stride,
                                       padding=pad_method)

    def drop_out(self, x, keep_prob):
        """Create drop out layer.
        """
        if self.train_mode:
            return tf.nn.dropout(x, keep_prob=keep_prob)
        else:
            return x


    def custom_getter(self, getter, name, *args, **kwargs):
        rename_quan={
                'bn_quantized':'batch_normalization',
                'conv_quantized':'conv',
            }

        short_name = name.split('/')[-2]
        if rename_quan and short_name in rename_quan:
            name_components = name.split('/')
            name_components[-2] = rename_quan[short_name]
            name = '/'.join(name_components)

        return getter(name, *args, **kwargs)


BasicCNN = BaseCNN

