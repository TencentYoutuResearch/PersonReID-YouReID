#conding=utf-8
# @Time  : 2019/9/29 20:04
# @Author: fufuyu
# @Email:  fufuyu@tencen.com


import torch
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np

class Resnet:
    def __init__(self):
        pass

    def forward(self, x):
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]])
        x = tf.layers.conv2d(x,
                          64,
                          kernel_size=7,
                          strides=2,
                          padding='SAME',
                          use_bias=False,
                          name='conv1')
        x = tf.layers.batch_normalization(x, name='bn1')
        x = tf.nn.relu(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='SAME')
        x = self._make_layer(x, 3, 1, 256, scope='layer1')
        x = self._make_layer(x, 4, 2, 512, scope='layer2')
        x = self._make_layer(x, 6, 2, 1024, scope='layer3')
        x = self._make_layer(x, 3, 2, 2048, scope='layer4')

        return x

    def _make_layer(self, x, num_layers, block_stride, channel_out, scope=None):
        with tf.variable_scope(scope):
            for i in range(num_layers):
                x = self.bottleneck_v1(x, block_stride if i==0 else 1, channel_out, scope=str(i))
        return x


    def bottleneck_v1(self, x, block_stride, channel_out, scope=None):
        channel_in = x.get_shape().dims[-1].value
        chn_bottleneck = int(channel_out // 4)
        shortcut = x
        with tf.variable_scope(scope):
            if block_stride != 1 or channel_in != channel_out:
                with tf.variable_scope('downsample'):
                    shortcut = tf.layers.conv2d(shortcut, channel_out, kernel_size=1,
                                                use_bias=False,
                                                strides=block_stride, padding='VALID',name='0')
                    shortcut =tf.layers.batch_normalization(shortcut, name='1')

            x = tf.layers.conv2d(x,
                              chn_bottleneck,
                              kernel_size=1,
                              strides=1,
                                 use_bias=False,
                              name='conv1')
            x = tf.layers.batch_normalization(x, name='bn1')
            x = tf.nn.relu(x)
            # 2
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
            x = tf.layers.conv2d(x,
                              chn_bottleneck,
                              kernel_size=3,
                              strides=block_stride,
                              padding='VALID',
                              use_bias=False,
                              name='conv2')
            x = tf.layers.batch_normalization(x, name='bn2')
            x = tf.nn.relu(x)
            # 3
            x = tf.layers.conv2d(x,
                              channel_out,
                              kernel_size=1,
                              strides=1,
                              use_bias=False,
                              name='conv3')
            x = tf.layers.batch_normalization(x, name='bn3')

            x = tf.add(x, shortcut, name='residual')
            x = tf.nn.relu(x, name='block_relu')

        return x

def convert(var_list, torch_dcit):
    ops = []
    words = [('/', '.'), ('kernel', 'weight'), ('gamma', 'weight'), ('beta', 'bias'), ('moving_mean', 'running_mean'),
             ('moving_variance', 'running_var')]
    for v in var_list:
        name = v.op.name
        dst_name = name
        for w in words:
            dst_name = dst_name.replace(w[0], w[1])
        if dst_name not in torch_dcit:
            print(name, dst_name)
            break
        if isinstance(torch_dcit[dst_name], torch.nn.parameter.Parameter):
            dst = torch_dcit[dst_name].data.numpy()
        else:
            dst = torch_dcit[dst_name].numpy()
        if dst.ndim == 4:
            dst = np.transpose(dst, axes=(2, 3, 1, 0))
        elif dst.ndim == 2:
            dst = np.transpose(dst)
        op = v.assign(dst)
        ops.append(op)
    return ops




model = Resnet()
input_tensor = tf.placeholder(tf.float32, shape=(1, 384, 128, 3))
_ = model.forward(input_tensor)

res = torch.load('resnet50-19c8e357.pth')
res_np = {}
# for name in res:
#     if isinstance(res[name], torch.nn.parameter.Parameter):
#         res_np[name] = res[name].data.numpy()
#     else:
#         res_np[name] = res[name].numpy()
# conv1_weight =  res['conv1.weight']
# print(res)
vars = tf.global_variables()
#
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ops = convert(vars, res)
    sess.run(ops)
    saver.save(sess, 'resnet50.ckpt', global_step=0)
