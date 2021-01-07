#conding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from val.config import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])
import tensorflow as tf
from tensorflow.python.framework import graph_util
from nets.factory import select_network


def save_pb(checkpoint, output_file):
    """"""
    model_config = config.get('model_config')
    with tf.Graph().as_default():
        height = 384
        if config.get('height'):
            height = config.get('height')
        inputs = tf.placeholder(dtype=tf.float32,
                             shape=[None, height, 128, 3],
                             name='graph_input')
        network_name = model_config['name']
        del model_config['name']
        model = select_network(network_name)(**model_config)
        outputs = model.forward(inputs)
        if network_name in ['source']:
            dest_node = outputs[0]
            features = tf.nn.l2_normalize(tf.concat(dest_node, axis=-1), axis=-1, name='output_features')
            print('features: ', features.get_shape())
            out_names = [features.op.name]
        elif network_name in ['cacenet']:
            features = tf.nn.l2_normalize(outputs[1][0], axis=-1, name='output_features')
            out_names = [features.op.name]
        elif network_name in ['mgn']:
            dest_node = []
            global_softmax_branches, local_softmax_branches, _, _ = outputs
            for o in global_softmax_branches:
                dest_node.append(tf.nn.l2_normalize(o, axis=-1))
            for o in local_softmax_branches:
                o = tf.nn.l2_normalize(tf.concat(o, axis=-1), axis=-1)
                dest_node.append(o)
            features = tf.nn.l2_normalize(tf.concat(dest_node, axis=-1), axis=-1, name='output_features')
            print('features: ', features.get_shape())
            out_names = [features.op.name]
        # load_vars = tf.global_variables()
        load_vars = {}
        for v in tf.global_variables():
            v_name = v.op.name
            if 'features' in v_name:
                v_name = 'tower_0/' + v_name
            # if 'non_local_graph/x1/kernel' in v_name:
            #     continue
            load_vars[v_name] = v
        saver = tf.train.Saver(load_vars)

        with tf.Session() as sess:
            print('load checkpoint from %s' % checkpoint)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint)

            cur_graphdef = sess.graph.as_graph_def()

            output_graphdef = graph_util.convert_variables_to_constants(sess,
                                    cur_graphdef,
                                    out_names)

            with tf.gfile.GFile(output_file, 'wb') as gf:
                gf.write(output_graphdef.SerializeToString())



def get_checkpoint(task_id, step):
    """"""
    if not os.path.exists(os.path.join('checkpoints', task_id)):
        os.makedirs(os.path.join('checkpoints', task_id))
    if not os.path.exists(os.path.join('checkpoints', task_id, 'model.ckpt-%d.index' % step)):
        src = '/train_task/%s/model.ckpt-%d' % (task_id, step)
        cmd1 = '~/rapidflow-client/cmd -cos get %s.index checkpoints/%s' % (src, task_id)
        print(cmd1)
        os.system(cmd1)
        cmd2 = '~/rapidflow-client/cmd -cos get %s.data-00000-of-00001 checkpoints/%s' % (src, task_id)
        print(cmd2)
        os.system(cmd2)

    if not os.path.exists(os.path.join('pb', task_id)):
        os.makedirs(os.path.join('pb', task_id))


def get_and_save(task_id, step):
    """"""
    get_checkpoint(task_id, step)
    checnpoint = os.path.join('checkpoints', task_id, 'model.ckpt-%d' % step)
    save_pb(checnpoint, os.path.join('pb', task_id, 'save_model-%d.pb' % step))
    import numpy as np
    os.chdir()


