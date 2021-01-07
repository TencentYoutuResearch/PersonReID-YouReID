#conding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from val.config import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])
import tensorflow as tf
import pickle as pkl
import numpy as np
from val.metric import load_feat, mean_ap, cmc, compute_dist, print_scores
import shutil
import cv2

def preprocess(img_file):
    """"""
    file_reader = tf.read_file(img_file)
    image = tf.image.decode_jpeg(file_reader, channels=3, dct_method='INTEGER_ACCURATE')
    height = 384
    if config.get('height'):
        height = config.get('height')
    image = tf.image.resize_images(image, [height, 128])

    # im_mean = [0.486, 0.459, 0.408]
    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]

    image = tf.divide(image, 255)
    image = tf.subtract(image, im_mean)
    image = tf.divide(image, im_std)

    return image

def gen_dataset(imglist, batch_size, get_next=True):
    """"""
    imglist = np.array(imglist)
    ds = tf.data.Dataset.from_tensor_slices(imglist)
    ds = ds.apply(
        tf.contrib.data.map_and_batch(
            map_func=preprocess,
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False))
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
    if get_next:
        return ds.make_one_shot_iterator().get_next()
    else:
        return ds.make_one_shot_iterator()

def get_val_data(dataset, batch_size):
    """"""
    partition = pkl.load(open(os.path.join(dataset, 'partitions.pkl'), 'rb'))
    img_list = partition['test_im_names']
    test_marks = partition['test_marks']
    q_list = []
    g_list = []
    print(img_list[0], test_marks[0])
    for im_name, test_mark in zip(img_list, test_marks):
        if test_mark == 0:
            q_list.append(os.path.join(dataset, 'images', im_name))
        else:
            g_list.append(os.path.join(dataset, 'images', im_name))
    query_data = gen_dataset(q_list, batch_size)
    gallery_data = gen_dataset(g_list, batch_size)

    return (q_list, query_data), (g_list, gallery_data)

def _write_feature(out_obj, file_name, fea_vec):
    out_obj.write(file_name)
    for x in np.nditer(fea_vec, order='C'):
        out_obj.write(' {0:f}'.format(float(x)))
    out_obj.write('\n')

def write_batch_feature(out_obj,batch_filename,features):
    features_list = features.tolist()
    for file_name,fea_vec in zip(batch_filename, features_list):
        fea_vec_array=np.array(fea_vec)
        _write_feature(out_obj, file_name, fea_vec_array)

def get_features(graph_file, dataset, batch_size=32, use_old_name=False):
    """"""
    g = tf.Graph()
    with g.as_default():
        graphdef = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as pb_f:
            graphdef.ParseFromString(pb_f.read())
        with tf.device('/gpu:0'):
            tf.import_graph_def(graphdef, name='')
    input_tensor = g.get_tensor_by_name('graph_input:0')
    output_tesnor = g.get_tensor_by_name('output_features:0')
    if use_old_name:
        output_tesnor = g.get_tensor_by_name('local_conv_list/0/flatten/Reshape:0')

    sconf = tf.ConfigProto(intra_op_parallelism_threads=10,
                           inter_op_parallelism_threads=10)

    pb_name = os.path.dirname(graph_file)

    with tf.Session(graph=g, config=sconf) as sess:
        query, gallery = get_val_data(dataset, batch_size)
        for i, (lines, data) in enumerate([query, gallery]):
            batch_filename = []
            batch_filename_ori = []
            if i == 0:
                file_name_suf = 'Reid_query_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
            else:
                file_name_suf = 'Reid_gallery_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
            with open('%s/%s.result' % (pb_name, file_name_suf), 'w') as fw:
                for i in range(len(lines)):
                    filename_ori = lines[i].strip().split(' ')[0]
                    filename = os.path.join(dataset, 'images', filename_ori)

                    if i == 0:
                        batch_filename_ori.append(filename_ori)
                        batch_filename.append(filename)
                        continue

                    if i % batch_size == 0:
                        print(i)
                        batch_img_data = sess.run(data)
                        features = sess.run(output_tesnor, feed_dict={input_tensor: batch_img_data})
                        if config.get('flip'):
                            batch_img_data_ = np.flip(batch_img_data, axis=2)
                            features_ = sess.run(output_tesnor, feed_dict={input_tensor: batch_img_data_})
                            features = (features + features_) / 2
                        write_batch_feature(fw, batch_filename_ori, features)
                        batch_filename = [filename]
                        batch_filename_ori = [filename_ori]
                    else:
                        batch_filename.append(filename)
                        batch_filename_ori.append(filename_ori)

                print(i)
                batch_img_data = sess.run(data)
                features = sess.run(output_tesnor, feed_dict={input_tensor: batch_img_data})
                if config.get('flip'):
                    batch_img_data_ = np.flip(batch_img_data, axis=2)
                    features_ = sess.run(output_tesnor, feed_dict={input_tensor: batch_img_data_})
                    features = (features + features_) / 2
                write_batch_feature(fw, batch_filename_ori, features)

                print('%s finished!' % file_name_suf)

class TestSet(object):
    def __init__(self, query_feat_path, gallery_feat_path):
        self.query_feat_path = query_feat_path
        self.gallery_feat_path = gallery_feat_path

        self.query_ids = None # ndarray
        self.gallery_ids = None # ndarray
        self.query_cams = None  # ndarray
        self.gallery_cams = None  # ndarray

    def load_data(self):
        # load q feat
        q_feat, q_ids, q_cams, q_paths = load_feat(self.query_feat_path)
        # load g feat
        g_feat, g_ids, g_cams, g_paths = load_feat(self.gallery_feat_path)

        self.query_ids = q_ids
        self.gallery_ids = g_ids
        self.query_cams = q_cams
        self.gallery_cams = g_cams
        self.q_paths = q_paths
        self.g_paths = g_paths

        return q_feat, g_feat

    def compute_score(self, dist_mat):
        # Compute mean AP
        aps, is_valid_query = mean_ap(
            distmat=dist_mat,
            query_ids=self.query_ids, gallery_ids=self.gallery_ids,
            query_cams=self.query_cams, gallery_cams=self.gallery_cams, average=False)
        # if is_valid_query is None:
        mAP = float(np.sum(aps)) / np.sum(is_valid_query)
        # Compute CMC scores
        cmc_scores, ret, is_valid_query = cmc(
            distmat=dist_mat,
            query_ids=self.query_ids, gallery_ids=self.gallery_ids,
            query_cams=self.query_cams, gallery_cams=self.gallery_cams,
            separate_camera_set=False,
            first_match_break=True,
            topk=10)


        return mAP, cmc_scores

    def draw_lines(self, img, color):
        h, w = img.shape[:-1]
        cv2.rectangle(img, (0, 0), (w, 0), color)
        cv2.rectangle(img, (w, 0), (w, h), color)
        cv2.rectangle(img, (0, 0), (0, h), color)
        cv2.rectangle(img, (0, h), (w, h), color)
        return img

    def eval(self, use_rerank=False, save_badcase=False):
        # load feat
        q_feat, g_feat = self.load_data()
        # cal q,g dist
        #q_g_dist = compute_dist(q_feat, g_feat, type='cosine')
        q_g_dist = compute_dist(q_feat, g_feat, type='euclidean')
        if use_rerank:
            from val.rerank import re_ranking
            q_q_dist = compute_dist(q_feat, q_feat, type='euclidean')
            g_g_dist = compute_dist(g_feat, g_feat, type='euclidean')
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = q_g_dist
        mAP, cmc_scores = self.compute_score(dist)

        return mAP, cmc_scores

def remote_val():
    task_id = config.get('pretrain_task')
    step = config.get('pretrain_model_step')
    dataset = config.get('eval_dataset')

    from val.save_pb import get_and_save
    # get_and_save(task_id, step)

    graph_file = os.path.join('pb', task_id, 'save_model-%d.pb' % step)
    # get_features(graph_file=graph_file, dataset=dataset)
    #
    pb_name = os.path.dirname(graph_file)

    query_suf = 'Reid_query_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
    query_feat_path = '%s/%s.result' % (pb_name, query_suf)

    gallery_suf = 'Reid_gallery_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
    gallery_feat_path = '%s/%s.result' % (pb_name, gallery_suf)

    test_set = TestSet(query_feat_path, gallery_feat_path)
    mAP, cmc_scores = test_set.eval()

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)


def local_val(get_pb=True, get_fea=True, use_rerank=False):
    # task_id = config.get('pretrain_task')
    # step = config.get('pretrain_model_step')
    dataset = config.get('eval_dataset')

    from val.save_pb import save_pb

    save_file =  os.path.join(config.get('root_path'), config.get('pretrain_task'), 'save_model.pb')
    if get_pb:
        save_pb(os.path.join(config.get('root_path'),  config.get('pretrain_task'),
                             'model.ckpt-%d' % config.get('pretrain_model_step')),
            save_file
            )
    shutil.copy('eval_config.yaml',
                os.path.join(config.get('root_path'), config.get('pretrain_task'), 'eval_config.yaml'))

    if get_fea:
        get_features(graph_file=save_file, dataset=dataset,
                     batch_size=config.get('batch_size'), use_old_name=config.get('use_old_name'))

    pb_name = os.path.dirname(save_file)

    query_suf = 'Reid_query_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
    query_feat_path = '%s/%s.result' % (pb_name, query_suf)

    gallery_suf = 'Reid_gallery_' + os.path.basename(os.path.normpath(dataset)) + '.txt'
    gallery_feat_path = '%s/%s.result' % (pb_name, gallery_suf)

    test_set = TestSet(query_feat_path, gallery_feat_path)
    mAP, cmc_scores = test_set.eval(use_rerank=use_rerank, save_badcase=config.get('save_badcase'))

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)
    print(config.get('pretrain_task'), config.get('pretrain_model_step'))

if __name__ == '__main__':
    # remote_val()
    local_val(get_pb=config.get('get_pb'),
              get_fea=config.get('get_fea'),
              use_rerank=config.get('use_rerank')
              )