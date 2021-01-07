#conding=utf-8

import os
import numpy as np
import pickle
from core.config import config
import tensorflow as tf
import copy



class HierDataset:
    def __init__(self, name, sample_processor, use_mask=False):
        """Create a ``TarData`` object. but use pickle instead of txt for pytorch data
        """
        # NOT use index_parser
        self._sample_processor = sample_processor
        self._data_root = config.get('data_root')

        if isinstance(name, str):
            self._index_file = os.path.join(self._data_root, name, 'partitions.pkl')
            self._data_dir = os.path.join(self._data_root, name, 'images/')
        else:
            self._index_file = os.path.join(self._data_root, name[0], 'partitions.pkl')
            self._data_dir = os.path.join(self._data_root, name[1], 'images/')

        self._data_id_map = self.get_id_map(name) # 获得ID融合文件路径
        self.num_images_per_class = config.get('num_images_per_id')
        self.use_mask = use_mask

        print('dataset name', name)
        print('id_map: ', self._data_id_map)
        self.class_num = 0

    def get_id_map(self, name):
        """针对一人多档的情况下， 获得ID融合文件路径"""
        data_config = config.get('data')
        for data in data_config:
            if data.get('name') == name:
                self.num_class_per_batch = data.get('num_id')
                break
        return None

    def load_pickle(self, path):
        """Check and load pickle object.
        According to this post: https://stackoverflow.com/a/41733927, cPickle and
        disabling garbage collector helps with loading speed."""
        # gc.disable()
        with open(path, 'rb') as f:
            ret = pickle.load(f)
        # gc.enable()
        return ret

    def parse_im_name(self, im_name, parse_type='id'):
        """Get the person id or cam from an image name."""
        assert parse_type in ('id', 'cam')
        if parse_type == 'id':
            parsed = int(im_name[:8])
        else:
            parsed = int(im_name[9:13])
        return parsed

    def make_dataset(self):
        label_2_images_src = self._build_label_image_dict()
        gen = self._build_index_inputs(label_2_images_src)
        output_types = [tf.string, tf.int64]
        output_shapes = [tf.TensorShape([]), tf.TensorShape([])]
        if self.use_mask:
            output_types.append(tf.int64)
            output_shapes.append(tf.TensorShape([24, 8]))
        output_types = tuple(output_types)
        output_shapes = tuple(output_shapes)
        self._dataset = tf.data.Dataset.from_generator(lambda : gen
                                              , output_types=output_types,
                                              output_shapes=output_shapes
                                              )
        return self._dataset

    def _init_iterator(self, session, iterator):
        session.run(iterator.initializer)

    def _build_label_image_dict(self):
        """"""
        # TODO load_pickle
        partitions = self.load_pickle(self._index_file)

        part = 'trainval'
        im_names = partitions['{}_im_names'.format(part)]
        ids2labels = partitions['{}_ids2labels'.format(part)]

        # TODO bychenyangguo
        # print (im_names, 'DEBUG')
        current_label = 0
        trainval_ids2labels = {}

        print('current_label begins: {}'.format(current_label))
        for id in ids2labels:
            trainval_ids2labels[id] = current_label
            current_label += 1


        self._sample_num = len(im_names)
        self.class_num = len(set(trainval_ids2labels.values()))

        print('sample_num: %d class_num: %d' % (self._sample_num, self.class_num))

        label_2_images_src = {}
        for line_i, im_name in enumerate(im_names):
            id = self.parse_im_name(im_name, 'id')
            label = trainval_ids2labels[id]
            if label in label_2_images_src:
                label_2_images_src[label].append(im_name)
            else:
                label_2_images_src[label] = [im_name]
        return label_2_images_src


    def _build_index_inputs(self, label_2_images_src):
        """

        :param total: 显卡数目
        :param mine:
        :return:
        """
        np.random.seed(1234) # 固定seed 使得每次函数调用得到的结果一致

        # 过滤少于num_images_per_class的类
        label_2_images_select = {}
        for label in label_2_images_src:
            if len(label_2_images_src[label]) >= self.num_images_per_class:
                label_2_images_select[label] = copy.deepcopy(label_2_images_src[label])
                np.random.shuffle(label_2_images_select[label])

        label_2_images_select_copy = copy.deepcopy(label_2_images_select)
        while True:
            keys = list(label_2_images_select_copy.keys())
            np.random.shuffle(keys)
            select_key = keys  # keys[::total]
            idp = 0
            while idp < int(self._sample_num / self.num_class_per_batch / self.num_images_per_class): # 一个epoch
                batch_key = np.random.choice(select_key, self.num_class_per_batch, replace=False)
                idp += 1
                for sk in batch_key:
                    if len(label_2_images_select_copy[sk]) < self.num_images_per_class:
                        sk_images = copy.deepcopy(label_2_images_select[sk])
                        np.random.shuffle(sk_images)
                        label_2_images_select_copy[sk] = sk_images
                    for i in range(self.num_images_per_class):
                        im_name = label_2_images_select_copy[sk].pop()
                        im_path = self._data_dir + im_name
                        mask_path = im_path.replace('images', 'masks').replace('jpg', 'npy')
                        if self.use_mask:
                            mask = np.load(mask_path).astype(np.int64)
                            yield im_path, sk, mask
                        else:
                            yield im_path, sk



    def base_batch_input(self,
                         batch_size,
                         process_parallel_num=1,
                         prefetch_buffer_size=1):
        """Make batch input of built dataset.

        Note:
            Data pipeline as following:
            shuffle -> repeat -> HOOK(after_shuffle_repeat_prefetch) ->
            sample_parser -> HOOK(after_sample_parser) ->
            filter_batch_after_parser -> HOOK(after_filter_batch_after_parser) -> HOOK(filter_batch_before_processor) ->
            filter_batch_before_processor -> batch

        Args:
            batch_size (``int``): Number of samples in a single batch.
            shuffle (``bool``): Shuffle samples or not.
            epoch_num (``int``, optional): Number of times the elements of this dataset should
                be repeated. The default behavior (if count is None or -1) is for
                the elements to be repeated indefinitely.
            shuffle_buffer_size (``int``, optional): Shuffle buffer size. Defaults to 1024.
            process_parallel_num (``int``, optional): Number of preprocessing threads. Defaults to 10.
            cache (``bool``, optinal): Cache samples in memory or not. Defaults to False.
            prefetch_buffer_size (``int``, optional): Maximum number of processed elements that will be buffered. Defaults to 2048.
            filter_batch_after_parser (``BaseSampleFilter``, optional): Batch filter for parsed samples.
            filter_batch_before_processor (``BaseSampleFilter``, optional): Batch filter for orignal samples.
            padded_shapes (``tf.TensorShape or tf.int64 vector tensor-like objects``): Shape to which the respective component of each input element should be padded prior to batching

        Return:
            self.
        """
        ds = self._dataset

        if self._sample_processor is not None:
            print('using map_and_batch with num_parallel_batches={0}, prefetch_buffer_size={1}'.format(
                process_parallel_num, prefetch_buffer_size))
            ds = ds.apply(
                tf.contrib.data.map_and_batch(
                    map_func=self._sample_processor,
                    batch_size=batch_size,
                    num_parallel_calls=32))
                    #num_parallel_batches=process_parallel_num))
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            # from tensorflow.contrib.data import map_and_batch
        file_names, images, labels = ds.make_one_shot_iterator().get_next()
        file_names.set_shape(shape=(batch_size,))
        images.set_shape(shape=(batch_size, config.get('height'), 128, 3))
        for k in labels:
            k_shape = labels[k].get_shape().as_list()
            k_shape[0] = batch_size
            labels[k].set_shape(shape=tuple(k_shape))
        return file_names, images, labels




