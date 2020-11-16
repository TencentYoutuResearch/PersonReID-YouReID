import struct
import cv2
import numpy as np
from .tfrecord_utils import yt_example_pb2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import utils.my_transforms as my_transforms
import os


class TFRecordDataset(data.Dataset):
    def __init__(self, mode, names, size=(384, 128), prng=np.random):
        self.names = names
        self.tfrecord_root = '/raid/home/fufuyu/dataset/'
        self.mode = mode
        self.getFileList()  #self.filelist, self.labellist =

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size),
                transforms.Pad(10),
                transforms.RandomCrop(size),
                # transforms.RandomRotation(20),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1), #, saturation=0.2
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                my_transforms.RandomErasing(mean=[0.485, 0.456, 0.406]
                                            ),

            ])


    def getFileList(self):
        self.filelist, self.labellist = [], []
        self.label_offset = 0
        for idx in self.names:
            index_filepath = os.path.join(self.tfrecord_root, idx, idx + '.txt')
            with open(index_filepath, "r") as idx_r:
                for line in idx_r:
                    data_name, tf_num, offset, label = line.rstrip().split('\t')[:4]
                    # file_name = '{0}*{1:05}*{2}'.format(data_name, int(tf_num), offset)
                    file_name = (data_name, str(tf_num).zfill(5), offset)
                    self.filelist.append(file_name)
                    label = int(label) + self.label_offset
                    self.labellist.append(label)
            self.label_offset = max(self.labellist)


    def get_file_loc(self, index):
        return self.filelist[index]

    def get_label(self, index):
        return self.labellist[index]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        file_loc, label = self.get_file_loc(index), self.get_label(index)
        src_img = self.getImageData(file_loc)
        img = cv2.imdecode(np.asarray(bytearray(src_img), dtype=np.uint8), 1)
        img = img[:, :, ::-1]
        img, mirrored = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.filelist)

    def getImageData(self, file_loc):
        data_name, tf_num, offset = file_loc
        tf_file = self.tfrecord_root + "/remote_tfrecord" + "/" + data_name + "/" + data_name + "-" + tf_num + ".tfrecord"

        with open(tf_file, 'rb') as tf:
            tf.seek(int(offset))
            pb_len_bytes = tf.read(8)
            if len(pb_len_bytes) < 8:
                print("read pb_len_bytes err,len(pb_len_bytes)=" +
                      str(len(pb_len_bytes)))
                return None

            pb_len = struct.unpack('L', pb_len_bytes)[0]

            len_crc_bytes = tf.read(4)
            if len(len_crc_bytes) < 4:
                print("read len_crc_bytes err,len(len_crc_bytes)=" +
                      str(len(len_crc_bytes)))
                return None

            len_crc = struct.unpack('I', len_crc_bytes)[0]

            pb_data = tf.read(pb_len)
            if len(pb_data) < pb_len:
                print("read pb_data err,len(pb_data)=" + str(len(pb_data)))
                return None

            data_crc_bytes = tf.read(4)
            if len(data_crc_bytes) < 4:
                print("read data_crc_bytes err,len(data_crc_bytes)=" +
                      str(len(data_crc_bytes)))
                return None

            data_crc = struct.unpack('I', data_crc_bytes)[0]

            example = yt_example_pb2.Example()
            example.ParseFromString(pb_data)

            image_data_feature = example.features.feature.get("image")
            label_feature = example.features.feature.get("label")

            if image_data_feature:
                image_data = image_data_feature.bytes_list.value[0]
                return image_data


# class TFRecordIterableDataset(data.IterableDataset):
#     def __init__(self, names, pre_process_im_kwargs, epoch=0, prng=np.random, seed=0):
#         self.names = names
#         self.tfrecord_root = '/youtu-reid/'
#         #self.filelist, self.labellist =
#         self.pre_process_im = PreProcessIm(prng=prng, **pre_process_im_kwargs)
#         self.epoch = epoch
#         self.seed =seed
#         self.get_length()
#
#     def get_length(self):
#         self.length = 0
#         for idx in self.names:
#             num_lines = sum(1 for _ in open(self.tfrecord_root + "/train_data/" + idx + '.txt'))
#             self.length += num_lines
#
#     def __len__(self):
#         return self.length
#
#     def __iter__(self):
#
#         worker_info = data.get_worker_info()
#         world_size = int(os.environ['WORLD_SIZE'])
#         rank = int(os.environ['RANK'])
#         num_worker = worker_info.num_workers
#         worker_id = worker_info.id
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#         indices = torch.randperm(world_size*num_worker, generator=g).tolist()
#
#         local_indices = indices[rank * num_worker + worker_id]
#         self.label_offset = 0
#         for idn, idx in enumerate(self.names):
#             index_filepath = self.tfrecord_root + "/train_data/" + idx + '.txt'
#             if idn > 0:
#                 self.label_offset = max(self.label_offset, label)
#             with open(index_filepath, "r") as idx_r:
#                 for line_i, line in enumerate(idx_r):
#                     data_name, tf_num, offset, label = line.rstrip().split('\t')[:4]
#                     label = int(label) + self.label_offset
#                     if line_i % (world_size*num_worker) == local_indices:
#                         file_name = (data_name, str(tf_num).zfill(5), offset)
#                         src_img = self.getImageData(file_name)
#                         img = cv2.imdecode(np.asarray(bytearray(src_img), dtype=np.uint8), 1)
#                         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                         img, mirrored = self.pre_process_im(img)
#                         yield img, label
#
#
#     def getImageData(self, file_loc):
#         data_name, tf_num, offset = file_loc
#         tf_file = self.tfrecord_root + "/remote_tfrecord" + "/" + data_name + "/" + data_name + "-" + tf_num + ".tfrecord"
#
#         with open(tf_file, 'rb') as tf:
#             tf.seek(int(offset))
#             pb_len_bytes = tf.read(8)
#             if len(pb_len_bytes) < 8:
#                 print("read pb_len_bytes err,len(pb_len_bytes)=" +
#                       str(len(pb_len_bytes)))
#                 return None
#
#             pb_len = struct.unpack('L', pb_len_bytes)[0]
#
#             len_crc_bytes = tf.read(4)
#             if len(len_crc_bytes) < 4:
#                 print("read len_crc_bytes err,len(len_crc_bytes)=" +
#                       str(len(len_crc_bytes)))
#                 return None
#
#             len_crc = struct.unpack('I', len_crc_bytes)[0]
#
#             pb_data = tf.read(pb_len)
#             if len(pb_data) < pb_len:
#                 print("read pb_data err,len(pb_data)=" + str(len(pb_data)))
#                 return None
#
#             data_crc_bytes = tf.read(4)
#             if len(data_crc_bytes) < 4:
#                 print("read data_crc_bytes err,len(data_crc_bytes)=" +
#                       str(len(data_crc_bytes)))
#                 return None
#
#             data_crc = struct.unpack('I', data_crc_bytes)[0]
#
#             example = yt_example_pb2.Example()
#             example.ParseFromString(pb_data)
#
#             image_data_feature = example.features.feature.get("image")
#             label_feature = example.features.feature.get("label")
#
#             if image_data_feature:
#                 image_data = image_data_feature.bytes_list.value[0]
#                 return image_data
