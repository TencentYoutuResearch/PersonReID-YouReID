#conding=utf-8
# @Time  : 2019/12/19 20:08
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import utils.my_transforms as my_transforms
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import time
import scipy.io as sio
import sys
sys.path.append("..")
from utils.iotools import read_image, is_image_file

import numpy as np
from copy import deepcopy
import pickle
import math

def find_classes(config):
    with open(config, 'r') as f:
        lines = f.readlines()
    lines.sort()
    classes = []

    for line in lines:
        filename, cls = line.strip().split(' ')
        if cls not in classes:
            classes.append(cls)

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, config, class_to_idx, classes):

    with open(config, 'r') as f:
        lines = f.readlines()
    lines.sort()
    images = []


    for line in lines:
        filename, c = line.strip().split(' ')
        if is_image_file(filename):
            path = os.path.join(root, filename)
            # print(filename)
            if c in classes:
                item = (path, class_to_idx[c], 0)
                images.append(item)


    return images

def make_gallery(root):
    images = [d for d in os.listdir(root)]
    imgs = []

    for i in images:
        path = os.path.join(root, i)
        # print(filename)

        item = (path, 0, 0)
        imgs.append(item)

    return imgs

def make_query(root, config):
    with open(config, 'r') as f:
        lines = f.readlines()
    images = []

    for line in lines:
        filename, c = line.strip().split(' ')

        path = os.path.join(root, filename)

        item = (path, int(c), 0)
        images.append(item)


    return images

class Pair(data.IterableDataset):
    def __init__(self, root='/data1/home/fufuyu/dataset/',
                 dataname='market1501', part='train',
                 loader=read_image, require_path=False, size=(384,128),
                 least_image_per_class=4, mgn_style_aug=False,
                 load_img_to_cash=False, default_transforms=None, **kwargs):

        self.root = os.path.join(root, dataname)
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.least_image_per_class = least_image_per_class
        self.load_img_to_cash = load_img_to_cash
        self.logger = kwargs.get('logger', print)
        self.mode = kwargs.get('mode', 'train')
        self.return_cam = kwargs.get('return_cam', False)
        self.epoch = kwargs.get('epoch', 0)

        with open(os.path.join(self.root, 'partitions.pkl'), 'rb') as f:
            partitions = pickle.load(f)

        if part == 'train':
            im_names = partitions['trainval_im_names']
            ids2labels = partitions['trainval_ids2labels']

            trainval_ids2labels = {}
            current_label = 0
            for id in ids2labels:
                trainval_ids2labels[id] = current_label
                current_label += 1

            imgs = []
            for line_i, im_name in enumerate(im_names):
                id, cam = self.parse_im_name(im_name)
                label = int(trainval_ids2labels[id])
                imgs.append((os.path.join(self.root, 'images', im_name), label * 10 + cam, cam))

            self.image_dict = self._postprocess(imgs, self.least_image_per_class)
            classes = self.image_dict.keys()
            self.len = sum([math.ceil(float(len(v)) / self.least_image_per_class) * self.least_image_per_class for v in self.image_dict.values()])
        else:
            if len(partitions['test_im_names']) > 0:
                img_list = partitions['test_im_names']
                test_marks = partitions['test_marks']
            else:
                img_list = partitions['val_im_names']
                test_marks = partitions['val_marks']
            q_list = []
            g_list = []
            classes = []
            for im_name, test_mark in zip(img_list, test_marks):
                if test_mark == 0:
                    q_list.append((os.path.join(self.root, 'images', im_name), 0, 0))
                else:
                    g_list.append((os.path.join(self.root, 'images', im_name), 0, 0))
            if part == 'query':
                imgs = q_list
            else:
                imgs = g_list

            if len(imgs) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"))
            self.imgs = imgs
            self.len = len(imgs)

        if default_transforms is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            if part in ['train', 'train_all'] and self.mode == 'train':
                if mgn_style_aug:
                    print('use random padding')
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        my_transforms.RandomPadding(),  #optional
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                else:
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
        else:
            self.transform = default_transforms



        self.classes = classes

        self.class_num = len(classes)

        self.logger('\n')
        self.logger('  **************** Summary ****************')
        self.logger('  #  ids      : {}'.format(self.class_num))
        self.logger('  #  images   : {}'.format(len(imgs)))
        self.logger('  *****************************************')
        self.logger('\n')

    def _postprocess(self, imgs, least_image_per_class=4):
        image_dict = {}
        for p, c ,_ in imgs:
            if c not in image_dict:
                image_dict[c] = [p]
            else:
                image_dict[c].append(p)

        temp = deepcopy(image_dict)

        for k,v in temp.items():
            if len(v) < least_image_per_class:
                image_dict.pop(k)

        new_image_dict = {i: image_dict[k] for i, k in enumerate(list(image_dict.keys()))}

        return new_image_dict


    def parse_im_name(self, im_name):
        """Get the person id or cam from an image name."""
        return int(im_name[:8]), int(im_name[9:13])

    def __iter__(self):

        worker_info = data.get_worker_info()
        # world_size = intintos.environ['WORLD_SIZE'])
        # rank = int(os.environ['RANK'])
        num_worker = worker_info.num_workers
        worker_id = worker_info.id
        np.random.seed(self.epoch)

        pairs = []
        for k, v in self.image_dict.items():
            if len(v) % self.least_image_per_class != 0:
                new_v = v + np.random.choice(v, self.least_image_per_class - len(v) % self.least_image_per_class).tolist()
            else:
                new_v = v
            np.random.shuffle(new_v)
            left = new_v[::2]
            right = new_v[1::2]
            pairs.extend(list(zip(left, right)))
        for p1, p2 in pairs[worker_id::num_worker]:
            img1 = self.loader(p1)
            img1 = self.transform(img1)
            img2 = self.loader(p2)
            img2 = self.transform(img2)
            yield img1, img2



    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is class_index of the target class.
    #     """
    #     if self.return_cam:
    #         path, target, cam = self.imgs[index]
    #     else:
    #         path, target = self.imgs[index]
    #     if not self.load_img_to_cash:
    #         img = self.loader(path)
    #         img = self.transform(img)
    #     else:
    #         src = self.cash_imgs[index]
    #         img = self.transform(src)
    #
    #     if self.require_path:
    #         _, path = os.path.split(path)
    #         return img, target, path
    #
    #     if self.return_cam:
    #         return img, target, cam
    #     else:
    #         return img, target

    def __len__(self):
        return self.len



