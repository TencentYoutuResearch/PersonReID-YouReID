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
import glob

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

class FormatData(data.Dataset):
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
                id = self.parse_im_name(im_name, 'id')
                label = trainval_ids2labels[id]
                imgs.append((os.path.join(self.root, 'images', im_name), label, 0))

            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
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
                    # crop_image_paths = glob.glob('/data1/home/fufuyu/dataset/msmt17_pcb/images/*.jpg')
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



        self.imgs = imgs
        if self.load_img_to_cash:
            mean, std = self.get_mean_and_std()
            print(mean, std)
        self.classes = classes
        self.len = len(imgs)
        self.class_num = len(classes)

        if self.load_img_to_cash:
            self.cash_imgs = []
            for index in range(self.len):
                path, target, _ = self.imgs[index]
                img = self.loader(path)
                self.cash_imgs.append(img)



        self.logger('\n')
        self.logger('  **************** Summary ****************')
        self.logger('  #  ids      : {}'.format(self.class_num))
        self.logger('  #  images   : {}'.format(len(imgs)))
        self.logger('  *****************************************')
        self.logger('\n')

    def _postprocess(self, imgs, least_image_per_class=4):
        image_dict = {}
        for _, c ,i in imgs:
            if c not in image_dict:
                image_dict[c] = 1
            else:
                image_dict[c] += 1

        temp = deepcopy(image_dict)

        for k,v in temp.items():
            if v < least_image_per_class:
                image_dict.pop(k)


        new_class_to_idx = {k: i for i, k in enumerate(list(image_dict.keys()))}

        new_imgs = []
        for path, c ,i in imgs:
            if c in new_class_to_idx:
                new_imgs.append((path, new_class_to_idx[c], 0))
        classes = list(range(len(new_class_to_idx)))
        return classes, new_imgs


    def parse_im_name(self, im_name, parse_type='id'):
        """Get the person id or cam from an image name."""
        assert parse_type in ('id', 'cam')
        if parse_type == 'id':
            parsed = int(im_name[:8])
        else:
            parsed = int(im_name[9:13])
        return parsed

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, _ = self.imgs[index]
        if not self.load_img_to_cash:
            img = self.loader(path)
            img = self.transform(img)
        else:
            src = self.cash_imgs[index]
            img = self.transform(src)

        if self.require_path:
            _, path = os.path.split(path)
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.imgs)


    def get_mean_and_std(self):
        '''Compute the mean and std value of dataset.'''
        mean, std = np.array([0, 0, 0]), np.array([0, 0, 0])
        for path, target, _ in self.imgs:
            img = self.loader(path)
            img = np.asarray(img)
            for i in range(3):
                mean[i] += img[:,:, i].mean()
                std[i] += img[:,:,i].std()

        mean = mean / len(self.imgs) / 255.0
        std = std / len(self.imgs) / 255.0
        return mean, std

# def test():
#     market = Market1501(part='train')
#
#     train_loader = torch.utils.data.DataLoader(
#         market,
#         batch_size=32, shuffle=True,
#         num_workers=4, pin_memory=True)
#     for i, (input, target) in enumerate(train_loader):
#         # print(flags.sum())
#         print(input, target)
#         print('*********')


if __name__ == '__main__':
    test()
