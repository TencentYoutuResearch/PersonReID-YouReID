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

class Occluded_Duke(data.Dataset):
    def __init__(self, root='/data1/home/fufuyu/dataset/Occluded_Duke', part='train',
                 loader=read_image, require_path=False, size=(384,128),
                 least_image_per_class=4, mgn_style_aug=False,
                 load_img_to_cash=False, default_transforms=None, **kwargs):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.least_image_per_class = least_image_per_class
        self.load_img_to_cash = load_img_to_cash

        if part == 'train':
            with open(os.path.join(root, 'train.list')) as rf:
                im_names = rf.read().splitlines()
            ids = set([])
            for line_i, im_name in enumerate(im_names):
                d = int(im_name.split('_')[0])
                if d not in ids:
                    ids.add(d)
            ids2labels = {d: idx for (idx, d) in enumerate(sorted(list(ids)))}
            imgs = []
            for line_i, im_name in enumerate(im_names):
                d = int(im_name.split('_')[0])
                new_label = ids2labels[d]
                imgs.append((os.path.join(root, 'bounding_box_train', im_name), new_label, 0))

            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
        else:
            classes = []
            if part == 'query':
                with open(os.path.join(root, 'query.list')) as rf:
                    q_list = rf.read().splitlines()
                    imgs = [os.path.join(root, 'bounding_box_test', q) for q in q_list]
            else:
                with open(os.path.join(root, 'gallery.list')) as rf:
                    g_list = rf.read().splitlines()
                    imgs = [os.path.join(root, 'bounding_box_test', g) for g in g_list]


        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        if default_transforms is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            if part in ['train', 'train_all']:
                if mgn_style_aug:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        my_transforms.RandomPadding(),  #optional
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    # re1 = my_transforms.RandomErasing(probability=0.5, sl=0.01, sh=0.04)
                    # re2 = my_transforms.RandomErasing(probability=0.5, sl=0.16, sh=0.4)
                    re = my_transforms.RandomErasing()
                    tlist = [
                                      transforms.RandomHorizontalFlip(),
                                      transforms.Resize(size),
                                      transforms.Pad(10),
                                      transforms.RandomCrop(size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                                         ]
                    t = tlist + [re]
                    self.transform = transforms.Compose(t)
                    # self.transform2 = transforms.Compose(t2)
        else:
            self.transform = default_transforms



        self.imgs = imgs
        self.classes = classes
        self.len = len(imgs)
        self.class_num = len(classes)

        if self.load_img_to_cash:
            self.cash_imgs = []
            for index in range(self.len):
                path, target, _ = self.imgs[index]
                img = self.loader(path)
                self.cash_imgs.append(img)



        print('\n')
        print('  **************** Summary ****************')
        print('  #  ids      : {}'.format(self.class_num))
        print('  #  images   : {}'.format(len(imgs)))
        print('  *****************************************')
        print('\n')

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
            # img2 = self.transform2(img)
        else:
            src = self.cash_imgs[index]
            img = self.transform(src)
            # img2 = self.transform2(src)

        # img = np.stack([img1, img2], axis=0)

        if self.require_path:
            _, path = os.path.split(path)
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.imgs)



def test():
    market = Occluded_Duke(part='train')

    train_loader = torch.utils.data.DataLoader(
        market,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)
    for i, (input, target) in enumerate(train_loader):
        # print(flags.sum())
        print(input.shape, target.shape)
        print('*********')


if __name__ == '__main__':
    test()
