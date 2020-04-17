import utils.my_transforms as my_transforms
import torch
import re
import torchvision.transforms as transforms
import random
import torch.utils.data as data
from PIL import Image
import os
import os.path
import time
import scipy.io as sio
# import sys
# sys.path.append("../..")
from utils import *
import numpy
from copy import deepcopy

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

class KESCI(data.Dataset):


    def __init__(self, root='/dockerdata/fufuyu/kesci/', part='train', round='round2',
                 loader=read_image, require_path=False, size=(384,128),
                 load_img_to_cash=False, least_image_per_class=2, use_random_pad=True,
                 camera_augmentation=False, default_transforms=None):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.least_image_per_class = least_image_per_class
        self.use_random_pad = use_random_pad
        self.subset = {'train': 'train_list.txt',
                       'trainval': 'trainval_list.txt',
                       'gallery': None,
                       'query': 'query_a_list.txt',
                       }
        root = os.path.join(root, round)

        if part in ['train', 'trainval']:
            config = os.path.join(root, self.subset[part])
            classes, class_to_idx = find_classes(config)
            imgs = make_dataset(root, config, class_to_idx, classes)
            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
        elif part == 'val':
            query_root = os.path.join(root, 'train')
            imgs = make_gallery(query_root)
            classes = []
        elif part == 'query':
            # config = os.path.join(root, self.subset[part])
            # imgs = make_query(root, config)
            # classes = []
            query_root = os.path.join(root, 'query_b')
            imgs = make_gallery(query_root)
            classes = []
        else:
            gallery_root = os.path.join(root, 'gallery_b')
            imgs = make_gallery(gallery_root)
            classes = []

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        if default_transforms is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            if part in ['train', 'train_all', 'train_val']:
                if use_random_pad:
                    print('use_random_pad')
                    self.transform = transforms.Compose([
                                                     transforms.RandomHorizontalFlip(),
                                                     #my_transforms.RandomCropping(),
                                                     my_transforms.RandomPadding(),
                                                     # my_transforms.PositionCrop((0,128,128,256)),
                                                     transforms.Resize(size),
                                                     #transforms.Pad(10),
                                                     #transforms.RandomCrop(size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225]),
                                                     #my_transforms.RandomErasing(),
                                                     ])
                else: 
                    print('use_random_crop_erase')
                    self.transform = transforms.Compose([
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.Resize(size),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop(size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225]),
                                                     my_transforms.RandomErasing(),
                                                     ])
        else:
            self.transform = default_transforms



        self.imgs = imgs
        self.classes = classes
        self.len = len(imgs)
        self.class_num = len(classes)




        print('\n')
        print('  **************** Summary ****************')
        print('  #  ids      : {}'.format(self.class_num))
        print('  #  images   : {}'.format(len(imgs)))
        print('  *****************************************')
        print('\n')

    def _postprocess(self, imgs, least_image_per_class=2):
        print(least_image_per_class)
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


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, _ = self.imgs[index]


        img = self.loader(path)
        img = self.transform(img)


        if self.require_path:
            _, path = os.path.split(path)
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.imgs)



def test():
    import torch

    import sys
    sys.path.append('..')


    market = KESCI(part='train')


    train_loader = torch.utils.data.DataLoader(
        market,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)
    for i, (input, target) in enumerate(train_loader):
        # print(flags.sum())
        pass


if __name__ == '__main__':
    test()
