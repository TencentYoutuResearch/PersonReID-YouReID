#conding=utf-8
# @Time  : 2019/12/19 20:08
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import torchvision.transforms as transforms
import torch.utils.data as data
import os
import os.path
import sys
sys.path.append("..")
from utils.iotools import read_image, is_image_file

import numpy as np
from .formatdata import make_gallery


class TestData(data.Dataset):
    def __init__(self, root='/data1/home/fufuyu/dataset/',
                 dataname='market1501', loader=read_image, require_path=False, size=(384,128),
                 **kwargs
                 ):

        self.root = os.path.join(root, dataname)
        self.loader = loader
        self.require_path = require_path
        self.logger = kwargs.get('logger', print)

        imgs = make_gallery(self.root)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


        self.imgs = imgs #* 128
        self.len = len(self.imgs)

        self.logger('\n')
        self.logger('  **************** Summary ****************')
        self.logger('  #  name : {}   part: {}'.format(dataname, 'test'))
        self.logger('  #  images   : {}'.format(len(self.imgs)))
        self.logger('  *****************************************')
        self.logger('\n')


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, cam = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)

        if self.require_path:
            _, path = os.path.split(path)
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.imgs)


