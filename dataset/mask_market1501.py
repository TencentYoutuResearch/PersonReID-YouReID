#conding=utf-8
# @Time  : 2019/12/19 20:08
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import utils.my_transforms as my_transforms
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as data
import numpy as np
import os
import os.path
import random
import scipy.io as sio
import sys
sys.path.append("..")
from utils.iotools import read_image, is_image_file

import numpy
from copy import deepcopy
import pickle
import glob

class MaskMarket1501(data.Dataset):
    def __init__(self,
                 root='/data1/home/fufuyu/dataset/market1501', part='train',
                 test_dataset='Partial_iLIDS', #Partial_iLIDS Partial-REID Occluded_REID
                 loader=read_image, require_path=False, size=(384,128),
                 least_image_per_class=4, mgn_style_aug=False,
                 load_img_to_cash=False, default_transforms=None, **kwargs):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.least_image_per_class = least_image_per_class
        self.load_img_to_cash = load_img_to_cash
        self.test_dataset = test_dataset
        with open(os.path.join(root, 'partitions.pkl'), 'rb') as f:
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
                mask_name = im_name.split('/')[-1].split('.')[0] + '.npy'
                mask_path = os.path.join(root, 'masks', mask_name)
                imgs.append((os.path.join(root, 'images', im_name), label, 0, mask_path))

            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
        else:
            classes = []
            if self.test_dataset in ['Partial_iLIDS', 'Partial-REID']:
                ext = 'jpg'
            else:
                ext = 'tif'
            q_img_list = sorted(glob.glob('/data1/home/fufuyu/dataset/%s/images/occluded_body_images/*.%s' % (self.test_dataset,ext)))
            g_img_list = sorted(glob.glob('/data1/home/fufuyu/dataset/%s/images/whole_body_images/*.%s' % (self.test_dataset, ext)))
            q_list, g_list = [], []
            for q in q_img_list:
                q_dir = os.path.dirname(q).replace('/images', '/masks')
                q_name = os.path.basename(q)
                if self.test_dataset == 'Partial_iLIDS':
                    q_mask_name = str(int(q_name.split('.')[0])).zfill(4) + '_c1_0000.npy'
                elif self.test_dataset == 'Partial-REID':
                    idx, idm = q_name.split('_')
                    q_mask_name = '_'.join([idx, 'c0', idm.replace('jpg', 'npy')])
                else:
                    q_mask_name = q_name.replace('tif', 'npy')
                q_list.append((q, 0, 0, os.path.join(q_dir, q_mask_name)))
            for g in g_img_list:
                g_dir = os.path.dirname(g).replace('/images', '/masks')
                g_name = os.path.basename(g)
                if self.test_dataset == 'Partial_iLIDS':
                    g_mask_name = str(int(g_name.split('.')[0])).zfill(4) + '_c0_0000.npy'
                elif self.test_dataset == 'Partial-REID':
                    idx, idm = g_name.split('_')
                    g_mask_name = '_'.join([idx, 'c1', idm.replace('jpg', 'npy')])
                else:
                    g_mask_name = g_name.replace('tif', 'npy')
                g_list.append((g, 0, 0, os.path.join(g_dir, g_mask_name)))
            if part == 'query':
                imgs = q_list
            else:
                imgs = g_list


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
                                                         transforms.Resize(size),
                                                         transforms.Pad(10),
                                                         transforms.RandomCrop(size),
                                                         transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.1), #, saturation=0.2
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

        if self.load_img_to_cash:
            self.cash_imgs = []
            for index in range(self.len):
                path, target, _, mask_p = self.imgs[index]
                img, mask = self.loader(path, mask_p)
                self.cash_imgs.append((img, mask))

        print('\n')
        print('  **************** Summary ****************')
        print('  #  ids      : {}'.format(self.class_num))
        print('  #  images   : {}'.format(len(imgs)))
        print('  *****************************************')
        print('\n')

    def _postprocess(self, imgs, least_image_per_class=4):
        image_dict = {}
        for _, c ,i,_ in imgs:
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
        for path, c ,i, mask_p in imgs:
            if c in new_class_to_idx:
                new_imgs.append((path, new_class_to_idx[c], 0, mask_p))
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
        path, target, _, mask_p = self.imgs[index]
        if not self.load_img_to_cash:
            img, mask = self.loader(path, mask_p)
        else:
            img, mask = self.cash_imgs[index]
        if not self.require_path and random.random() < 0.5:
            img = F.hflip(img)
            mask = np.flip(mask, 2)
        img = self.transform(img)
        mask = torch.from_numpy(mask.copy())
        mask = mask.float()
        if self.require_path:
            _, path = os.path.split(path)
            return img, mask, target, mask_p

        return img, mask, target

    def __len__(self):
        return len(self.imgs)



def test():
    market = Market1501(part='train')

    train_loader = torch.utils.data.DataLoader(
        market,
        batch_size=4, shuffle=True,
        num_workers=4, pin_memory=True)
    for i, (input, mask, target) in enumerate(train_loader):
        # print(flags.sum())
        print(input, mask, target)
        print('*********')


if __name__ == '__main__':
    test()
