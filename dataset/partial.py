#conding=utf-8
# @Time  : 2019/12/19 20:08
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import os.path

from utils.iotools import read_image

import glob


class PartialOrOccluded(data.Dataset):
    def __init__(self, root='/data1/home/fufuyu/dataset/',
                 name = 'Partial_iLIDS',
                 part='train', style='partial',
                 loader=read_image, size=(384,128),
                 default_transforms=None, **kwargs):
        assert style in ['partial', 'occluded']
        self.root = os.path.join(root, name)
        self.part = part
        self.loader = loader

        if part == 'train':
            imgs = []
        else:
            q_list = []
            if style == 'partial':
                prefix = 'partial_body_images'
            else:
                prefix = 'occluded_body_images'
            for q in glob.glob(os.path.join(self.root, "%s/*.jpg" % prefix)):
                q_list.append((q, 0, 0))
            g_list = []
            for g in glob.glob(os.path.join(self.root, "whole_body_images/*.jpg")):
                g_list.append((g, 0, 1))

            if part == 'query':
                imgs = q_list
            else:
                imgs = g_list

        # if len(imgs) == 0:
        #     raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        if default_transforms is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = default_transforms

        self.imgs = imgs
        self.classes = []
        self.len = len(imgs)
        self.class_num = 736

        print('\n')
        print('  **************** Summary ****************')
        print('  #  ids      : {}'.format(self.class_num))
        print('  #  images   : {}'.format(len(imgs)))
        print('  *****************************************')
        print('\n')


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

        _, path = os.path.split(path)
        return img, target, path

    def __len__(self):
        return len(self.imgs)



def test():
    market = PartialOrOccluded(part='gallery', style='occluded')

    query_loader = torch.utils.data.DataLoader(
        market,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)
    for i, (input, target, path) in enumerate(query_loader):
        # print(flags.sum())
        print(input, target, path)
        print('*********')


if __name__ == '__main__':
    test()
