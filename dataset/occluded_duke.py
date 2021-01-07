import os
import sys
sys.path.append("..")

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import utils.my_transforms as my_transforms
from utils.iotools import read_image
from .formatdata import FormatData

class OccludedDuke(FormatData):
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
            for im_name in im_names:
                d = int(im_name.split('_')[0])
                if d not in ids:
                    ids.add(d)
            ids2labels = {d: idx for (idx, d) in enumerate(sorted(list(ids)))}
            imgs = []
            for im_name in im_names:
                d = int(im_name.split('_')[0])
                new_label = ids2labels[d]
                imgs.append((os.path.join(root, 'bounding_box_train', im_name), new_label, 0))

            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
        else:
            classes = []
            if part == 'query':
                with open(os.path.join(root, 'query.list')) as rf:
                    q_list = rf.read().splitlines()
                    imgs = [(os.path.join(root, 'query', q), 0, 0) for q in q_list]
            else:
                with open(os.path.join(root, 'gallery.list')) as rf:
                    g_list = rf.read().splitlines()
                    imgs = [(os.path.join(root, 'bounding_box_test', g), 0, 0) for g in g_list]


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
                path, _, _ = self.imgs[index]
                img = self.loader(path)
                self.cash_imgs.append(img)



        print('\n')
        print('  **************** Summary ****************')
        print('  #  ids      : {}'.format(self.class_num))
        print('  #  images   : {}'.format(len(imgs)))
        print('  *****************************************')
        print('\n')



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
