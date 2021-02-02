import os
import json
import numpy as np
import pickle as pkl
import torchvision.transforms as transforms
# import torch.utils.data as data

from dataset.formatdata import FormatData
from utils.iotools import read_image
import utils.my_transforms as my_transforms


class OccludedDuke(FormatData):


    def __init__(self, root='/data1/home/fufuyu/dataset/', dataname='Occluded_Duke', part='train',
                 loader=read_image, require_path=False, size=(384, 128),
                 least_image_per_class=4,  **kwargs):

        self.root = os.path.join(root, dataname)
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.least_image_per_class = least_image_per_class

        self.part_num = kwargs.get('part_num')
        self.threshold = kwargs.get('threshold')
        self.all_one = kwargs.get('all_one', False)
        self.mode = kwargs.get('mode', 'train')
        self.return_cam = kwargs.get('return_cam', False)

        imgs, classes = self.get_imgs()

        if part in ['train', 'train_all'] and self.mode == 'train':
            brightness = kwargs.get('brightness')
            contrast = kwargs.get('contrast')
            saturation = kwargs.get('saturation')
            hue = kwargs.get('hue')

            transform_train_list = []
            if brightness is not None or contrast is not None or saturation is not None or hue is not None:
                transform_train_list.append(transforms.ColorJitter(brightness=brightness,
                                                               contrast=contrast,
                                                               saturation=saturation,
                                                               hue=hue))
            transform_train_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size),
                transforms.Pad(10),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                my_transforms.RandomErasing(
                    mean=[0.485, 0.456, 0.406]),  # sl=sl, sh=sh, r1=r1,
            ])
            self.transform = transforms.Compose(transform_train_list)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])



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

    def get_imgs(self):
        """"""
        if self.part == 'train':
            with open(os.path.join(self.root, 'train.list')) as rf:
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
                imgs.append((os.path.join(self.root, 'bounding_box_train', im_name), new_label, 0))

            classes, imgs = self._postprocess(imgs, self.least_image_per_class)
        else:
            classes = []
            if self.part == 'query':
                with open(os.path.join(self.root, 'query.list')) as rf:
                    q_list = rf.read().splitlines()
                    imgs = [(os.path.join(self.root, 'query', q), 0, 0) for q in q_list]
            else:
                with open(os.path.join(self.root, 'gallery.list')) as rf:
                    g_list = rf.read().splitlines()
                    imgs = [(os.path.join(self.root, 'bounding_box_test', g), 0, 0) for g in g_list]

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"))

        return imgs, classes

    def part_label_generate(self, imgname, part_num, imgh, threshold):
        """"""
        key_point_root = os.path.join(self.root, 'masks')
        jsonname = os.path.basename(imgname).replace('jpg', 'json')
        if not os.path.isfile(os.path.join(key_point_root, jsonname)):  ##If there is no pose json file, part_label=1
            if self.all_one:
                final_label = np.ones(part_num, dtype=np.int)
            else:
                final_label = np.zeros(part_num, dtype=np.int)
        else:
            with open(os.path.join(key_point_root, jsonname), 'r') as f:
                a = json.load(f)
                person = a['people']
            p_count = 0
            if len(person) == 0:
                if self.all_one:
                    final_label = np.ones(part_num, dtype=np.int)
                else:
                    final_label = np.zeros(part_num, dtype=np.int)
                print('no detected person')
                return final_label
            ####If there are more than one person, use the person with the largest number of landmarks
            for i in range(len(person)):
                p_points = person[i]
                p_points = p_points['pose_keypoints_2d']
                p_points = np.array(p_points)
                p_points = p_points.reshape(18, 3)
                p_points = p_points[p_points[:, 2] > threshold]
                count = p_points.shape[0]
                if count >= p_count:
                    final_point = p_points
                    p_count = count
            ####
            if final_point.shape[0] < 3:
                if self.all_one:
                    final_label = np.ones(part_num, dtype=np.int)
                else:
                    final_label = np.zeros(part_num, dtype=np.int)
            else:
                label = np.zeros(part_num)
                for j in range(len(final_point)):
                    w, h = final_point[j][:2]
                    # print("w:",w,"h:",h)
                    for k in range(part_num):
                        if h > (float(k) / part_num) * imgh and h < (float(k + 1.) / part_num) * imgh:
                            label[k] = 1
                final_label = label
        return final_label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, cam = self.imgs[index]
        img = self.loader(path)
        height = img.height
        img = self.transform(img)

        if self.require_path:
            _, path = os.path.split(path)
            return img, target, path

        if self.mode == 'train':
            part_label = self.part_label_generate(path, self.part_num, height, self.threshold)
            label = (target, part_label.astype(np.int64))
        else:
            label = target

        if self.return_cam:
            return img, label, cam
        else:
            return img, label


class OccludedMarket(OccludedDuke):


    def __init__(self, root='/data1/home/fufuyu/dataset/', dataname='market1501', part='train',
                 loader=read_image, require_path=False, size=(384, 128),
                 least_image_per_class=4,  **kwargs):
        """"""
        super(OccludedMarket, self).__init__(root, dataname, part, loader, require_path, size, least_image_per_class, **kwargs)


    def get_imgs(self):
        with open(os.path.join(self.root, 'partitions.pkl'), 'rb') as f:
            partitions = pkl.load(f)

        if self.part == 'train':
            im_names = partitions['trainval_im_names']
            ids2labels = partitions['trainval_ids2labels']

            trainval_ids2labels = {}
            current_label = 0
            for id in ids2labels:
                trainval_ids2labels[id] = current_label
                current_label += 1

            imgs = []
            for im_name in im_names:
                id, cam = self.parse_im_name(im_name)
                label = trainval_ids2labels[id]
                imgs.append((os.path.join(self.root, 'images', im_name), label, cam))

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
                elif test_mark == 1:
                    g_list.append((os.path.join(self.root, 'images', im_name), 0, 0))
            if self.part == 'query':
                imgs = q_list
            else:
                imgs = g_list

        return imgs, classes


class OccludedReID(OccludedDuke):


    def __init__(self, root='/data1/home/fufuyu/dataset/', dataname='Occluded_REID', part='train',
                 loader=read_image, require_path=False, size=(384, 128),
                 least_image_per_class=4,  **kwargs):
        """"""
        super(OccludedReID, self).__init__(root, dataname, part, loader, require_path, size, least_image_per_class, **kwargs)

    def get_imgs_from_dir(self, for_query=False):
        """"""
        prefix = 'occluded_body_images' if for_query else 'whole_body_images'
        imgs = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(self.root, prefix)):
            if not filenames:
                continue
            for name in filenames:
                if name.startswith('.'):
                    continue
                imgpath = os.path.join(dirpath, name)
                class_id = int(name.split('.')[0].split('_')[0])
                imgs.append((imgpath, class_id, int(for_query)))
        return imgs, []

    def get_imgs(self):
        if self.part == 'query':
            self.root = os.path.join(self.root)
            imgs, classes = self.get_imgs_from_dir(for_query=True)
        else:
            imgs, classes = self.get_imgs_from_dir(for_query=False)

        return imgs, classes