import torch.utils.data as data
import numpy as np
import os
import pickle
import h5py
import json
from PIL import Image
from scipy.misc import imread, imresize

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class CuhkPedes(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, image_root, split, transform=None):

        self.image_root = image_root
        self.transform = transform
        self.split = split.lower()
        self.read_json(image_root)

        #if os.path.exists(self.image_root):
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       'Please follow the directions to generate datasets')

        print('Reading data from reid_raw.json')
    
    def read_json(self, json_path):
            self.images = []
            self.captions = []
            self.labels = []
            json_path = os.path.join(json_path, 'reid_raw.json')
            with open(json_path) as f:
                raw_data = json.load(f)
                for data in raw_data:
                    if data['split'] == self.split:
                        file_path = data['file_path']
                        cap_list = data['captions']
                        label = data['id'] - 1
                        for cap in cap_list:
                            self.images.append(file_path)
                            self.captions.append(cap)
                            self.labels.append(label)
            
            if self.split == 'test':
                unique = []
                new_test_images = []
                for test_image in self.images:
                    if test_image in new_test_images:
                        unique.append(0)
                    else:
                        unique.append(1)
                        new_test_images.append(test_image)
                self.unique = unique
            
            

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """


        img_path, caption, label = self.images[index], self.captions[index], self.labels[index]

        middle_path = "imgs"
        if middle_path not in img_path:
            img_path = os.path.join(self.image_root, middle_path, img_path)
        else:
            img_path = os.path.join(self.image_root, img_path)


        img = imread(img_path)
        img = imresize(img, (384, 128))
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, caption, label                        

    def __len__(self):
       return len(self.labels)
