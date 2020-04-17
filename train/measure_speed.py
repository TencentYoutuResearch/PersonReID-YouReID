#conding=utf-8
# @Time  : 2019/12/20 17:20
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import sys

sys.path.append("..")
#from utils import *
#args = ini_parser().parse_args()
import torch
import time
import dataset
from utils.sampler import RandomIdentitySampler

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    for num_workers in range(0,32,4):
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        data = dataset.__dict__['Market1501'](part='train', size=(384, 128),
                                           load_img_to_cash=False, least_image_per_class=4)
        train_sampler = RandomIdentitySampler(data, 64, 4, use_tf_sample=True)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=64, shuffle=False, sampler=train_sampler,
            num_workers=num_workers, pin_memory=False)

        start = time.time()
        for epoch in range(1, 5):
            for batch_idx, (data, target) in enumerate(train_loader): #
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end-start,num_workers))
