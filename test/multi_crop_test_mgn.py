import numpy
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys

sys.path.append("..")
sys.path.append("../../config/pycharm-debug-py3k.egg")
import models
import dataset
from utils import *
import evaluate
import loss

best_prec1 = 0
args = ini_parser().parse_args()
if args.evaluate is False:
    sys.stdout = Logger(os.path.join('../../snapshot', args.code))
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
net = models.__dict__[args.net]

def main():

    global args, best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data = dataset.__dict__[args.data](part='train', size=(args.height, args.width))
    # Data loading code
    test_trans =  transforms.Compose([
                transforms.RandomResizedCrop(size=(args.height, args.width), scale=(0.75,0.95)),
                transforms.Resize((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    test_loader = {
        'query':
        torch.utils.data.DataLoader(dataset.__dict__[args.data](part='query', require_path=True, size=(args.height, args.width),
                                                                default_transforms=test_trans),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
        'gallery':
        torch.utils.data.DataLoader(dataset.__dict__[args.data](part='gallery', require_path=True, size=(args.height, args.width),
                                                                default_transforms=test_trans),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),}

    # if args.evaluate:
    #     extract(test_loader, args, net)
    #     evaluate.eval_kesci(code=args.code, data=args.data)
    #     return

    # create models

    model = net(class_num=data.class_num)
    model = torch.nn.DataParallel(model).cuda()



    # define loss function (criterion) and optimizer
    # if args.pretrained:
    #     parameters = model.module.get_param(args.lr)
    # else:

    # lr_scheduler = WarmupMultiStepLR(optimizer, [5,30,50], gamma=0.1, last_epoch=-1)
    # optionally resume from a checkpoint

    cudnn.benchmark = True

    checkpoint_path = os.path.join('../../snapshot', args.code, 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    extract(test_loader, model)
    evaluate.eval_kesci_multi_crop(code=args.code, data=args.data)





def extract(test_data, model):



    # switch to evaluate mode
    model.eval()
    for t in range(5):
        for p, val_loader in test_data.items():

            with torch.no_grad():
                paths = []
                for i, (input, target, path) in enumerate(val_loader):

                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    # compute output
                    outputs = model(input)
                    feat = outputs[0]

                    input_ = input.flip(3)
                    outputs = model(input_)
                    feat_ = outputs[0]
                    if type(feat_) is list:
                        feat_ = torch.cat(feat_, dim=1)

                    feat = (feat + feat_) / 2

                    feature = feat.cpu()
                    target = target.cpu()

                    nd_label = target.numpy()
                    nd_feature = feature.numpy()
                    if i == 0:
                        all_feature = nd_feature
                        all_label = nd_label

                    else:
                        all_feature = numpy.vstack((all_feature, nd_feature))
                        all_label = numpy.concatenate((all_label, nd_label))

                    paths.extend(path)
                all_label.shape = (all_label.size, 1)

                print(all_feature.shape, all_label.shape)
                save_feature(p, args.code, args.data, all_feature, all_label, paths, t)


def save_feature(part, code, data, features, labels, paths, t, root='../../feature'):
    path = os.path.join(root, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(path, data +'_'+ part+ '_'+ str(t) +'.mat'),
                {'feature':features, 'label':labels, 'path':paths})

if __name__ == '__main__':
    if args.debug:
        remote_debug()
    main()
