import random
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import models
import dataset
from utils import *
import evaluate

from core.config import config
from core.loss import normalize

if config.get('use_fp16'):
    from apex import amp

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])
logger = Logger(config.get('task_id'))

def bulid_dataset():
    """"""
    cfg = config.get('dataset_config')
    params = {'logger': logger}
    data = dataset.__dict__[cfg['name']](dataname=cfg['train_name'], part='train',
                                         size=(cfg['height'], cfg['width']),
                                         require_path = True,
                                         least_image_per_class=cfg['least_image_per_class'],
                                         load_img_to_cash= cfg['load_img_to_cash'],
                                         **params
                                         )
    train_sampler = RandomIdentitySampler(data, cfg['batch_size'],
                                          cfg['least_image_per_class'],
                                          cfg['use_tf_sample']
                                          )
    train_loader = {'train': torch.utils.data.DataLoader(
        data,
        batch_size=cfg['batch_size'], shuffle=False, sampler=train_sampler,
        num_workers=cfg['workers'], pin_memory=True)}

    test_loader = {
        'query':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](dataname=cfg['test_name'], part='query',
                                            require_path=True, size=(cfg['height'], cfg['width']),
                                              **params
                                            ),
                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True),
        'gallery':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](dataname=cfg['test_name'],
                                              part='gallery', require_path=True,
                                              size=(cfg['height'], cfg['width']),
                                              **params
                                            ),
                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True), }

    return train_loader, test_loader

def main():
    if config.get('seed') is not None:
        random.seed(config.get('seed'))
        torch.manual_seed(config.get('seed'))
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    train_loader, test_loader = bulid_dataset()
    # create models
    mconfig = config.get('model_config')
    model_name = mconfig['name']
    del mconfig['name']
    net = models.__dict__[model_name]
    if not config.get('use_fp16') and len(config.get('gpus')) != 1:
        model = net(num_classes=train_loader['train'].dataset.class_num, **mconfig).resnet
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = net(num_classes=train_loader['train'].dataset.class_num, **mconfig).resnet.cuda()
    mcfg = config.get('model_config')
    ckpt = os.path.join(config.get('task_id'), 'checkpoint.pth')
    checkpoint = torch.load(ckpt)
    keys = list(checkpoint['state_dict'].keys())
    for k in keys:
        if 'fc_layer' in k:
            del checkpoint['state_dict'][k]
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(model)
    print("=> loading checkpoint '{}'".format(ckpt))
    extract(train_loader, model)
    evaluate.eval_result(config.get('dataset_config')['test_name'],
                         root=config.get('task_id'),
                         use_pcb_format=True,
                         logger=logger
                         )

def extract(test_data, model):
    model.eval()
    for p, val_loader in test_data.items():
        with torch.no_grad():
            paths = []
            for i, (input, target, path) in enumerate(val_loader):
                # print(input[0])
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                # compute output
                outputs = model(input)
                feat = normalize(torch.cat(outputs[1], dim=1))

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


if __name__ == '__main__':
    main()