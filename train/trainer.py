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

import sys

import models
import dataset
from utils import *
import evaluate
import loss

from core.config import config
from loss.loss import normalize

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])

model_config = config.get('model_config')
net = models.__dict__[model_config['name']]

def bulid_dataset():
    """"""
    cfg = config.get('dataset_config')
    data = dataset.__dict__[cfg['name']](part='train', size=(cfg['height'], cfg['width']))
    train_sampler = RandomIdentitySampler(data, cfg['batch_size'], cfg['least_image_per_class'])
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=cfg['batch_size'], shuffle=False, sampler=train_sampler,
        num_workers=cfg['workers'], pin_memory=True)

    test_loader = {
        'query':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](part='query',
                                            require_path=True, size=(cfg['height'], cfg['width']),
                                            ),
                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True),
        'gallery':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](part='gallery', require_path=True, size=(cfg['height'], cfg['width']),
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

    model = net(class_num=train_loader.dataset.class_num)
    model = torch.nn.DataParallel(model).cuda()
    mcfg = config.get('model_config')

    if config.get('eval'):
        ckpt = os.path.join(config.get('task_id'), 'checkpoint.pth')
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loading checkpoint '{}'".format(ckpt))
        extract(test_loader, model)
        evaluate.eval_result(config.get('dataset_config')['name'], root=config.get('task_id'))
        return

    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = loss.CrossEntropyLabelSmooth(num_classes=data.class_num).cuda()
    tri_criterion = loss.TripletLoss(margin=mcfg['margin']).cuda()
    # tri_criterion = loss.SoftTripletLoss()

    ocfg = config.get('optm_config')
    if ocfg['name'] == 'SGD':
        optimizer = torch.optim.SGD(parameters, ocfg['lr'],
                                    momentum=ocfg['momentum'],
                                    weight_decay=ocfg['weight_decay'])
    else:
        optimizer = torch.optim.Adam(parameters, ocfg['lr'],weight_decay=ocfg['weight_decay'])

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1, last_epoch=-1)
    # lr_scheduler = WarmupMultiStepLR(optimizer, [10, 40, 60], warmup_iters=10, warmup_factor=1.0, gamma=0.1, last_epoch=-1)
    lr_scheduler =  CosineAnnealingWarmUp(optimizer,
                                          T_0=5,
                                          T_end=ocfg.get('epochs'),
                                          warmup_factor=ocfg.get('warmup_factor'),
                                          last_epoch=-1)

    # lr_scheduler = WarmupMultiStepLR(optimizer, [5,30,50], gamma=0.1, last_epoch=-1)

    # optionally resume from a checkpoint
    start_epoch = ocfg.get('start_epoch')
    if ocfg['mode'] == 'train':
        ckpt = os.path.join(config.get('task_id'), 'checkpoint.pth')
        if os.path.exists(ckpt):
            print("=> loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(ckpt))



    cudnn.benchmark = True

    print(model)
    print(optimizer)

    optimizer.step()
    for epoch in range(start_epoch, ocfg.get('epochs')):
        # train for one epoch
        train(train_loader, model, criterion, tri_criterion, optimizer, lr_scheduler, epoch)
        # save checkpoint
        if not os.path.exists(config.get('task_id')):
            os.makedirs(config.get('task_id'))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, root=config.get('task_id'))

        #
    extract(test_loader, model)
    evaluate.eval_result(config.get('dataset_config')['name'], root=config.get('task_id'))


def train(train_loader, model, criterion, tri_criterion, optimizer, lr_scheduler, epoch):

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end

        lr_scheduler.step(epoch + float(i) / len(train_loader))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        outputs = model(input)
        ce_loss = torch.sum(torch.stack([criterion(logit, target) for logit in outputs[0]], dim=0))
        tri_loss = torch.sum(torch.stack([tri_criterion(feat, target) for feat in outputs[1]], dim=0))
        loss = ce_loss + tri_loss #args.weight*
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        if i % config.get('print_freq') == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'CE_Loss {loss:.4f}\t'
                  'Tri_Loss {tri_loss:.4f}\t'
                  'lr {lr: .6f}\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=ce_loss.item(),
                tri_loss=tri_loss.item(), lr=lr_scheduler.optimizer.param_groups[0]['lr']))



def extract(test_data, model):
    model.eval()
    for p, val_loader in test_data.items():
        # if os.path.exists(os.path.join(config.get('task_id'),
        #                                config.get('dataset_config')['name'] + '_' + p + '.mat')):
        #     return
        with torch.no_grad():
            paths = []
            for i, (input, target, path) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                # compute output
                outputs = model(input)
                feat = normalize(outputs[1][0])

                input_ = input.flip(3)
                outputs = model(input_)
                feat_ = normalize(outputs[1][0])

                feat = (feat + feat_) / 2
                feat = normalize(feat)

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
            save_feature(p, config.get('dataset_config')['name'], all_feature, all_label, paths)


def save_feature(part, data, features, labels, paths):
    if not os.path.exists(config.get('task_id')):
        os.makedirs(config.get('task_id'))
    sio.savemat(os.path.join(config.get('task_id'), data +'_'+ part+'.mat'),
                {'feature':features, 'label':labels, 'path':paths})

if __name__ == '__main__':
    main()
