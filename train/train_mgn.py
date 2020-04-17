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
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    data = dataset.__dict__[args.data](part='train', size=(args.height, args.width))
    train_sampler = RandomIdentitySampler(data, args.batch_size, 2)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)


    test_loader = {
        'query':
        torch.utils.data.DataLoader(dataset.__dict__[args.data](part='query', require_path=True, size=(args.height, args.width),
                                                                ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
        'gallery':
        torch.utils.data.DataLoader(dataset.__dict__[args.data](part='gallery', require_path=True, size=(args.height, args.width),
                                                                ),
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
    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = loss.CrossEntropyLabelSmooth(num_classes=data.class_num).cuda()
    tri_criterion = loss.TripletLoss(margin=args.margin).cuda()
    # tri_criterion = loss.SoftTripletLoss()

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr,weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1, last_epoch=-1)
    # lr_scheduler = WarmupMultiStepLR(optimizer, [10, 40, 60], warmup_iters=10, warmup_factor=1.0, gamma=0.1, last_epoch=-1)
    lr_scheduler =  CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=0, last_epoch=-1)

    # lr_scheduler = WarmupMultiStepLR(optimizer, [5,30,50], gamma=0.1, last_epoch=-1)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        checkpoint_path = os.path.join('../../snapshot', args.code, 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        extract(test_loader, model)
        evaluate.eval_kesci(code=args.code, data=args.data, rerank=args.rerank, signature=args.signature)
        return

    cudnn.benchmark = True

    print(model)
    print(optimizer)

    optimizer.step()
    for epoch in range(args.start_epoch, args.epochs):


        lr_scheduler.step(epoch)

        # train for one epoch
        train(train_loader, model, criterion, tri_criterion, optimizer, epoch)

        # evaluate on validation set

        # remember best prec@1 and save checkpoint
        if (epoch in [0, args.epochs - 1]) or (epoch % 5 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, code=args.code, is_best=True)
        #
    extract(test_loader, model)
    evaluate.eval_kesci(code=args.code, data=args.data, rerank=args.rerank, signature=args.signature)


def train(train_loader, model, criterion, tri_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    tri_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        outputs = model(input)

        ce_loss = torch.sum(torch.stack([criterion(logit, target) for logit in outputs[7:]], dim=0))
        tri_loss = torch.sum(torch.stack([tri_criterion(feat.renorm(2, 0, 1e-5).mul(1e5), target) for feat in outputs[1:7]], dim=0))
        # tri_loss = torch.sum(torch.stack([tri_criterion(feat, target) for feat in outputs[1:7]], dim=0))
        loss = ce_loss + args.weight*tri_loss




        # measure accuracy and record loss

        prec1, prec5 = accuracy(outputs[7], target, topk=(1, 5))
        ce_losses.update(ce_loss.item(), input.size(0))
        tri_losses.update(tri_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'CE_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Tri_Loss {tri_loss.val:.4f} ({tri_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=ce_losses, tri_loss=tri_losses, top1=top1, top5=top5))



def extract(test_data, model):



    # switch to evaluate mode
    model.eval()

    for p, val_loader in test_data.items():

        with torch.no_grad():
            paths = []
            for i, (input, target, path) in enumerate(val_loader):

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                outputs = model(input)
                feat = outputs[0].renorm(2, 0, 1e-5).mul(1e5)

                input_ = input.flip(3)
                outputs = model(input_)
                feat_ = outputs[0].renorm(2, 0, 1e-5).mul(1e5)
                if type(feat_) is list:
                    feat_ = torch.cat(feat_, dim=1)

                feat = (feat + feat_) / 2
                feat = feat.renorm(2, 0, 1e-5).mul(1e5)

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
            save_feature(p, args.code, args.data, all_feature, all_label, paths)


def save_feature(part, code, data, features, labels, paths, root='../../feature'):
    path = os.path.join(root, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(path, data +'_'+ part+'.mat'),
                {'feature':features, 'label':labels, 'path':paths})

if __name__ == '__main__':
    if args.debug:
        remote_debug()
    main()
