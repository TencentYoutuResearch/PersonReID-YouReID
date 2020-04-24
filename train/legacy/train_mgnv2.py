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
if not args.evaluate:
    sys.stdout = Logger(os.path.join('../../snapshot', args.code))
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
net = models.__dict__[args.net]


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

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
    data = dataset.__dict__[args.data](part=args.part, size=(args.height, args.width),  use_random_pad=args.use_random_pad,
                                       load_img_to_cash=args.load_img_to_cash, least_image_per_class=args.least_image_per_class)
    train_sampler = RandomIdentitySampler(data, args.batch_size, args.least_image_per_class, use_tf_sample=args.use_tf_sample)
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
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
        'val':
        torch.utils.data.DataLoader(dataset.__dict__[args.data](part='val', require_path=True, size=(args.height, args.width),
                                                                ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
    }

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

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr,weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1, last_epoch=-1)
    # lr_scheduler = WarmupMultiStepLR(optimizer, [10, 40, 60], warmup_iters=10, warmup_factor=1.0, gamma=0.1, last_epoch=-1)
    # lr_scheduler =  CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=0, last_epoch=-1)
    lr_scheduler = CosineAnnealingWarmUp(optimizer, T_0=5, T_end=args.epochs, warmup_factor=0, last_epoch=-1)

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
        #model.load_state_dict(checkpoint['state_dict'])
        #extract(test_loader, model)
        if args.data == 'KESCI':
            #evaluate.eval_kesci(code=args.code, data=args.data, rerank=True)
            evaluate.part_re_ranking_group(code=args.code, data=args.data)
        else:
            evaluate.eval_result(code=args.code, data=args.data) 
        return

    cudnn.benchmark = True

    print(model)
    print(optimizer)

    optimizer.step()
    for epoch in range(args.start_epoch, args.epochs):


        # lr_scheduler.step(epoch)

        # train for one epoch
        train(lr_scheduler, train_loader, model, criterion, tri_criterion, optimizer, epoch)

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
    print('eval', args.data)
    if args.data == 'KESCI':
        evaluate.eval_kesci(code=args.code, data=args.data)
    else:
        evaluate.eval_result(code=args.code, data=args.data)


def train(lr_scheduler, train_loader, model, criterion, tri_criterion, optimizer, epoch):

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        lr_scheduler.step(epoch + i / len(train_loader))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        global_softmax, local_softmaxs, global_triplet, local_triplet = model(input)
        losses, losses_name = [], []
        flags = [0]
        for idx, g_s in enumerate(global_softmax):
            losses.append(criterion(g_s, target))
            losses_name.append('global_cls_b%d' % idx)
        flags.append(len(losses_name))
        for idx, l_list in enumerate(local_softmaxs):
            for idy, l_s in enumerate(l_list):
                losses.append(criterion(l_s, target))
                losses_name.append('local_cls_b%d_%d' % (idx, idy))
        flags.append(len(losses_name))
        for idx, g_t in enumerate(global_triplet):
            losses.append(tri_criterion(normalize(g_t), target))
            losses_name.append('global_triplet_b%d' % idx)
        flags.append(len(losses_name))
        for idx, l_t in enumerate(local_triplet):
            losses.append(tri_criterion(normalize(l_t), target))
            losses_name.append('local_triplet_b%d' % (idx))
        flags.append(len(losses_name))
        loss = torch.sum(torch.stack(losses, dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        if i % args.print_freq == 0:
            out_str = 'Epoch[{0}][{1}/{2}] lr={lr: .9f} '.format(epoch, i, len(train_loader),
                                                                 lr=optimizer.param_groups[0]['lr'])
            for idf in range(len(flags) - 1):
                for cost, name in zip(losses[flags[idf]:flags[idf + 1]], losses_name[flags[idf]:flags[idf + 1]]):
                    out_str += ' %s: %.3f ' % (name, cost.item())
            print(out_str)



def extract(test_data, model):



    # switch to evaluate mode
    model.eval()

    for p, val_loader in test_data.items():
        if os.path.exists(os.path.join('../../feature',  str(args.code), args.data +'_'+ p+'.mat')):
            continue
        print('extract', str(args.code), args.data +'_'+ p+'.mat')
        with torch.no_grad():
            paths = []
            for i, (input, target, path) in enumerate(val_loader):
                print(i, len(val_loader))
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                outputs = model(input)
                feat = []
                feat.extend([normalize(o) for o in outputs[2]]) # global triplet
                feat.extend([normalize(o) for o in outputs[3]])# local triplet
                feats = normalize(torch.cat(feat, dim=1))
                # print(feats.size(1))

                input_ = input.flip(3)
                outputs = model(input_)
                feat_ = []
                feat_.extend([normalize(o) for o in outputs[2]])
                feat_.extend([normalize(o) for o in outputs[3]])
                feats_ = normalize(torch.cat(feat_, dim=1))

                feat = (feats + feats_) / 2
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

