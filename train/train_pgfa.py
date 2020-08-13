#conding=utf-8
# @Time  : 2020/6/18 17:05
# @Author: fufuyu
# @Email:  fufuyu@tencent.com

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

if config.get('use_fp16'):
    from apex import amp

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])

import numpy as np
import json

logger = Logger(config.get('task_id'))

def bulid_dataset():
    """"""
    cfg = config.get('dataset_config')
    if cfg['name'] == 'PartialOrOccluded':
        # params = {'style': cfg['style'], 'name': cfg['dataname']}
        params = {}
    else:
        params = {'mgn_style_aug': cfg['mgn_style_aug']}
    data = dataset.__dict__[cfg['name']](part='train',
                                         size=(cfg['height'], cfg['width']),
                                         least_image_per_class=cfg['least_image_per_class'],
                                         **params
                                         )
    train_sampler = RandomIdentitySampler(data, cfg['batch_size'],
                                          cfg['least_image_per_class'],
                                          cfg['use_tf_sample']
                                          )
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=cfg['batch_size'], shuffle=False, sampler=train_sampler,
        num_workers=cfg['workers'], pin_memory=True)

    test_loader = {
        'query':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](part='query',
                                            require_path=True, size=(cfg['height'], cfg['width']),
                                              **params
                                            ),
                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True),
        'gallery':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['name']](part='gallery', require_path=True,
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
        model = net(num_classes=train_loader.dataset.class_num, **mconfig)
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = net(num_classes=train_loader.dataset.class_num, **mconfig).cuda()
    mcfg = config.get('model_config')

    if config.get('eval'):
        # config['use_fp16'] = False
        ckpt = os.path.join(config.get('task_id'), 'checkpoint.pth')
        checkpoint = torch.load(ckpt)
        keys = list(checkpoint['state_dict'].keys())
        for k in keys:
            if 'fc_layer' in k:
                del checkpoint['state_dict'][k]
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("=> loading checkpoint '{}'".format(ckpt))
        extract(test_loader, model)
        evaluate.eval_result_mask(config.get('dataset_config')['name'],
                             root=config.get('task_id'),
                             use_pcb_format=config.get('dataset_config')['name'] in ['Market1501']
                             )
        return


    parameters = model.parameters()

    ocfg = config.get('optm_config')
    if ocfg['name'] == 'SGD':
        optimizer = torch.optim.SGD(parameters, ocfg['lr'],
                                    momentum=ocfg['momentum'],
                                    weight_decay=ocfg['weight_decay'])
    else:
        optimizer = torch.optim.Adam(parameters, ocfg['lr'],weight_decay=ocfg['weight_decay'])

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1, last_epoch=-1)
    lr_scheduler = CosineAnnealingWarmUp(optimizer,
                                          T_0=5,
                                          T_end=ocfg.get('epochs'),
                                          warmup_factor=ocfg.get('warmup_factor'),
                                          last_epoch=-1)

    if config.get('use_fp16'):
        fp16_cfg = config.get('fp16_config')
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=fp16_cfg['level'],
                                          keep_batchnorm_fp32=fp16_cfg['keep_batchnorm_fp32'],
                                          loss_scale=fp16_cfg['loss_scale']
                                          )
        model = nn.DataParallel(model).cuda()


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
            logger.write("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
            del checkpoint
        else:
            logger.write("=> no checkpoint found at '{}'".format(ckpt))

    cudnn.benchmark = True

    print(model)
    print(optimizer)

    optimizer.step()
    start = time.time()
    for epoch in range(start_epoch, ocfg.get('epochs')):
        # train for one epoch
        train(train_loader, model, optimizer, lr_scheduler, epoch)
        # save checkpoint
        if not os.path.exists(config.get('task_id')):
            os.makedirs(config.get('task_id'))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, root=config.get('task_id'))

    end = time.time()
    cost = end - start
    cost_h = cost // 3600
    cost_m = (cost - cost_h * 3600) // 60
    cost_s = cost - cost_h * 3600 - cost_m * 60
    logger.write('cost time: %d H %d M %d s' % (cost_h, cost_m, cost_s))
        #
    extract(test_loader, model)
    evaluate.eval_result_mask(config.get('dataset_config')['name'],
                         root=config.get('task_id'),
                         use_pcb_format=False
                         )


def train(train_loader, model, optimizer, lr_scheduler, epoch):

    model.train()
    end = time.time()
    for i, (input, mask, target) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end

        lr_scheduler.step(epoch + float(i) / len(train_loader))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        # compute output
        # ce_losses, tri_losses = model(input, target=target)
        output = model(input)
        if len(config.get('gpus')) > 1:
            ce_losses, tri_losses = model.module.compute_loss(output, mask, target)
        else:
            ce_losses, tri_losses = model.compute_loss(output, mask, target)
        ce_loss = torch.sum(torch.stack(ce_losses, dim=0))
        tri_loss = torch.sum(torch.stack(tri_losses, dim=0))
        loss = ce_loss + tri_loss #args.weight*
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if config.get('use_fp16'):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        if i % config.get('print_freq') == 0:
            show_loss = ' '
            for ce_id, ce in enumerate(ce_losses):
                show_loss += 'CE_%d: %f ' % (ce_id, ce.item())
            for tri_id, tri in enumerate(tri_losses):
                show_loss += 'Tri_%d: %f ' % (tri_id, tri.item())
            logger.write('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time:.3f} '
                  'Data {data_time:.3f} '
                  'lr {lr: .6f} '
                  '{loss}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=show_loss,
                lr=lr_scheduler.optimizer.param_groups[0]['lr']))

def extract(test_data, model):
    model.eval()
    for p, val_loader in test_data.items():
        # if os.path.exists(os.path.join(config.get('task_id'),
        #                                config.get('dataset_config')['name'] + '_' + p + '.mat')):
        #     return
        with torch.no_grad():
            paths = []
            for i, (input, mask, target, path) in enumerate(val_loader):
                # print(input[0])
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                # compute output
                output = model(input)
                if len(config.get('gpus')) > 1:
                    local_logit, global_logit = model.module.extract_test_feature(output, mask)
                else:
                    local_logit, global_logit = model.extract_test_feature(output, mask)

                input_ = input.flip(3)
                mask_ = mask.flip(3)
                output = model(input_)
                if len(config.get('gpus')) > 1:
                    local_logit_, global_logit_ = model.module.extract_test_feature(output, mask_)
                else:
                    local_logit_, global_logit_ = model.extract_test_feature(output, mask_)

                local_logit = (local_logit + local_logit_) / 2
                global_logit = (global_logit + global_logit_) / 2

                local_logit = local_logit.cpu()
                global_logit = global_logit.cpu()
                target = target.cpu()

                nd_label = target.numpy()
                nd_local_logit = local_logit.numpy()
                nd_global_logit = global_logit.numpy()
                if i == 0:
                    all_local_logit = nd_local_logit
                    all_global_logit = nd_global_logit
                    all_label = nd_label
                else:
                    all_local_logit = numpy.vstack((all_local_logit, nd_local_logit))
                    all_global_logit = numpy.vstack((all_global_logit, nd_global_logit))
                    all_label = numpy.concatenate((all_label, nd_label))

                paths.extend(path)
            all_label.shape = (all_label.size, 1)
            print(all_global_logit.shape, all_local_logit.shape, all_label.shape)
            save_feature(p, config.get('dataset_config')['name'],
                         all_global_logit, all_local_logit, all_label, paths)


def save_feature(part, data, gf, lf, labels, paths):
    # if part == 'query':
    #     tag = 'occluded_body_images'
    # else:
    #     tag = 'whole_body_images'
    if not os.path.exists(config.get('task_id')):
        os.makedirs(config.get('task_id'))
    final_labels = []
    for p in paths:
        final_labels.append(part_label_generate(p))
    final_labels = numpy.vstack(final_labels)
    sio.savemat(os.path.join(config.get('task_id'), data +'_'+ part+'.mat'),
                {'global_feature':gf, 'local_feature': lf,
                 'label':labels, 'path':paths, 'part_label': final_labels})

def part_label_generate(p, part_num=6,imgh=384):
    # jp = os.path.join(
    #     '/data1/home/fufuyu/dataset', name, 'mask', part,
    #     str(int(os.path.basename(p).split('.')[0])).zfill(4) + '_c1_0000.json'
    # )
    jp = p.replace('npy', 'json')
    if not os.path.isfile(jp): ##If there is no pose json file, part_label=1
        final_label=np.ones(part_num)
    else:
        with open(jp,'r') as f:
            a=json.load(f)
            person=a['people']
        p_count=0
        if len(person)==0:
            final_label=np.ones(part_num)
            print('no detected person')
            return final_label
        ####If there are more than one person, use the person with the largest number of landmarks
        for i in range(len(person)):
            p_points=person[i]
            p_points=p_points['pose_keypoints_2d']
            p_points=np.array(p_points)
            p_points=p_points.reshape(18,3)
            p_points=p_points[p_points[:,2]>0.2]
            count=p_points.shape[0]
            if count>=p_count:
                final_point=p_points
                p_count=count
        ####
        if final_point.shape[0]<3:
            final_label=np.ones(part_num)
        else:
            label=np.zeros(part_num)
            for j in range(len(final_point)):
                w,h = final_point[j][:2]
                for k in range(part_num):
                    if h> (float(k)/part_num)*imgh and h<(float(k+1.)/part_num)*imgh:
                        label[k]=1
            final_label=label
    return final_label

if __name__ == '__main__':
    main()