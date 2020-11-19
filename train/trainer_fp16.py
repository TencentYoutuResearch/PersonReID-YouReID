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
from core.layers import convert_dsbn, convert_dsbnConstBatch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])

logger = Logger(config.get('task_id'))

def bulid_dataset():
    """"""
    cfg = config.get('dataset_config')
    params = {'logger': logger}
    # if cfg['name'] == 'PartialOrOccluded':
    #     # params = {'style': cfg['style'], 'name': cfg['dataname']}
    #     params = {}
    # else:
    #     params = {'mgn_style_aug': cfg['mgn_style_aug']}
    data = dataset.__dict__[cfg['train_class']](root=cfg['root'], dataname=cfg['train_name'], part='train',
                                         size=(cfg['height'], cfg['width']),
                                         least_image_per_class=cfg['least_image_per_class'],
                                         load_img_to_cash= cfg['load_img_to_cash'],
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
                dataset.__dict__[cfg['test_class']](root=cfg['root'], dataname=cfg['test_name'], part='query',
                                            require_path=True, size=(cfg['height'], cfg['width']),
                                              **params
                                            ),
                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True),
        'gallery':
            torch.utils.data.DataLoader(
                dataset.__dict__[cfg['test_class']](root=cfg['root'], dataname=cfg['test_name'],
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
    if 'split_bn' in mconfig:
        split_bn = mconfig['split_bn']
        del mconfig['split_bn']
    else:
        split_bn = None
    model = net(num_classes=train_loader.dataset.class_num, **mconfig)
    if split_bn:
        # convert_dsbn(model)
        convert_dsbnConstBatch(model, batch_size=config.get('dataset_config')['batch_size'], constant_batch=32)
    model = torch.nn.DataParallel(model).cuda()
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
        print(model)
        print("=> loading checkpoint '{}'".format(ckpt))
        extract(test_loader, model)
        evaluate.eval_result(config.get('dataset_config')['test_name'],
                             root=config.get('task_id'),
                             use_pcb_format=True,
                             logger=logger
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

    scaler = torch.cuda.amp.GradScaler()


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
        train(scaler, train_loader, model, optimizer, lr_scheduler, epoch)
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
    evaluate.eval_result(config.get('dataset_config')['test_name'],
                         root=config.get('task_id'),
                         use_pcb_format=True,
                         logger=logger
                         )


def train(scaler, train_loader, model, optimizer, lr_scheduler, epoch):

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end

        lr_scheduler.step(epoch + float(i) / len(train_loader))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        # ce_losses, tri_losses = model(input, target=target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input)
            ce_losses, tri_losses = model.module.compute_loss(output, target)
            ce_loss = torch.sum(torch.stack(ce_losses, dim=0))
            tri_loss = torch.sum(torch.stack(tri_losses, dim=0))
            loss = ce_loss + tri_loss  # args.weight*

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            for i, (input, target, path) in enumerate(val_loader):
                # print(input[0])
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                # compute output
                outputs = model(input)
                feat = normalize(torch.cat(outputs[1], dim=1))

                input_ = input.flip(3)
                outputs = model(input_)
                feat_ = normalize(torch.cat(outputs[1], dim=1))

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
            save_feature(p, config.get('dataset_config')['test_name'], all_feature, all_label, paths)


def save_feature(part, data, features, labels, paths):
    if not os.path.exists(config.get('task_id')):
        os.makedirs(config.get('task_id'))
    sio.savemat(os.path.join(config.get('task_id'), data +'_'+ part+'.mat'),
                {'feature':features, 'label':labels, 'path':paths})

if __name__ == '__main__':
    main()