import random
import time
import warnings
import os
import scipy.io as sio
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import models
import dataset
from utils import *

from core.config import config
from core.loss import normalize
from core.layers import convert_dsbnConstBatch
from torch.nn.modules.upsampling import Upsample

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])


class BaseTrainer(object):

    def __init__(self):
        self.logger = Logger(config.get('task_id'), rank=os.environ['RANK'] if 'RANK' in os.environ else '0')

    def init_distirbuted_mode(self):
        """"""
        print(os.environ['CUDA_VISIBLE_DEVICES'], os.environ['WORLD_SIZE'], os.environ['RANK'], os.environ['LOCAL_RANK'])
        dist.init_process_group(backend="nccl", world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    def init_seed(self):
        random.seed(config.get('seed'))
        torch.manual_seed(config.get('seed'))
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    def build_dataset(self, target_w_train=False):
        cfg = config.get('dataset_config')
        params = cfg.get('kwargs') if cfg.get('kwargs') else {}
        params['logger'] = self.logger

        source_data = dataset.__dict__[cfg['train_class']](root=cfg['root'], dataname=cfg['train_name'],
                                                           part='train',
                                                           size=(cfg['height'], cfg['width']),
                                                           least_image_per_class=cfg['least_image_per_class'],
                                                           **params
                                                           )
        if config.get('debug'):
            source_train_sampler = RandomIdentitySampler(source_data.imgs, cfg['batch_size'],
                                                         cfg['least_image_per_class'],
                                                         cfg['use_tf_sample']
                                                         )
        else:
            source_train_sampler = DistributeRandomIdentitySampler(source_data.imgs, cfg['batch_size'],
                                                                   cfg['sample_image_per_class'],
                                                                   cfg['use_tf_sample'],
                                                                   rnd_select_nid=cfg['rnd_select_nid'],
                                                                   )
        source_train_loader = torch.utils.data.DataLoader(
            source_data,
            batch_size=cfg['batch_size'], shuffle=False, sampler=source_train_sampler,
            num_workers=cfg['workers'], pin_memory=True
        )

        target_loader = {
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
                    batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True),
        }

        if target_w_train:
            target_loader['train'] = torch.utils.data.DataLoader(
                    dataset.__dict__[cfg['test_class']](root=cfg['root'], dataname=cfg['test_name'],
                                                        part='train', mode='val',
                                                        require_path=True, size=(cfg['height'], cfg['width']),
                                                        **params
                                                        ),
                    batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)

        return source_train_loader, target_loader

    def bulid_model(self, class_num):

        mconfig = config.get('model_config')
        model_name = mconfig['name']
        del mconfig['name']
        if 'split_bn' in mconfig:
            split_bn = mconfig['split_bn']
            del mconfig['split_bn']
        else:
            split_bn = None
        net = models.__dict__[model_name]
        model = net(num_classes=class_num, **mconfig)
        if split_bn:
            # convert_dsbn(model)
            convert_dsbnConstBatch(model, batch_size=config.get('dataset_config')['batch_size'], constant_batch=32)
            # convert_dsbnShare(model, constant_batch=32)

        if config.get('debug'):
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[int(os.environ['LOCAL_RANK'])],
                                                        find_unused_parameters=True
                                                        )
        cudnn.benchmark = True
        self.logger.write(model)

        return model

    def evalution(self, model, test_loader):

        ckpt = os.path.join(config.get('task_id'), 'best_model.pth')
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.logger.write(model)
        self.logger.write("=> loading checkpoint '{}'".format(ckpt))
        if config.get('convert_to_onnx'):
            self.convert_to_onnx(model, test_loader)
            # torch.onnx.export(model, )
        else:
            self.extract_and_eval(test_loader, model)


    def extract_and_eval(self, test_loader, model):
        self.extract(test_loader, model)
        mAP, rank_1 = self.eval_result()
        return mAP, rank_1

    def build_opt_and_lr(self, model):

        parameters = model.parameters()
        if config.get('debug'):
            lr_mul = 1
        else:
            lr_mul = len(config.get('gpus'))
        ocfg = config.get('optm_config')
        if ocfg['name'] == 'SGD':
            optimizer = torch.optim.SGD(parameters, float(ocfg['lr']) * lr_mul,
                                        momentum=ocfg['momentum'],
                                        weight_decay=ocfg['weight_decay'])
        else:
            optimizer = torch.optim.Adam(parameters, float(ocfg['lr']) * lr_mul,
                                         weight_decay=ocfg['weight_decay'])

        if 'multistep' in ocfg and ocfg['multistep']:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                ocfg['step'],
                                                                gamma=ocfg['gamma'],
                                                                last_epoch=-1)
        else:
            lr_scheduler = CosineAnnealingWarmUp(optimizer,
                                             T_0=5,
                                             T_end=ocfg.get('epochs'),
                                             warmup_factor=ocfg.get('warmup_factor'),
                                             last_epoch=-1)
        self.logger.write(optimizer)
        return optimizer, lr_scheduler

    def load_ckpt(self, model, optimizer=None, ckpt_path=None, add_module_prefix=False):
        """"""
        ocfg = config.get('optm_config')
        start_epoch = ocfg.get('start_epoch')
        ckpt = os.path.join(config.get('task_id'), 'checkpoint.pth')
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = ckpt_path
        if os.path.exists(ckpt):
            self.logger.write("=> loading checkpoint '{}'".format(ckpt))
            if 'LOCAL_RANK' in os.environ:
                checkpoint = torch.load(ckpt, map_location="cuda:" + str(os.environ['LOCAL_RANK']))
            else:
                checkpoint = torch.load(ckpt)
            if not ckpt_path:
                start_epoch = checkpoint['epoch']
                if optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            if add_module_prefix:
                params_names = checkpoint['state_dict'].keys()
                new_map = {}
                for k in params_names:
                    new_map['module.' + k] = checkpoint['state_dict'][k]
            else:
                new_map = checkpoint['state_dict']
            model.load_state_dict(new_map, strict=True)
            self.logger.write("=> loaded checkpoint '{}' (epoch {})"
                              .format(ckpt, checkpoint['epoch']))
            del checkpoint
        else:
            self.logger.write("=> no checkpoint found at '{}'".format(ckpt))

        return start_epoch

    def train_body(self, model, optimizer, lr_scheduler, train_loader, test_loader, start_epoch=0):
        ocfg = config.get('optm_config')
        optimizer.step()
        start = time.time()
        mAP = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(start_epoch, ocfg.get('epochs')):
            # train for one epoch
            if not config.get('debug'):
                train_loader.sampler.set_epoch(epoch)
            self.train_loop(scaler, train_loader, model, optimizer, lr_scheduler, epoch)
            # save checkpoint
            if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
                if not os.path.exists(config.get('task_id')):
                    os.makedirs(config.get('task_id'))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, root=config.get('task_id'), logger=self.logger)

                if self.eval_status(epoch):
                    cur_mAP, _ = self.extract_and_eval(test_loader, model)
                    if cur_mAP > mAP:
                        mAP = cur_mAP
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, root=config.get('task_id'), flag='best_model.pth', logger=self.logger)

        end = time.time()
        cost = end - start
        cost_h = cost // 3600
        cost_m = (cost - cost_h * 3600) // 60
        cost_s = cost - cost_h * 3600 - cost_m * 60
        self.logger.write('cost time: %d H %d M %d s' % (cost_h, cost_m, cost_s))

    def train_loop(self, scaler, train_loader, model, optimizer, lr_scheduler, epoch):

        model.train()
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time = time.time() - end

            lr_scheduler.step(epoch + float(i) / len(train_loader) / len(config.get('gpus')))
            input = input.cuda(non_blocking=True)
            if isinstance(target, (list, tuple)):
                target = [t.cuda(non_blocking=True) for t in target]
            else:
                target = target.cuda(non_blocking=True)
            # compute output
            # ce_losses, tri_losses = model(input, target=target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input, target)
                losses, losses_names = model.module.compute_loss(output, target)
                loss = torch.sum(torch.stack(losses, dim=0))  # args.weight*
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % config.get('print_freq') == 0:
                show_loss = ' '
                for name, l in zip(losses_names, losses):
                    show_loss += '%s: %f ' % (name, l.item())
                self.logger.write('Epoch: [{0}][{1}/{2}] '
                                  'Time {batch_time:.3f} '
                                  'Data {data_time:.3f} '
                                  'lr {lr: .6f} '
                                  '{loss}'.format(
                    epoch, i, len(train_loader) // len(config.get('gpus')), batch_time=batch_time,
                    data_time=data_time, loss=show_loss,
                    lr=lr_scheduler.optimizer.param_groups[0]['lr']))

        if i % config.get('print_freq') != 0:
            show_loss = ' '
            for name, l in zip(losses_names, losses):
                show_loss += '%s: %f ' % (name, l.item())
            self.logger.write('Epoch: [{0}][{1}/{2}] '
                              'Time {batch_time:.3f} '
                              'Data {data_time:.3f} '
                              'lr {lr: .6f} '
                              '{loss}'.format(
                epoch, i, len(train_loader) // len(config.get('gpus')), batch_time=batch_time,
                data_time=data_time, loss=show_loss,
                lr=lr_scheduler.optimizer.param_groups[0]['lr']))

    def extract(self, test_data, model):
        model.eval()
        res = {}
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
                    if isinstance(outputs, (list, tuple)):
                        if isinstance(outputs[1], (list, tuple)):
                            feat = normalize(torch.cat([normalize(x) for x in outputs[1]], dim=1))
                        else:
                            feat = normalize(outputs[1])
                    else:
                        feat = normalize(outputs)
                    #
                    if config.get('with_flip'):
                        input_ = input.flip(3)
                        outputs = model(input_)
                        if isinstance(outputs, (list, tuple)):
                            feat_ = normalize(torch.cat(outputs[1], dim=1), axis=1)
                        else:
                            feat_ = normalize(outputs, axis=1)
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
                        all_feature = np.vstack((all_feature, nd_feature))
                        all_label = np.concatenate((all_label, nd_label))

                    paths.extend(path)
                all_label.shape = (all_label.size, 1)

                print(all_feature.shape, all_label.shape)
                if 'test_name' in config.get('dataset_config'):
                    test_name = config.get('dataset_config')['test_name']
                else:
                    test_name = config.get('dataset_config')['target_name']
                self.save_feature(p, test_name, all_feature, all_label, paths)
                res[p] = (all_feature, all_label)
        return res

    def convert_to_onnx(self, model, test_loader):
        model.eval()
        for _, val_loader in test_loader.items():
            with torch.no_grad():
                for _, (input, _, _) in enumerate(val_loader):
                    input = input.cuda(non_blocking=True)
                    torch.onnx.export(model, input, os.path.join(config.get('task_id'), 'reid.onnx'),
                                      verbose=True, export_params=True, do_constant_folding=True,
                                      input_names=['input'], output_names=['output']
                                      )
                    break
            break

    def eval_status(self, epoch):
        ocfg = config.get('optm_config')
        # return ocfg.get('epochs') - 10 <= epoch <= ocfg.get('epochs')
        return epoch == (ocfg.get('epochs') + 1)

    def save_feature(self, part, data, features, labels, paths):
        if not os.path.exists(config.get('task_id')):
            os.makedirs(config.get('task_id'))
        self.logger.write('save at %s' % os.path.join(config.get('task_id'), data + '_' + part + '.mat'))
        sio.savemat(os.path.join(config.get('task_id'), data + '_' + part + '.mat'),
                    {'feature': features, 'label': labels, 'path': paths})

    def eval_result(self, **kwargs):
        return evaluate.eval_result(config.get('dataset_config')['test_name'],
                         root=config.get('task_id'),
                         use_pcb_format=True,
                         logger=self.logger
                         )

    def train_or_val(self):
        self.logger.write(config._config)
        self.init_seed()
        if not config.get('debug'):
            self.init_distirbuted_mode()
        source_train_loader, target_loader = self.build_dataset()
        model = self.bulid_model(source_train_loader.dataset.class_num)
        if config.get('eval'):
            self.evalution(model, target_loader)
            return
        optimizer, lr_scheduler = self.build_opt_and_lr(model)
        start_epoch = self.load_ckpt(model, optimizer)
        self.train_body(model, optimizer, lr_scheduler, source_train_loader, target_loader, start_epoch)


if __name__ == '__main__':
    trainer = BaseTrainer()
    trainer.train_or_val()
