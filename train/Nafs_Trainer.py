import os
import re
import random
import time

import torch
import torch.nn.parallel

import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from dataset.pedes import CuhkPedes
from models.nafs import NAFS, compute_topk
from utils import *
from core.config import config
from train.BaseTrainer import BaseTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])

class NafsTrainer(BaseTrainer):

    def __init__(self):
        super(NafsTrainer, self).__init__()

    def build_dataset(self):
        dconfig = config.get('dataset_config')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        data_train = CuhkPedes(dconfig['dataset_dir'], 'train', transform=train_transform)
        data_test = CuhkPedes(dconfig['dataset_dir'], 'test', transform=test_transform)
         
        train_loader = data.DataLoader(data_train, dconfig['batch_size'], shuffle=True,
                                       num_workers=dconfig['workers'], drop_last=True)
        test_loader = data.DataLoader(data_test, dconfig['batch_size'], shuffle=False,
                                      num_workers=dconfig['workers'], drop_last=False)
        unique_image = data_test.unique        

        return train_loader, test_loader, unique_image

    def bulid_model(self):

        mconfig = config.get('model_config')

        model = NAFS()
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

        if mconfig['image_model_path'] is not None:
            print('==> Loading from pretrained models')
            model_dict = model.state_dict()
            # process keyword of pretrained model
            cnn_pretrained = torch.load(mconfig['image_model_path'])
            model_keys = model_dict.keys()
            prefix = 'module.image_model.'
            update_pretrained_dict = {}
            for k,v in cnn_pretrained.items():
                if prefix+k in model_keys:
                    update_pretrained_dict[prefix+k] = v
                if prefix+'branch2_'+k in model_keys:
                    update_pretrained_dict[prefix+'branch2_'+k] = v
                if prefix+'branch3_'+k in model_keys:
                    update_pretrained_dict[prefix+'branch3_'+k] = v
                if prefix+k not in model_keys and prefix+'branch2_'+k not in model_keys and prefix+'branch3_'+k not in model_keys:
                    print("warning: " + k + ' not load')
            model_dict.update(update_pretrained_dict)
            model.load_state_dict(model_dict)

        return model

   

    def evalution(self, model, test_loader, unique_image):
        mconfig = config.get('model_config')
        if config.get('eval'):
            ckpt_path = os.path.join(config.get('task_id'), 'best_model.pth.tar')
            start_epoch = self.load_ckpt(model, ckpt_path=ckpt_path)
        
        model.eval()
        max_size = config.get('dataset_config')['batch_size'] * len(test_loader)
        global_img_feat_bank = torch.zeros((max_size, mconfig['feature_size'])).cuda()
        global_text_feat_bank = torch.zeros((max_size, mconfig['feature_size'])).cuda()

        local_img_query_bank = torch.zeros((max_size, mconfig['part2'] +mconfig['part3'] + 1, mconfig['feature_size'])).cuda()
        local_img_value_bank = torch.zeros((max_size, mconfig['part2'] + mconfig['part3'] + 1, mconfig['feature_size'])).cuda()

        local_text_key_bank = torch.zeros((max_size, 98 + 2 + 1, mconfig['feature_size'])).cuda()
        local_text_value_bank = torch.zeros((max_size, 98 + 2 + 1, mconfig['feature_size'])).cuda()

        labels_bank = torch.zeros(max_size).cuda()
        length_bank = torch.zeros(max_size, dtype=torch.long).cuda()
        index = 0

        batch_time = [0.0, 0]

        with torch.no_grad():
            end = time.time()
            for images, captions, labels in test_loader:
                sep_captions = []
                n_sep = 2

                for i, c in enumerate(captions):
                    c = re.split(r'[;,!?.]', c)
                    if len(c) > n_sep or len(c) == n_sep:
                        sep_captions = sep_captions + c[0:n_sep]
                    else:
                        pad_length = n_sep - len(c)
                        padding = ["[PAD]" for j in range(pad_length)]
                        sep_captions = sep_captions + c + padding

                tokens, segments, input_masks, caption_length = model.module.language_model.pre_process(captions)
                sep_tokens, sep_segments, sep_input_masks, sep_caption_length = model.module.language_model.pre_process(sep_captions)

                tokens = tokens.cuda()
                segments = segments.cuda()
                input_masks = input_masks.cuda()
                caption_length = caption_length.cuda()

                sep_tokens = sep_tokens.cuda()
                sep_segments = sep_segments.cuda()
                sep_input_masks = sep_input_masks.cuda()
            
                images = images.cuda()
                labels = labels.cuda()
                interval = images.shape[0]

                p2 = [i for i in range(mconfig['part2'])]
                p3 = [i for i in range(mconfig['part3'])]

                global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = model(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3,  stage='train')

                global_img_feat_bank[index: index + interval] = global_img_feat
                global_text_feat_bank[index: index + interval] = global_text_feat
                local_img_query_bank[index: index + interval, :, :] = local_img_query
                local_img_value_bank[index: index + interval, :, :] = local_img_value
                local_text_key_bank[index: index + interval, :, :] = local_text_key
                local_text_value_bank[index: index + interval, :, :] = local_text_value
                labels_bank[index:index + interval] = labels
                length_bank[index:index + interval] = caption_length

                batch_time[0] += time.time() - end
                batch_time[1] += 1

                end = time.time()
                index = index + interval

            global_img_feat_bank = global_img_feat_bank[:index]
            global_text_feat_bank = global_text_feat_bank[:index]
            local_img_query_bank = local_img_query_bank[:index]
            local_img_value_bank = local_img_value_bank[:index]
            local_text_key_bank = local_text_key_bank[:index]
            local_text_value_bank = local_text_value_bank[:index]
            labels_bank = labels_bank[:index]
            length_bank = length_bank[:index]
            unique_image = torch.tensor(unique_image) == 1

            global_result, local_result, result = compute_topk(global_img_feat_bank[unique_image], local_img_query_bank[unique_image], local_img_value_bank[unique_image], global_text_feat_bank, local_text_key_bank,
                                                        local_text_value_bank, length_bank, labels_bank[unique_image], labels_bank, [1, 5, 10], True)

            ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = result

        
            return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, batch_time[0]/batch_time[1]
        

    
    def build_opt_and_lr(self, model):

        ocfg = config.get('optm_config')

        cnn_params = list(map(id, model.module.image_model.parameters()))
        lang_params = list(map(id, model.module.language_model.parameters()))
        cnn_lan_params = cnn_params + lang_params
        other_params = filter(lambda p: id(p) not in cnn_lan_params, model.parameters())
        other_params = list(other_params)
        
       
        param_groups = [{'params':other_params},
            {'params':model.module.image_model.parameters(), 'weight_decay':ocfg['weight_decay'], 'lr':ocfg['lr']/10},
            {'params':model.module.language_model.parameters(), 'lr':ocfg['lr']/10}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = ocfg['lr'], betas=(ocfg['adam_alpha'], ocfg['adam_beta']), eps=ocfg['epsilon'])
        
        epoches_list = ocfg['epoch_decay'].split('_')
        epoches_list = [int(e) for e in epoches_list]
        scheduler = WarmupMultiStepLR(optimizer, epoches_list, 0.1, 0.01, 10, ocfg['warmup_method'])
        
        self.logger.write(optimizer)
        return optimizer, scheduler

    def load_ckpt(self, model, optimizer=None, ckpt_path=None):

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        self.logger.write('==> Loading checkpoint "{}"'.format(ckpt_path))
        
        return start_epoch

    def save_checkpoint(self, state, checkpoint_dir):
        filename = os.path.join(checkpoint_dir, 'best_model') + '.pth.tar'
        torch.save(state, filename)

    def train_body(self, model, optimizer, lr_scheduler, train_loader, test_loader, unique_image, start_epoch=0):
        ocfg = config.get('optm_config')
        start = time.time()
        ac_t2i_top1_best = 0.0
        best_epoch = 0
    
        for epoch in range(start_epoch, ocfg['num_epoches']):
            
            self.train_loop(train_loader, model, optimizer, lr_scheduler, epoch)
            if epoch >= 0:
                ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = self.evalution(model, test_loader, unique_image)
        
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': start_epoch + epoch}
           
                if ac_top1_t2i > ac_t2i_top1_best:
                    best_epoch = epoch
                    ac_t2i_top1_best = ac_top1_t2i
                    self.save_checkpoint(state, config.get('task_id'))
            
                self.logger.write('epoch:{}'.format(epoch))
                self.logger.write('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
                ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))
            
        self.logger.write('Best epoch:{}'.format(best_epoch))
        self.logger.write('Train done')
        # self.logger.write(config.get('model_config')['checkpoint_dir'])
        self.logger.write(config.get('task_id'))

        end = time.time()
        cost = end - start
        cost_h = cost // 3600
        cost_m = (cost - cost_h * 3600) // 60
        cost_s = cost - cost_h * 3600 - cost_m * 60
        self.logger.write('cost time: %d H %d M %d s' % (cost_h, cost_m, cost_s))
    
    def train_loop(self, train_loader, model, optimizer, lr_scheduler, epoch):

        batch_time = [0.0, 0]
        train_loss = [0.0, 0]
        image_pre = [0.0, 0]
        text_pre = [0.0, 0]


        model.train()
        end = time.time()
        for step, (images, captions, labels) in enumerate(train_loader):
            sep_captions = []
            n_sep = 2
            for i, c in enumerate(captions):
                c = re.split(r'[;,!?.]', c)
                if len(c) > n_sep or len(c) == n_sep:
                    sep_captions = sep_captions + c[0:n_sep]
                else:
                    pad_length = n_sep - len(c)
                    padding = ["[PAD]" for j in range(pad_length)]
                    sep_captions = sep_captions + c + padding

            tokens, segments, input_masks, caption_length = model.module.language_model.pre_process(captions)
            sep_tokens, sep_segments, sep_input_masks, sep_caption_length = model.module.language_model.pre_process(sep_captions)
            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            caption_length = caption_length.cuda()

            sep_tokens = sep_tokens.cuda()
            sep_segments = sep_segments.cuda()
            sep_input_masks = sep_input_masks.cuda()

            images = images.cuda()
            labels = labels.cuda()

            p2 = [i for i in range(config.get('model_config')['part2'])]
            p3 = [i for i in range(config.get('model_config')['part3'])]
            random.shuffle(p2)
            random.shuffle(p3)

            global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = model(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3,  stage='train')

            cmpm_loss, cmpc_loss, cont_loss, loss, image_precision, text_precision, pos_avg_sim, neg_arg_sim, local_pos_avg_sim, local_neg_avg_sim = model.module.compute_loss(
                global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value, caption_length, labels)

            if step % 10 == 0:
                self.logger.write('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}, cont_loss:{:.3f}, pos_sim_avg:{:.3f}, neg_sim_avg:{:.3f}, lpos_sim_avg:{:.3f}, lneg_sim_avg:{:.3f}'.
                    format(epoch, step, cmpm_loss, cmpc_loss, cont_loss, pos_avg_sim, neg_arg_sim, local_pos_avg_sim, local_neg_avg_sim))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time[0] += time.time() - end
            batch_time[1] += 1
            end = time.time()

            train_loss[0] += loss.item()*images.shape[0]
            train_loss[1] += images.shape[0]

            image_pre[0] += image_precision.item()*images.shape[0]
            image_pre[1] += images.shape[0]

            text_pre[0] += text_precision.item()*images.shape[0]
            text_pre[1] += images.shape[0]
        self.logger.write('Train done for epoch-{}'.format(epoch))
        self.logger.write('Epoch:  [{}|{}], train_time: {:.3f}, train_loss: {:.3f}'.format(epoch, config.get('optm_config')['num_epoches'], batch_time[0]/batch_time[1], train_loss[0]/train_loss[1]))
        self.logger.write('image_precision: {:.3f}, text_precision: {:.3f}'.format(image_pre[0]/image_pre[1], text_pre[0]/text_pre[1]))
        
        lr_scheduler.step()



    def train_or_val(self):
        self.logger.write(config._config)
        self.init_seed()
        
        train_loader, test_loader, unique_image = self.build_dataset()
        model = self.bulid_model()
        if config.get('eval'):
            self.evalution(model, test_loader, unique_image)
            return
        optimizer, lr_scheduler = self.build_opt_and_lr(model)
        self.train_body(model, optimizer, lr_scheduler, train_loader, test_loader, unique_image)

    

if __name__ == '__main__':
    trainer = NafsTrainer()
    trainer.train_or_val()
