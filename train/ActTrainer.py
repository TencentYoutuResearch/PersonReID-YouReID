import random
import time
import warnings
import os
import models
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import tqdm
from utils import *

from core.config import config
from core.loss import TripletLoss
from train.BaseTrainer import BaseTrainer
from dataset.formatdata import FormatData
import dataset
from sklearn.cluster import DBSCAN
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])


def label_unknown(known_feat, all_lab, unknown_feat):
    disMat = pairwise_dist(known_feat, unknown_feat)
    labLoc = disMat.argmin(dim=0)
    return all_lab[labLoc]


def label_noise(feature, labels):
    # features and labels with -1
    noiseFeat, pureFeat = feature[labels == -1, :], feature[labels != -1, :]
    labels = labels[labels != -1]
    unLab = label_unknown(pureFeat, labels, noiseFeat)
    return unLab.numpy()


def pairwise_dist(q_feature, g_feature):  # 246s
    x, y = F.normalize(q_feature), F.normalize(g_feature)
    # x, y = q_feature, g_feature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    print('-----* Distance Matrix has been computed*-----')
    return disMat.clamp_(min=1e-5)


def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1):
    all_num = input_feature.shape[0]
    # feat = input_feature.astype(np.float16)
    feat = torch.from_numpy(input_feature)  # target
    del input_feature

    if lambda_value != 0:
        print('Computing source distance...')
        srcFeat, tarFeat = torch.from_numpy(input_feature_source), feat
        # all_num_source  = input_feature_source.shape[0]
        # sour_tar_dist = np.power(cdist(input_feature, input_feature_source), 2).astype(np.float32) #608s
        sour_tar_dist = pairwise_dist(srcFeat, tarFeat).t().numpy()
        sour_tar_dist = 1 - np.exp(-sour_tar_dist)  # tar-src
        source_dist_vec = np.min(sour_tar_dist, axis=1)
        source_dist_vec = source_dist_vec / (np.max(source_dist_vec) + 1e-3)  # for trget
        source_dist = np.zeros([all_num, all_num])  # tar size
        for i in range(all_num):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del sour_tar_dist
        del source_dist_vec

    print('Computing original distance...')
    original_dist = pairwise_dist(feat, feat).cpu().numpy()
    print('done...')
    # original_dist = np.power(original_dist,2).astype(np.float16)
    del feat
    # original_dist = np.concatenate(dist,axis=0)
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / (np.max(original_dist, axis=0)))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.

    print('Starting re_ranking...')
    for i in tqdm.tqdm(range(all_num)):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,
                                :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,
                                 :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in tqdm.tqdm(range(all_num)):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in tqdm.tqdm(range(gallery_num)):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in tqdm.tqdm(range(all_num)):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    if lambda_value == 0:
        return jaccard_dist
    else:
        final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        return final_dist

def calScores(clusters, labels):
    """
    compute pair-wise precision pair-wise recall
    """
    from scipy.special import comb
    if len(clusters) == 0:
        return 0, 0
    else:
        curCluster = []
        for curClus in clusters.values():
            curCluster.append(labels[curClus])
        TPandFP = sum([comb(len(val), 2) for val in curCluster])
        TP = 0
        for clusterVal in curCluster:
            for setMember in set(clusterVal):
                if sum(clusterVal == setMember) < 2: continue
                TP += comb(sum(clusterVal == setMember), 2)
        FP = TPandFP - TP
        # FN and TN
        TPandFN = sum([comb(labels.tolist().count(val), 2) for val in set(labels)])
        FN = TPandFN - TP
        # cal precision and recall
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        fScore = 2 * precision * recall / (precision + recall)
        return precision, recall, fScore

class ActTrainer(BaseTrainer):
    def __init__(self):
        super(ActTrainer, self).__init__()
        self.cfg = config.get('dataset_config')

    def cluster(self, source_train_loader, target_loader, model, iter_n, cluster=None):
        src_feature = self.extract_feature_for_cluster(source_train_loader, model)
        tgt_feature = self.extract_feature_for_cluster(target_loader['train'], model)
        rerank_dist = re_ranking(src_feature, tgt_feature, lambda_value=0.1)
        # get cluster
        if iter_n == 0:
            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(1.6e-3 * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            self.logger('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)
        labels = cluster.fit_predict(rerank_dist)
        np.save(os.path.join(config.get('task_id'), 'cluster_%d.npy' % iter_n), labels)
        num_ids = len(set(labels)) - 1
        self.logger(f'Iteration {iter_n} have {num_ids} training ids')
        # generate new dataset
        new_dataset, unknown_dataset = [], []
        unknown_lab = label_noise(torch.from_numpy(tgt_feature), torch.from_numpy(labels))
        # generate temporary loader
        un_counter, index = 0, 0
        from collections import defaultdict
        realIDs, fake_ids = defaultdict(list), []
        for (fname, realPID, cam), label in zip(target_loader['train'].dataset.imgs, labels):
            if label == -1:
                unknown_dataset.append((fname, int(unknown_lab[un_counter]), cam))  # unknown data
                fake_ids.append(int(unknown_lab[un_counter]))
                realIDs[realPID].append(index)
                un_counter += 1
                index += 1
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, cam))
            fake_ids.append(label)
            realIDs[realPID].append(index)
            index += 1
        # get temporary loader
        self.logger('Iter {} have {} training images'.format(iter_n, len(new_dataset)))
        precision, recall, fscore = calScores(realIDs, np.asarray(fake_ids))
        self.logger('precision:{}, recall:{}, fscore: {}'.format(100 * precision, 100 * recall, fscore))
        cfg = config.get('dataset_config')

        labeled_loader = torch.utils.data.DataLoader(
            FormatData(part='train', require_path=True,
                       size=(cfg['height'], cfg['width']),
                       imgs=new_dataset),
            batch_size=self.cfg['batch_size'], shuffle=False,
            sampler=RandomIdentitySampler(new_dataset, self.cfg['batch_size'],
                                          self.cfg['least_image_per_class'],
                                          self.cfg['use_tf_sample']),
            num_workers=self.cfg['workers'], pin_memory=True, drop_last=True
        )

        noise_loader = torch.utils.data.DataLoader(
            FormatData(part='train', require_path=True,
                      size=(cfg['height'], cfg['width']),
                      imgs=unknown_dataset),
            batch_size=self.cfg['batch_size'], shuffle=False,
            sampler=RandomIdentitySampler(unknown_dataset, self.cfg['batch_size'],
                                          self.cfg['least_image_per_class'],
                                          self.cfg['use_tf_sample']),
            num_workers=self.cfg['workers'], pin_memory=True, drop_last=True
        )
        return noise_loader, labeled_loader, cluster

    def extract_feature_for_cluster(self, val_loader, model):
        model.eval()
        with torch.no_grad():
            for i, (input, _, path) in enumerate(val_loader):
                input = input.cuda(non_blocking=True)
                # compute output
                outputs = model(input)
                if isinstance(outputs, (list, tuple)):
                    feat = outputs[1]
                else:
                    feat = outputs
                nd_feature = feat.cpu().numpy()
                if i == 0:
                    all_feature = nd_feature
                else:
                    all_feature = np.vstack((all_feature, nd_feature))
        return all_feature

    def get_loss(self, outputs, targets, loss_func):
        return loss_func(outputs[0], targets) + loss_func(outputs[1], targets)

    def train_batch(self, model, co_model, labeled_loader, noise_loader, rem_rate,
                    optimizers, epoch, lr_scheduler):
        loss_func = TripletLoss(config.get('model_config')['margin'], reduce='none')
        loss_optim = TripletLoss(config.get('model_config')['margin'], reduce='mean')
        end = time.time()
        for i, (inputs, targets, names) in enumerate(labeled_loader):
            data_time = time.time() - end
            if i % 2 == 0:
                # update CNNB
                output_pool5, output_fc, _ = model(inputs)  # fc out
                loss_pure = self.get_loss([output_pool5, output_fc], targets, loss_func)  # assigned samples
                # easy samples
                loss_idx = torch.argsort(loss_pure)
                pure_input = inputs[loss_idx[:int(rem_rate * loss_pure.shape[0])], ...]
                pure_lab = targets[loss_idx[:int(rem_rate * loss_pure.shape[0])]].long()
                # loss for cnn B
                pure_pool5, pure_fc, _ = co_model(pure_input)
                loss_cnnb = self.get_loss([pure_pool5, pure_fc], pure_lab, loss_optim)
                loss_cnnb = loss_cnnb.mean()
                optimizers[1].zero_grad()
                loss_cnnb.backward()
                for param in co_model.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                optimizers[1].step()
            else:
                # update CNNA
                try:
                    noise_input, noise_targets, noise_names = next(noise_temp_loader)
                except:
                    noise_temp_loader = iter(noise_loader)
                    noise_input, noise_targets, noise_names = next(noise_temp_loader)
                noise_pool5, noise_fc, _ = co_model(noise_input)
                loss_noise = self.get_loss([noise_pool5, noise_fc], noise_targets, loss_func)
                # sample mining
                loss_idx = torch.argsort(loss_noise)
                noise_input, noise_lab = noise_input[loss_idx], noise_targets[loss_idx]
                noise_input, noise_lab = noise_input[:int(rem_rate * loss_noise.shape[0]), ...], noise_lab[:int(
                    rem_rate * loss_noise.shape[0])]
                # mix update, part assigned and part unassigned
                mix_input, mix_lab = torch.cat([inputs, noise_input]), torch.cat([targets.long(), noise_lab])
                mix_pool5, mix_fc, _ = model(mix_input)
                loss_mix = self.get_loss([mix_pool5, mix_fc], mix_lab, loss_optim)
                loss_cnna = loss_mix.mean()
                # update CNNA
                optimizers[0].zero_grad()
                loss_cnna.backward()
                for param in model.parameters():
                    try:
                        param.grad.data.clamp(-1., 1.)
                    except:
                        continue
                # update modelA
                optimizers[0].step()
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            if i % config.get('print_freq') == 0:
                show_loss = '%s: %f ' % ('TripLet Loss', loss_cnnb.item() if i % 2 == 0 else loss_cnna.item())
                self.logger.write('Epoch: [{0}][{1}/{2}] '
                                  'Time {batch_time:.3f} '
                                  'Data {data_time:.3f} '
                                  'lr {lr: .6f} '
                                  '{loss}'.format(
                    epoch, i, len(labeled_loader) // len(config.get('gpus')), batch_time=batch_time,
                    data_time=data_time, loss=show_loss, lr=lr_scheduler.optimizer.param_groups[0]['lr']))

    def train_act(self, model, co_model, optimizer_a, optimizer_b, lr_scheduler,
                  source_train_loader, target_loader, target_train_loader, start_epoch):
        ocfg = config.get('optm_config')
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()

        mAP, _ = self.extract_and_eval(target_loader, model)

        save_checkpoint({
                'epoch': 0,
                'state_dict': model.state_dict(),
        }, root=config.get('task_id'), flag='best_model.pth', logger=self.logger)

        for iter_n in range(ocfg.get('iteration')):
            if iter_n == 0:
                noise_loader, labeled_loader, cluster = self.cluster(source_train_loader, target_loader, model, iter_n)
            else:
                noise_loader, labeled_loader, cluster = self.cluster(source_train_loader, target_loader, model, iter_n, cluster)
            for epoch in range(start_epoch, ocfg.get('epochs')):
                rem_rate = 0.2 + (0.8 / ocfg.get('iteration')) * (1 + iter_n)
                # predict pseudo labels
                # train for one epoch
                if not config.get('debug'):
                    target_train_loader.sampler.set_epoch(epoch)
                self.train_batch(model, co_model, labeled_loader, noise_loader,
                                 rem_rate, [optimizer_a, optimizer_b], epoch, lr_scheduler)
                # save checkpoint
                if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
                    if not os.path.exists(config.get('task_id')):
                        os.makedirs(config.get('task_id'))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict()
                    }, root=config.get('task_id'), logger=self.logger)

                    if self.eval_status(epoch):
                        cur_mAP, _ = self.extract_and_eval(target_loader, model)
                        if cur_mAP > mAP:
                            mAP = cur_mAP
                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                            }, root=config.get('task_id'), flag='best_model.pth', logger=self.logger)

    def eval_status(self, epoch):
        return True

    def bulid_model(self, class_num=0):
        mconfig = config.get('model_config')
        dconfig = config.get('dataset_config')
        model_name = mconfig['name']
        del mconfig['name']
        net = models.__dict__[model_name]
        if dconfig['train_name'] == 'dukemtmc':
            model = net(num_classes=632, pretrained=False, **mconfig)
            co_model = net(num_classes=632, pretrained=False, **mconfig)
        elif dconfig['train_name'] == 'market1501':
            model = net(num_classes=676, pretrained=False, **mconfig)
            co_model = net(num_classes=676, pretrained=False, **mconfig)
        elif dconfig['train_name'] == 'cuhk03':
            model = net(num_classes=1230, pretrained=False, **mconfig)
            co_model = net(num_classes=1230, pretrained=False, **mconfig)
        else:
            raise RuntimeError('Please specify the number of classes (ids) of the network.')

        if config.get('debug'):
            model = torch.nn.DataParallel(model).cuda()
            co_model = torch.nn.DataParallel(co_model).cuda()
        else:
            model = model.cuda()
            co_model = co_model.cuda()
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[int(os.environ['LOCAL_RANK'])],
                                                        find_unused_parameters=True)
            co_model = nn.parallel.DistributedDataParallel(co_model,
                                                           device_ids=[int(os.environ['LOCAL_RANK'])],
                                                           find_unused_parameters=True)
        cudnn.benchmark = True
        self.logger.write(model)
        return model, co_model


    def build_dataset(self, target_w_train=False):
        cfg = config.get('dataset_config')
        params = cfg.get('kwargs') if cfg.get('kwargs') else {}
        params['logger'] = self.logger


        source_train_loader = torch.utils.data.DataLoader(
                    dataset.__dict__[cfg['train_class']](root=cfg['root'], dataname=cfg['train_name'],
                                                        part='train', mode='val',
                                                        require_path=True, size=(cfg['height'], cfg['width']),
                                                        least_image_per_class=1, **params
                                                        ),
                    batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)

        target_train_data = dataset.__dict__[cfg['test_class']](root=cfg['root'], dataname=cfg['test_name'],
                                                           part='train',
                                                           size=(cfg['height'], cfg['width']),
                                                           least_image_per_class=cfg['least_image_per_class'],
                                                           **params
                                                           )
        if config.get('debug'):
            target_train_sampler = RandomIdentitySampler(target_train_data, cfg['batch_size'],
                                                         cfg['least_image_per_class'],
                                                         cfg['use_tf_sample']
                                                         )
        else:
            target_train_sampler = DistributeRandomIdentitySampler(target_train_data, cfg['batch_size'],
                                                                   cfg['sample_image_per_class'],
                                                                   cfg['use_tf_sample'],
                                                                   rnd_select_nid=cfg['rnd_select_nid'],
                                                                   )
        target_train_loader = torch.utils.data.DataLoader(
            target_train_data,
            batch_size=cfg['batch_size'], shuffle=False, sampler=target_train_sampler,
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

        return source_train_loader, target_loader, target_train_loader

    def train_or_val(self, check_point=None):
        self.logger.write(config._config)
        self.init_seed()
        if not config.get('debug'):
            self.init_distirbuted_mode()
        source_train_loader, target_loader, target_train_loader = self.build_dataset(target_w_train=True)
        model, co_model = self.bulid_model()
        if config.get('eval'):
            self.evalution(model, target_loader)
            return
        optimizer, lr_scheduler = self.build_opt_and_lr(model)
        co_optimizer, _ = self.build_opt_and_lr(co_model)
        start_epoch = self.load_ckpt(model, ckpt_path=config.get('ckpt_path'), add_module_prefix=True)
        _ = self.load_ckpt(co_model, ckpt_path=config.get('ckpt_path'), add_module_prefix=True)
        self.train_act(model, co_model, optimizer, co_optimizer, lr_scheduler,
                       source_train_loader, target_loader, target_train_loader, start_epoch)


if __name__ == '__main__':
    trainer = ActTrainer()
    trainer.train_or_val()
