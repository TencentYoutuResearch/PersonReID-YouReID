#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import torch
from torch import nn
from .backbones import model_zoo
import copy
from core.loss import *


class PGFA(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 reduce_dim=256,
                 loss_type=['softmax, triplet'],
                 margin=0.5,
                 coef=0.2,
                 stripe=6,
                 use_pose=True,
                 use_gmp=True
                 ):
        super(PGFA, self).__init__()
        self.resnet = model_zoo[num_layers](
            pretrained=True, last_stride=1
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.stripe = stripe
        self.coef = coef
        self.use_pose = use_pose
        self.use_gmp = use_gmp
        if self.use_gmp:
            fea_dim = 2048 * 2
        else:
            fea_dim = 2048
        local_embedding_layer = nn.Sequential(
                                    nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(reduce_dim)
                                )
        nn.init.kaiming_normal_(local_embedding_layer[0].weight, mode='fan_out')
        self._init_bn(local_embedding_layer[1])

        fc_layer = nn.Sequential(nn.Linear(reduce_dim, num_classes)) #nn.Dropout(),
        self._init_fc(fc_layer)

        self.local_embedding_layers = nn.ModuleList([copy.deepcopy(local_embedding_layer) for _ in range(stripe)])
        self.local_fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(stripe)])

        if self.use_pose:
            global_embedding_dim = fea_dim * 2
        else:
            global_embedding_dim = fea_dim
        self.global_embedding_layer = nn.Sequential(
                                    nn.Linear(global_embedding_dim, reduce_dim * 2, bias=False),
                                    nn.BatchNorm1d(reduce_dim * 2)
                                )
        nn.init.kaiming_normal_(self.global_embedding_layer[0].weight, mode='fan_out')
        self._init_bn(self.global_embedding_layer[1])
        self.global_fc_layer = nn.Linear(reduce_dim * 2, num_classes) #copy.deepcopy(fc_layer)

        self.loss_type = loss_type

        if 'labelsmooth' in self.loss_type:
            self.ce_loss = CrossEntropyLabelSmooth(num_classes)
        else:
            self.ce_loss = nn.CrossEntropyLoss()  # .cuda()

        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin, normalize_feature=not 'circle' in self.loss_type)  # .cuda(

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn.weight, 1.)
        nn.init.constant_(bn.bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[0].weight, std=0.001)
        nn.init.constant_(fc[0].bias, 0.)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def compute_loss(self, output, mask, target):
        x = output
        local_softmax_logits, local_triplet_logits = [], []
        stride = 24 // self.stripe
        for i in range(self.stripe):
            s = x[:, :, i * stride:(i + 1) * stride, :]
            s0 = self.gap(s)
            if self.use_gmp:
                s1 = self.gmp(s)
                s = torch.cat([s0, s1], dim=1)
            else:
                s = s0
            s = self.local_embedding_layers[i](s).squeeze(dim=3).squeeze(dim=2)
            t = self.local_fc_layers[i](s)
            local_triplet_logits.append(s)
            local_softmax_logits.append(t)
        # local_triplet_logits = torch.cat(local_triplet_logits, dim=1)
        # local_triplet_loss = 0 #self.tri_loss(local_triplet_logits, target)
        local_cls_losses = []
        for ce_logit in local_softmax_logits:
            local_cls_loss = self.ce_loss(ce_logit, target) * self.coef
            local_cls_losses.append(local_cls_loss)

        global_feature = self.gap(x).squeeze()
        if self.use_gmp:
            global_feature_1 = self.gap(x).squeeze()
            global_feature = torch.cat([global_feature, global_feature_1], dim=1)
        if self.use_pose:
            global_x = torch.unsqueeze(x, 1)  # x.shape b,c,h,w -> b, 1, c, h, w
            mask = torch.unsqueeze(mask, 2)  # mask.shape b,18,h,2 -> b, 18, 1, h, w
            global_mask_feature = global_x * mask  # global_feature.shape b,18,c,h,w
            global_mask_feature = torch.mean(global_mask_feature, (3, 4))  # global_feature.shape b,18,c
            global_mask_feature = torch.max(global_mask_feature, dim=1, keepdim=False)[0]
            global_feature = torch.cat([global_feature, global_mask_feature], dim=1)

        global_embedding = self.global_embedding_layer(global_feature)
        global_logit = self.global_fc_layer(global_embedding)

        global_triplet_loss = self.tri_loss(global_embedding, target)
        cls_losses = local_cls_losses + [self.ce_loss(global_logit, target) * (1 - self.coef)]

        tri_losses = [global_triplet_loss] #local_triplet_loss,

        return cls_losses, tri_losses

    def extract_test_feature(self, output, mask):
        x = output
        local_logits = []
        stride = 24 // self.stripe
        for i in range(self.stripe):
            s = x[:, :, i * stride:(i + 1) * stride, :]
            s0 = self.gap(s)
            if self.use_gmp:
                s1 = self.gmp(s)
                s = torch.cat([s0, s1], dim=1)
            else:
                s = s0
            s = self.local_embedding_layers[i](s).squeeze(dim=3).squeeze(dim=2)
            s = normalize(s)
            s = torch.unsqueeze(s, 1)
            local_logits.append(s)
        local_logit = torch.cat(local_logits, dim=1)
        global_feature = self.gap(x).squeeze()
        if self.use_gmp:
            global_feature_1 = self.gap(x).squeeze()
            global_feature = torch.cat([global_feature, global_feature_1], dim=1)
        if self.use_pose:
            global_x = torch.unsqueeze(x, 1)  # x.shape b,c,h,w -> b, 1, c, h, w
            mask = torch.unsqueeze(mask, 2)  # mask.shape b,18,h,w -> b, 18, 1, h, w
            global_mask_feature = global_x * mask  # global_feature.shape b,18,c,h,w
            global_mask_feature = torch.mean(global_mask_feature, (3, 4))  # global_feature.shape b,18,c
            global_mask_feature = torch.max(global_mask_feature, dim=1, keepdim=False)[0]
            global_feature = torch.cat([global_feature, global_mask_feature], dim=1)
        global_logit = self.global_embedding_layer(global_feature)
        global_logit = normalize(global_logit)

        return local_logit, global_logit