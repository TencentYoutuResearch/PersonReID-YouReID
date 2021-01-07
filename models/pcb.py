#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import torch
from torch import nn
from .backbones.resnet import resnet50, resnet101, resnext101_32x8d
from .backbones.resnest import resnest50, resnest101, resnest200, resnest269
import copy

resnest_zoo = {
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269
}

class PCB(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 gcb=False,
                 with_ibn=False,
                 reduce_dim=256,
                 stripe=6,
                 stage_with_gcb_str='0,1,2,3'):
        super(PCB, self).__init__()
        stage_with_gcb = [False, False, False, False]
        if gcb and stage_with_gcb_str:
            stage_with_gcb_list = map(int, stage_with_gcb_str.split(','))
            for n in stage_with_gcb_list:
                stage_with_gcb[n] = True
        if num_layers in [50, 101, '101_32x8d']:
            if num_layers == 50:
                resnet_fn = resnet50
            elif num_layers == 101:
                resnet_fn = resnet101
            elif num_layers == '101_32x8d':
                resnet_fn = resnext101_32x8d
            self.resnet = resnet_fn(pretrained=True,
                                    gcb=gcb,
                                    with_ibn=with_ibn,
                                    stage_with_gcb=stage_with_gcb)
        elif num_layers in ['resnest50', 'resnest101', 'resnest200', 'resnest269']:
            self.resnet = resnest_zoo[num_layers](pretrained=True,
                                                  with_top=False,
                                                  last_stride=1,
                                                  )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.stripe = stripe


        embedding_layer = nn.Sequential(
                                    nn.Conv2d(2048, reduce_dim, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(reduce_dim)
                                )
        nn.init.kaiming_normal_(embedding_layer[0].weight, mode='fan_out')
        self._init_bn(embedding_layer[1])

        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim, num_classes))
        self._init_fc(fc_layer)

        self.embedding_layers = nn.ModuleList([copy.deepcopy(embedding_layer) for _ in range(stripe)])
        self.fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(stripe)])

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn.weight, 1.)
        nn.init.constant_(bn.bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x):
        x = self.resnet(x)
        # print(x.shape)
        softmax_logits, triplet_logits = [], []
        stride = 24 // self.stripe
        for i in range(self.stripe):
            s = x[:, :, i*stride:(i+1) * stride, :]
            s = self.gap(s)
            s = self.embedding_layers[i](s).squeeze(dim=3).squeeze(dim=2)
            t = self.fc_layers[i](s)
            triplet_logits.append(s)
            softmax_logits.append(t)
        triplet_logit = torch.cat(triplet_logits, dim=1)

        return softmax_logits, [triplet_logit]
