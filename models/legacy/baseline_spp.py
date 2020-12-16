#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

import torch
from torch import nn
from .backbones.resnet import resnet50, resnet101, resnext101_32x8d
from .backbones.resnest import resnest50, resnest101, resnest200, resnest269
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a

resnest_zoo = {
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269
}

class Baseline(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 gcb=False,
                 with_ibn=False,
                 reduce_dim=768,
                 stage_with_gcb_str='0,1,2,3'):
        super(Baseline, self).__init__()
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
        elif num_layers == '101_ibn':
            self.resnet = resnet101_ibn_a(pretrained=True, last_stride=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap_2 = nn.AvgPool2d((12, 4))
        self.gmp_2 = nn.MaxPool2d((12, 4))
        self.gap_4 = nn.AvgPool2d((6, 2))
        self.gmp_4 = nn.MaxPool2d((6, 2))

        self.embedding_layer = nn.Conv2d(4096, reduce_dim, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode='fan_out')
        self.bn = nn.Sequential(nn.BatchNorm2d(reduce_dim))
        self._init_bn(self.bn)

        self.fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim, num_classes))
        self._init_fc(self.fc_layer)

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn[0].weight, 1.)
        nn.init.constant_(bn[0].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x):
        x = self.resnet(x)
        b = x.size(0)
        # print(x.shape)
        x1 = self.gap(x)
        x2 = self.gmp(x)
        x1_2 = self.gap_2(x).view((b, -1))
        x2_2 = self.gmp_2(x).view((b, -1))
        x1_4 = self.gap_4(x).view((b, -1))
        x2_4 = self.gmp_4(x).view((b, -1))
        x = torch.cat([x1, x2, x1_2, x2_2, x1_4, x2_4], 1)
        x = self.embedding_layer(x)
        x = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        y = self.fc_layer(x)

        return [y], [x]

    # def compute_loss(self, x):
    #     cls_logit, tri_logit = self.forward(x)
