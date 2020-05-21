import copy
import torch
from torch import nn
from .backbones.resnet import resnet50, resnet101, resnext101_32x8d
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
import math

class MGN(nn.Module):
    def __init__(self, num_classes=1000, stripes=[2, 3], num_layers=50):
        super(MGN, self).__init__()
        self.stripes = stripes
        if num_layers == 50:
            resnet = resnet50(pretrained=True, last_stride=1)
        elif num_layers == 101:
            resnet = resnet101(pretrained=True, last_stride=1)
        elif num_layers == '101_32x8d':
            resnet = resnext101_32x8d(pretrained=True, last_stride=1)
        elif num_layers == '50_ibn':
            resnet = resnet50_ibn_a(pretrained=True, last_stride=1)
        elif num_layers == '101_ibn':
            resnet = resnet101_ibn_a(pretrained=True, last_stride=1)
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        self.gap = nn.AdaptiveAvgPool2d(1)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))  # , nn.ReLU())
        self._init_reduction(reduction)
        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(256, num_classes))
        self._init_fc(fc_layer)

        branches = []
        for stripe_id, stripe in enumerate(stripes):
            embedding_layers = nn.ModuleList([copy.deepcopy(reduction) for _ in range(stripe+1)])
            fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(stripe+1)])
            branches.append(
                nn.ModuleList([
                    nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(resnet.layer4)),
                    embedding_layers, fc_layers])
            )
        self.branches = nn.ModuleList(branches)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        # nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        nn.init.normal_(reduction[0].weight, std=math.sqrt(2. / 256))
        # bn
        nn.init.constant_(reduction[1].weight, 1.)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x):
        '''
        ('input.shape:', (64, 3, 384, 128))
        '''
        x = self.backone(x)
        logits, tri_logits = [], []
        for idx, stripe in enumerate(self.stripes):
            branch = self.branches[idx]
            backbone, reduces, fcs = branch
            net = backbone(x)
            # global
            global_feat = self.gap(net)
            global_feat_reduce = reduces[0](global_feat).squeeze(dim=3).squeeze(dim=2)
            global_feat_logit = fcs[0](global_feat_reduce)
            logits.append(global_feat_logit)
            tri_logits.append(global_feat_reduce)
            # local
            local_tri_logits = []
            for i in range(stripe):
                stride = 24 // stripe
                local_feat = net[:, :, i*stride: (i+1)*stride, :]
                local_feat = self.gap(local_feat)
                local_feat_reduce = reduces[i+1](local_feat).squeeze(dim=3).squeeze(dim=2)
                local_feat_logit = fcs[i+1](local_feat_reduce)
                logits.append(local_feat_logit)
                local_tri_logits.append(local_feat_reduce)
            tri_logits.append(torch.cat(local_tri_logits, dim=1))
        return logits, tri_logits


