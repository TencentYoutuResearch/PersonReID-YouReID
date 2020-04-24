import copy

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck, resnet101, resnet152 #, resnext101_32x8d
from .backbones import senet
def make_model(args):
    return MGN(args)

class MGN(nn.Module):
    def __init__(self, num_classes=1000, stripes=[2, 3]):
        super(MGN, self).__init__()
        self.stripes = stripes

        resnet = resnet50(pretrained=True)
        # resnet = resnext101_32x8d(pretrained=True)
        resnet.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        resnet.layer4[0].downsample = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                                                    nn.BatchNorm2d(2048))

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
        pool2d = nn.AvgPool2d
        self.gap = nn.AdaptiveAvgPool2d(1)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))  # , nn.ReLU())
        self._init_reduction(reduction)
        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(256, num_classes))
        self._init_fc(fc_layer)

        self.branches = nn.Sequential()
        for stripe_id, stripe in enumerate(stripes):
            branch = nn.Sequential()
            branch.add_module('branch_backbone', nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(resnet.layer4)))
            branch.add_module('branch_pool', pool2d(kernel_size=(24 // stripe, 8)))
            branch_reduces, branch_stripe_fc = nn.Sequential(), nn.Sequential()
            for i in range(stripe + 1): # global + local
                branch_reduces.add_module(str(i), copy.deepcopy(reduction))
                branch_stripe_fc.add_module(str(i), copy.deepcopy(fc_layer))
            branch.add_module('branch_reduce', branch_reduces)
            branch.add_module('branch_fc',branch_stripe_fc)
            self.branches.add_module(str(stripe_id), branch)
        # self.branches = nn.Sequential(*self.branches)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)
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
            branch_backbone, pool, reduce, fc = branch
            net = branch_backbone(x)
            # global
            global_feat = self.gap(net)
            global_feat_reduce = reduce[0](global_feat).squeeze(dim=3).squeeze(dim=2)
            global_feat_logit = fc[0](global_feat_reduce)
            logits.append(global_feat_logit)
            tri_logits.append(global_feat_reduce)
            # local
            local_tri_logits = []
            for i in range(stripe):
                stride = 24 // stripe
                local_feat = net[:, :, i*stride: (i+1)*stride, :]
                local_feat = pool(local_feat)
                local_feat_reduce = reduce[i+1](local_feat).squeeze(dim=3).squeeze(dim=2)
                local_feat_logit = fc[i+1](local_feat_reduce)
                logits.append(local_feat_logit)
                local_tri_logits.append(local_feat_reduce)
            tri_logits.append(torch.cat(local_tri_logits, dim=1))
        return logits, tri_logits


