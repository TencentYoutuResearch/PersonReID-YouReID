#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

from .backbones.resnet import ResNet, Bottleneck
from core.loss import *
from core.layers import GeneralizedMeanPoolingP

class SplitResNet(ResNet):
    def __init__(self,
                 pretrained=True,
                 split_stage=3,
                 split_num=3,
                 last_stride=1,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 model_name = 'resnet50',
                 use_non_local=False,
                 groups=1, width_per_group=64):
        super(SplitResNet, self).__init__(
            last_stride=last_stride, block=block, layers=layers,
            model_name=model_name, use_non_local=use_non_local, groups=groups,
            width_per_group=width_per_group
        )
        self.split_stage= split_stage
        self.split_num = split_num
        if pretrained:
            self.load_pretrain()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.split_stage == 3:
            x = torch.split(x, x.size()[2] // self.split_num, dim=2)
            x = torch.cat(x, dim=0)
        x = self.layer3(x)
        if self.split_stage == 4:
            x = torch.split(x, x.size()[2] // self.split_num, dim=2)
            x = torch.cat(x, dim=0)
        x = self.layer4(x)
        if self.split_stage > 0:
            x = torch.split(x, x.size()[0] // self.split_num, dim=0)
            x = torch.cat(x, dim=2)
        return x

class BaselineSplit(nn.Module):
    def __init__(self,
                 split_stage=3,
                 split_num=3,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=768,
                 pool_type='baseline',
                 loss_type=['softmax, triplet'],
                 margin=0.5,
                 use_non_local=False
                 ):
        super(BaselineSplit, self).__init__()
        kwargs = {
            'use_non_local': use_non_local
        }
        self.resnet = SplitResNet(split_stage=split_stage,
                                  split_num=split_num
                                  )

        self.pool_type = pool_type
        if self.pool_type == 'baseline':
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.gmp = nn.AdaptiveMaxPool2d(1)
            input_dim = 4096
        elif self.pool_type == 'gemm':
            print('use use_gem_pool')
            self.gemp = GeneralizedMeanPoolingP()
            input_dim = 2048
        elif self.pool_type == 'norm':
            input_dim = 2048

        self.embedding_layer = nn.Conv2d(input_dim, reduce_dim,
                                         kernel_size=1, stride=1, bias=False
                                         )
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode='fan_out')
        self.bn = nn.Sequential(nn.BatchNorm2d(reduce_dim))
        self._init_bn(self.bn)

        self.loss_type = loss_type
        if 'softmax' in self.loss_type:
            self.fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim, num_classes))
            self._init_fc(self.fc_layer)
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()  # .cuda()
        elif 'arcface' in self.loss_type:
            pass
        elif 'circle' in self.loss_type:
            self.fc_layer = Circle(num_classes, reduce_dim)

        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin, normalize_feature=not 'circle' in self.loss_type) #.cuda()

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
        # print(x.shape)
        if self.pool_type == 'baseline':
            x1 = self.gap(x)
            x2 = self.gmp(x)
            x = torch.cat([x1, x2], 1)
        elif self.pool_type == 'gemm':
            x = self.gemp(x)
        elif self.pool_type == 'norm':
            # norm = torch.norm(x, 2, 1, keepdim=True)
            b, c, h, w = x.size()
            score = torch.softmax(x.view((b, c, h*w)), dim=-1)
            score = score.view((b, c, h, w))
            x = torch.sum(x * score, dim=[2, 3], keepdim=True)

        x = self.embedding_layer(x)
        x = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        if 'softmax' in self.loss_type:
            y = self.fc_layer(x)
            return [y], [x]
        else:
            return [x], [x]

    def compute_loss(self, output, target):
        ce_logit, tri_logit = output
        if 'arcface' in self.loss_type or 'circle' in self.loss_type:
            ce_logit[0] = self.fc_layer(tri_logit[0], target)
        cls_loss = self.ce_loss(ce_logit[0], target)
        if 'triplet' in self.loss_type:
            tri_loss = self.tri_loss(tri_logit[0], target)
            return [cls_loss], [tri_loss]
        else:
            return [cls_loss], []



