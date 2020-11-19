#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

from .backbones import model_zoo
from core.loss import *
from core.layers import GeneralizedMeanPoolingP

class Baseline(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=768,
                 pool_type='baseline',
                 loss_type=['softmax, triplet'],
                 margin=0.5,
                 use_non_local=False
                 ):
        super(Baseline, self).__init__()
        kwargs = {
            'use_non_local': use_non_local
        }
        self.resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
            **kwargs
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
            self.fc_layer = ArcMarginProduct(reduce_dim, num_classes)
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()  # .cuda()
        elif 'circle' in self.loss_type:
            self.fc_layer = Circle(num_classes, reduce_dim)
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()  # .cuda()
        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin, normalize_feature=not 'circle' in self.loss_type) #.cuda()
        if 'multisimilarity' in self.loss_type:
            self.tri_loss = MultiSimilarityLoss()
    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn[0].weight, 1.)
        nn.init.constant_(bn[0].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x, label=None):
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
        if self.training:
            if 'softmax' in self.loss_type:
                y = self.fc_layer(x)
                return [y], [x]
            elif 'arcface' in self.loss_type or 'circle' in self.loss_type:
                y = self.fc_layer(x, label)
                return [y], [x]
        else:
            return [x], [x]

    def compute_loss(self, output, target):
        ce_logit, tri_logit = output
        cls_losses, tri_losses = [], []
        if 'softmax' in self.loss_type or 'arcface' in self.loss_type:
            cls_loss = self.ce_loss(ce_logit[0], target)
            cls_losses.append(cls_loss)
        if 'triplet' in self.loss_type or 'multisimilarity' in self.loss_type:
            tri_loss = self.tri_loss(tri_logit[0], target)
            tri_losses.append(tri_loss)
        return cls_losses, tri_losses



