import copy
import torch
from torch import nn
from .backbones import model_zoo


class PCB(nn.Module):


    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=256,
                 stripe=6,
                 use_non_local=False):

        super(PCB, self).__init__()
        kwargs = {
            'use_non_local': use_non_local
        }
        self.resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
            **kwargs
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

    def compute_loss(self, output, target):
        ce_logit, tri_logit = output
        losses, losses_names = [], []
        if 'softmax' in self.loss_type or 'arcface' in self.loss_type:
            cls_loss = self.ce_loss(ce_logit[0], target)
            losses.append(cls_loss)
            losses_names.append('cls_loss')
        if len(set(['triplet',  'soft_triplet']) & set(self.loss_type)) == 1:
            tri_loss = self.tri_loss(tri_logit[0], target)
            losses.append(tri_loss)
            losses_names.append('tri_loss')
        return losses, losses_names