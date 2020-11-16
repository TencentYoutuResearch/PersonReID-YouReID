#conding=utf-8
# @Time  : 2019/12/23 10:57
# @Author: fufuyu
# @Email:  fufuyu@tencen.com

from .backbones import model_zoo
from core.loss import *
from core.layers import GeneralizedMeanPoolingP

class SWAV(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=768,
                 pool_type='baseline',
                 epsilon=0.05,
                 temperature=0.1,
                 use_non_local=False
                 ):
        super(SWAV, self).__init__()
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

        self.fc = nn.Linear(reduce_dim, num_classes)
        self.epsilon = epsilon
        self.temperature = temperature

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn[0].weight, 1.)
        nn.init.constant_(bn[0].bias, 0.)

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
        y = self.fc(x)
        return y, x

    def compute_loss(self, output):
        ce_logits, tri_logits = output
        ce_logits = torch.split(ce_logits, split_size_or_sections=2)
        # tri_logits = torch.split(tri_logits, split_size_or_sections=2)
        # target = torch.split(targetet, split_size_or_sections=2)
        bs = ce_logits[0].size(0)
        loss = 0
        for i in range(2):
            with torch.no_grad():
                out = ce_logits[i]
                q = torch.exp(out / self.epsilon).t()
                q = distributed_sinkhorn(q, 3)[-bs:]
            p = torch.softmax(ce_logits[1-i] / self.temperature, dim=-1)
            loss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
        loss /= 2.

        return [loss],  []



