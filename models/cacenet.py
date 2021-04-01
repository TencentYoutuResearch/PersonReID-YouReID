from core.loss import *
from core.layers import GeneralizedMeanPoolingP, PairGraph
from .backbones import model_zoo


class CACENET(nn.Module):


    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=768,
                 pool_type='baseline',
                 margin=0.5,
                 alpha=0.9,
                 use_non_local=False
                 ):
        super(CACENET, self).__init__()
        kwargs = {
            'use_non_local': use_non_local
        }
        self.resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
            **kwargs
        )
        self.alpha = alpha
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

        self.pair_graph = PairGraph(2048)
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(input_dim, reduce_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduce_dim)
        )
        nn.init.kaiming_normal_(self.embedding_layer[0].weight, mode='fan_out')
        self._init_bn(self.embedding_layer[1])

        self.fc_layer = nn.Sequential(nn.ReLU(inplace=True),
                                      nn.Dropout(),
                                      nn.Linear(reduce_dim, num_classes, bias=False))
        self._init_fc(self.fc_layer)
        self.pair_fc_layer = nn.Sequential(nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(reduce_dim, num_classes, bias=False))
        self._init_fc(self.pair_fc_layer)

        self.ce_loss_ls = CrossEntropyLabelSmooth(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

        self.tri_loss = TripletLoss(margin, normalize_feature=True)
        self.pair_tri_loss = PairTripletLoss(margin, normalize_feature=True)
        # self.tri_loss = SoftTripletLoss(margin, normalize_feature=True)

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn.weight, 1.)
        nn.init.constant_(bn.bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[-1].weight, std=0.001)
        # nn.init.constant_(fc[1].bias, 0.)

    def get_pair_feature(self, x):
        b = x.size(0)
        x1 = x.repeat(b, 1, 1, 1)
        x2 = x.repeat_interleave(b, dim=0)

        return torch.cat([x1, x2], dim=-1)

    def get_pair_label(self, x):
        b = x.size(0)
        x1 = x.repeat(b)
        x2 = x.repeat_interleave(b, dim=0)

        return x1, x2

    def head(self, x):
        if self.pool_type == 'gemm':
            x = self.gemp(x)
        elif self.pool_type == 'baseline':
            x1 = self.gap(x)
            x2 = self.gmp(x)
            x = torch.cat([x1, x2], 1)

        x = self.embedding_layer(x)
        x = x.squeeze(dim=3).squeeze(dim=2)

        return x


    def forward(self, x, label=None):
        x = self.resnet(x)
        _, _, _, w = x.size()
        feat_pair = self.get_pair_feature(x)
        feat_pair = self.pair_graph(feat_pair)
        # print(feat_pair.size())
        feat_0, feat_1 = feat_pair[:, :, :, :w], feat_pair[:, :, :, w:]
        f = self.head(x)
        f_0 = self.head(feat_0)
        f_1 = self.head(feat_1)

        if self.training:
            logit = self.fc_layer(f)
            logit_0 = self.pair_fc_layer(f_0)
            logit_1 = self.pair_fc_layer(f_1)
            return f, f_0, f_1, logit, logit_0, logit_1
        else:
            return f

    def compute_loss(self, output, target):
        f, f_0, f_1, logit, logit_0, logit_1 = output
        losses, losses_names = [], []
        # logit = self.fc_layer(f)
        loss = self.ce_loss_ls(logit, target)
        losses.append(loss)
        losses_names.append('cls_loss')

        target_0, target_1 = self.get_pair_label(target)

        # logit_0 = self.pair_fc_layers[0](f_0)
        loss_0_0 = self.ce_loss(logit_0, target_0)
        loss_0_1 = self.ce_loss(logit_0, target_1)
        loss_0 = self.alpha * loss_0_0 + (1 - self.alpha) * loss_0_1
        losses.append(loss_0)
        losses_names.append('mixup_loss_0')

        # logit_1 = self.pair_fc_layers[1](f_1)
        loss_1_0 = self.ce_loss(logit_1, target_1)
        loss_1_1 = self.ce_loss(logit_1, target_0)
        loss_1 = self.alpha * loss_1_0 + (1 - self.alpha) * loss_1_1
        losses.append(loss_1)
        losses_names.append('mixup_loss_1')

        tri_loss = self.tri_loss(f, target)
        losses.append(tri_loss)
        losses_names.append('tri_loss')

        pair_tri_loss = self.pair_tri_loss(f_0, f_1, target)
        losses.append(pair_tri_loss)
        losses_names.append('pair_tri_loss')

        return losses, losses_names
