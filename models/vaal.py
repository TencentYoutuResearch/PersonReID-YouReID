import copy
import torch
from torch import nn
from core.loss import *
from .backbones import model_zoo


class VAALBase(nn.Module):


    def __init__(self,
                 last_stride=1,
                 use_non_local=False,
                 num_classes=1000,
                 stripes=None,
                 num_layers=50,
                 loss_type=None,
                 margin=0.5
                 ):
        super(VAALBase, self).__init__()
        if loss_type is None:
            loss_type = ['softmax', 'triplet']
        if stripes is None:
            stripes = [1, 2, 3]
        self.stripes = stripes
        self.margin = margin
        self.loss_type = loss_type
        self.num_classes = num_classes
        kwargs = {
            'use_non_local': use_non_local
        }
        resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
            **kwargs
        )
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))  # , nn.ReLU())
        self._init_reduction(reduction)
        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(256, num_classes))
        self._init_fc(fc_layer)

        branches = []
        for stripe in stripes:
            # if stripe == 1:
            #     num_fc = stripe
            # else:
            #     num_fc = stripe + 1
            num_fc = stripe
            embedding_layers = nn.ModuleList([copy.deepcopy(reduction) for _ in range(num_fc)])
            fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(num_fc)])
            branches.append(
                nn.ModuleList([
                    nn.Sequential(copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4)),
                    embedding_layers, fc_layers])
            )
        self.branches = nn.ModuleList(branches)

        if 'softmax' in self.loss_type:
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            elif 'adaptiveLSR' in self.loss_type:
                self.ce_loss = AdapativeCrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin, normalize_feature=not 'circle' in self.loss_type)
        if 'soft_triplet' in self.loss_type:
            self.tri_loss = SoftTripletLoss(margin, normalize_feature=not 'circle' in self.loss_type)

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

    def forward(self, x, label=None):
        '''
        ('input.shape:', (64, 3, 384, 128))
        '''
        x = self.backone(x)
        global_logits, local_logits, tri_logits = [], [], []
        for idx, stripe in enumerate(self.stripes):
            branch = self.branches[idx]
            backbone, reduces, fcs = branch
            net = backbone(x)
            if stripe == 1:
                # global
                global_feat = self.gap(net)
                global_feat_reduce = reduces[0](global_feat).squeeze(dim=3).squeeze(dim=2)
                if self.training:
                    global_feat_logit = fcs[0](global_feat_reduce)
                    global_logits.append(global_feat_logit)
                tri_logits.append(global_feat_reduce)
            else:
                # local
                local_tri_logits = []
                for i in range(stripe):
                    stride = 24 // stripe
                    local_feat = net[:, :, i*stride: (i+1)*stride, :]
                    local_feat = self.gap(local_feat)
                    local_feat_reduce = reduces[i](local_feat).squeeze(dim=3).squeeze(dim=2)
                    if self.training:
                        local_feat_logit = fcs[i](local_feat_reduce)
                        local_logits.append(local_feat_logit)
                    local_tri_logits.append(local_feat_reduce)
                tri_logits.append(torch.cat(local_tri_logits, dim=1))

        if self.training:
            return global_logits, local_logits, tri_logits
        else:
            return torch.cat(tri_logits, dim=1)

    def compute_loss(self, output, target):
        global_logits, local_logits, tri_logits = output
        ce_logits = global_logits + local_logits
        losses, losses_names = [], []
        for ce_id, ce_logit in enumerate(ce_logits):
            cls_loss = self.ce_loss(ce_logit, target)
            losses.append(cls_loss)
            losses_names.append('cls_%d' % ce_id)
        for tri_id, tri_logit in enumerate(tri_logits):
            tri_loss = self.tri_loss(tri_logit, target)
            losses.append(tri_loss)
            losses_names.append('tri_%d' % tri_id)
        return losses, losses_names


class VAAL(VAALBase):


    def __init__(self,
                 last_stride=1,
                 use_non_local=False,
                 num_classes=1000,
                 stripes=None,
                 num_layers=50,
                 loss_type=None,
                 margin=0.5,
                 epsilon=0.2,
                 adaptive=True
                 ):
        super(VAAL, self).__init__(last_stride=last_stride,
                                   use_non_local=use_non_local,
                                   num_classes=num_classes,
                                   stripes=stripes,
                                   num_layers=num_layers,
                                   loss_type=loss_type,
                                   margin=margin
                                   )
        assert 1 in self.stripes
        stripe_1_ind = self.stripes.index(1)
        self.branches[stripe_1_ind][-1][0][-1] = nn.Linear(256, self.num_classes * 3)   # three directions

        assert 'lsr_direct' in self.loss_type
        self.direct_ce_loss = LSRWithDirection(num_classes, epsilon=epsilon, adaptive=adaptive)

    def compute_loss(self, output, targets):
        global_logits, local_logits, tri_logits = output
        id_targets, direct_targets = targets

        losses, losses_names = [], []
        cls_loss = self.direct_ce_loss(global_logits[0], targets)
        losses.append(cls_loss)
        losses_names.append('direct_cls')

        for ce_id, ce_logit in enumerate(local_logits):
            cls_loss = self.ce_loss(ce_logit, id_targets)
            losses.append(cls_loss)
            losses_names.append('cls_%d' % ce_id)
        for tri_id, tri_logit in enumerate(tri_logits):
            tri_loss = self.tri_loss(tri_logit, id_targets)
            losses.append(tri_loss)
            losses_names.append('tri_%d' % tri_id)
        return losses, losses_names


class VAALArcface(nn.Module):


    def __init__(self,
                 last_stride=1,
                 use_non_local=False,
                 num_classes=1000,
                 stripes=None,
                 num_layers=50,
                 loss_type=None,
                 margin=0.5
                 ):
        super(VAALArcface, self).__init__()
        if loss_type is None:
            loss_type = ['arcface', 'triplet']
        if stripes is None:
            stripes = [1, 2, 3]
        self.stripes = stripes
        self.margin = margin
        self.loss_type = loss_type
        self.num_classes = num_classes
        kwargs = {
            'use_non_local': use_non_local
        }
        resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
            **kwargs
        )
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))  # , nn.ReLU())
        self._init_reduction(reduction)
        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(256, num_classes))
        self._init_fc(fc_layer)

        branches = []
        for stripe in stripes:
            if stripe == 1:
                num_fc = stripe
                embedding_layers = copy.deepcopy(reduction)
                fc_layer =
            fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(num_fc)])
            branches.append(
                nn.ModuleList([
                    nn.Sequential(copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4)),
                    embedding_layers, fc_layers])
            )
        self.branches = nn.ModuleList(branches)

        if 'softmax' in self.loss_type:
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            elif 'adaptiveLSR' in self.loss_type:
                self.ce_loss = AdapativeCrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin, normalize_feature=not 'circle' in self.loss_type)
        if 'soft_triplet' in self.loss_type:
            self.tri_loss = SoftTripletLoss(margin, normalize_feature=not 'circle' in self.loss_type)

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

    def forward(self, x, label=None):
        '''
        ('input.shape:', (64, 3, 384, 128))
        '''
        x = self.backone(x)
        global_logits, local_logits, tri_logits = [], [], []
        for idx, stripe in enumerate(self.stripes):
            branch = self.branches[idx]
            backbone, reduces, fcs = branch
            net = backbone(x)
            if stripe == 1:
                # global
                global_feat = self.gap(net)
                global_feat_reduce = reduces[0](global_feat).squeeze(dim=3).squeeze(dim=2)
                if self.training:
                    global_feat_logit = fcs[0](global_feat_reduce)
                    global_logits.append(global_feat_logit)
                tri_logits.append(global_feat_reduce)
            else:
                # local
                local_tri_logits = []
                for i in range(stripe):
                    stride = 24 // stripe
                    local_feat = net[:, :, i*stride: (i+1)*stride, :]
                    local_feat = self.gap(local_feat)
                    local_feat_reduce = reduces[i](local_feat).squeeze(dim=3).squeeze(dim=2)
                    if self.training:
                        local_feat_logit = fcs[i](local_feat_reduce)
                        local_logits.append(local_feat_logit)
                    local_tri_logits.append(local_feat_reduce)
                tri_logits.append(torch.cat(local_tri_logits, dim=1))

        if self.training:
            return global_logits, local_logits, tri_logits
        else:
            return torch.cat(tri_logits, dim=1)