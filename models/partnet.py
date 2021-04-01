import copy
import torch
from torch import nn
from .backbones import model_zoo
from core.loss import *

class PartNet(nn.Module):


    def __init__(self,
                 num_classes=1000,
                 num_layers=50,
                 last_stride=1,
                 reduce_dim=256,
                 stripe=6,
                 loss_type=None,
                 margin=0.5,
                 lamb=0.8,
                 p_w=1.,
                 g_w=1.
                 ):
        super(PartNet, self).__init__()
        self.resnet = model_zoo[num_layers](
            pretrained=True, last_stride=last_stride,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.stripe = stripe

        local_embedding_layer = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduce_dim)
        )
        nn.init.kaiming_normal_(local_embedding_layer[0].weight, mode='fan_out')

        id_fc_layer = nn.Sequential(nn.Linear(reduce_dim, num_classes))
        part_fc_layer = nn.Sequential(nn.Linear(reduce_dim, 2))

        self.local_embedding_layers = nn.ModuleList([copy.deepcopy(local_embedding_layer) for _ in range(stripe)])
        self.local_fc_layers = nn.ModuleList([copy.deepcopy(id_fc_layer) for _ in range(stripe)])
        self.part_embedding_layers = nn.ModuleList([copy.deepcopy(local_embedding_layer) for _ in range(stripe)])
        self.part_fc_layers = nn.ModuleList([copy.deepcopy(part_fc_layer) for _ in range(stripe)])

        self.global_embedding_layer = nn.Sequential(
            nn.Conv2d(4096, reduce_dim * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduce_dim * 2)
        )
        nn.init.kaiming_normal_(self.global_embedding_layer[0].weight, mode='fan_out')
        self.global_fc_layer = nn.Sequential(nn.Linear(reduce_dim * 2, num_classes))

        self.loss_type = loss_type
        if 'softmax' in self.loss_type:
            if 'labelsmooth' in self.loss_type:
                self.ce_loss = CrossEntropyLabelSmooth(num_classes)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

        if 'triplet' in self.loss_type:
            self.tri_loss = TripletLoss(margin)
        if 'soft_triplet' in self.loss_type:
            self.tri_loss = SoftTripletLoss(margin)

        if 'vgtri' in self.loss_type:
            self.use_vgtri = True
            self.vgtri_loss = VGTripletLoss(margin)
        else:
            self.use_vgtri = False

        self.lamb =lamb
        self.p_w, self.g_w = p_w, g_w

    def forward(self, x, label=None):
        x = self.resnet(x)
        h = x.size(2)
        # print(x.shape)
        local_id_logits, local_triplet_logits, part_softmax_logits = [], [], []

        stride = h // self.stripe
        for i in range(self.stripe):
            s = x[:, :, i * stride:(i + 1) * stride, :]
            s = self.gap(s)
            s1 = self.local_embedding_layers[i](s).squeeze(dim=3).squeeze(dim=2)
            t1 = self.local_fc_layers[i](s1)
            local_triplet_logits.append(s1)
            local_id_logits.append(t1)

            s2 = self.part_embedding_layers[i](s).squeeze(dim=3).squeeze(dim=2)
            t2 = self.part_fc_layers[i](s2)
            part_softmax_logits.append(t2)

        # triplet_logits = [torch.cat(local_triplet_logits, dim=1)]

        g1, g2 = self.gap(x), self.gmp(x)
        x = torch.cat([g1, g2], dim=1)
        x = self.global_embedding_layer(x).squeeze(dim=3).squeeze(dim=2)
        # triplet_logits.append(x)

        gloabl_id_logits = self.global_fc_layer(x)

        if self.training:
            return gloabl_id_logits, local_id_logits, x, local_triplet_logits, part_softmax_logits
        else:
            return x, torch.stack(local_triplet_logits, dim=1), torch.softmax(torch.stack(part_softmax_logits, dim=1), dim=-1)

    def compute_loss(self, output, target):
        gloabl_id_logits, local_id_logits, global_triplet_logits, local_triplet_logits, part_softmax_logits = output
        losses, losses_names = [], []
        label, part_label = target

        cls_loss = self.ce_loss(gloabl_id_logits, label)
        losses.append(cls_loss * (1 - self.lamb))  #
        losses_names.append('cls_loss')

        for idx, logit in enumerate(local_id_logits):
            cls_loss = self.ce_loss(logit, label)
            losses.append(cls_loss * self.lamb)  #
            losses_names.append('local_cls_%d' % idx)

        for idx, logit in enumerate(part_softmax_logits):
            cls_loss = self.ce_loss(logit, part_label[:, idx])
            losses.append(cls_loss)
            losses_names.append('part_loss_%d' % idx)

        local_tri_loss = self.tri_loss(torch.cat(local_triplet_logits, dim=1), label)
        losses.append(local_tri_loss * self.p_w)
        losses_names.append('local_tri_loss')

        vgtri = self.vgtri_loss((global_triplet_logits, torch.stack(local_triplet_logits, dim=1)), target)
        losses.append(vgtri * self.g_w)
        losses_names.append('vgtri')

        return losses, losses_names




