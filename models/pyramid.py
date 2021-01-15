import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os
from .backbones import model_zoo
from core.loss import *


class Pyramid(nn.Module):
    def __init__(
        self,
        last_stride=1,
        num_layers = 'resnet50',
        last_conv_dilation=1,
        num_stripes=6,
        used_levels=(1, 1, 1, 1, 1, 1),
        num_conv_out_channels=128,
        num_classes=0,
        loss_type=None,
        margin=0.5,
    ):

        super(Pyramid, self).__init__()

        print("num_stripes:{}".format(num_stripes))
        print("num_conv_out_channels:{},".format(num_conv_out_channels))

        self.resnet = model_zoo[num_layers](
            pretrained=True,
            last_stride=last_stride,
            )

        self.dropout_layer = nn.Dropout(p=0.2)

        self.num_classes = num_classes
        self.num_stripes = num_stripes
        self.used_levels = used_levels

        input_size0 = 2048
        self.pyramid_conv_list0 = nn.ModuleList()
        self.pyramid_fc_list0 = nn.ModuleList()
        Pyramid.basic_branch(self, num_conv_out_channels,
                                              input_size0,
                                              self.pyramid_conv_list0,
                                              self.pyramid_fc_list0)
        input_size1 = 1024
        self.pyramid_conv_list1 = nn.ModuleList()
        self.pyramid_fc_list1 = nn.ModuleList()
        Pyramid.basic_branch(self, num_conv_out_channels,
                                              input_size1,
                                              self.pyramid_conv_list1,
                                              self.pyramid_fc_list1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.tri_loss = TripletLoss(margin, normalize_feature=True)
        self.k_id = 0.0
        self.k_tri = 0.0
        self.loss_alpha = 0.25
        self.loss_sigma = 0.16
        self.loss_gamma = 2.0
        self.first_step = True

    def compute_loss(self, output, target ):
        logits_list, feats_list = output
        losses, losses_names = [], []
        cls_loss = torch.stack([self.ce_loss(logits, target)
                             for logits in logits_list], dim=0).sum()
        tri_loss = torch.stack([self.tri_loss(feats, target)
                               for feats in feats_list]).sum()

        new_k_id = self.loss_alpha * cls_loss.item() + (1.0 - self.loss_alpha) * self.k_id
        new_k_tri = self.loss_alpha * tri_loss.item() + (1.0 - self.loss_alpha) * self.k_tri

        if self.first_step:
            p_id = 0.5
            p_tri = 0.5
            self.first_step = False
        else:
            p_id = min(new_k_id, self.k_id) / self.k_id
            p_tri = min(new_k_tri, self.k_tri) / self.k_tri

        fl_id = - ( 1.0 - p_id) ** self.loss_gamma * np.log(p_id)
        fl_tri = - ( 1.0 - p_tri) ** self.loss_gamma * np.log(p_tri)
        self.k_id = new_k_id
        self.k_tri = new_k_tri

        if fl_tri / fl_id < self.loss_sigma:
            losses.append(cls_loss)
            losses_names.append('cls_loss')
        else:
            losses.append(cls_loss)
            losses_names.append('cls_loss')
            losses.append(tri_loss)
            losses_names.append('tri_loss')
        return losses, losses_names

    def forward(self, x, target):
        """
        Returns:
        feat_list: each member with shape [N, C]
        logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat0 = self.resnet(x)

        assert feat0.size(2) % self.num_stripes == 0
        feat_list = []
        logits_list = []

        Pyramid.pyramid_forward(self, feat0,
                                                 self.pyramid_conv_list0,
                                                 self.pyramid_fc_list0,
                                                 feat_list,
                                                 logits_list)

        """
        PCB_plus_dropout_pyramid.pyramid_forward(self, feat1,
                        self.pyramid_conv_list1,
                        self.pyramid_fc_list1,
                        feat_list,
                        logits_list)
        """
        #return feat_list, logits_list
        return logits_list, feat_list


    @ staticmethod
    def basic_branch(self, num_conv_out_channels,
                     input_size,
                     pyramid_conv_list,
                     pyramid_fc_list):
        # the level indexes are defined from fine to coarse,
        # the branch will contain one more part than that of its previous level
        # the sliding step is set to 1
        self.num_in_each_level = [i for i in range(self.num_stripes, 0, -1)]
        self.num_levels = len(self.num_in_each_level)
        self.num_branches = sum(self.num_in_each_level)

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            pyramid_conv_list.append(nn.Sequential(
                nn.Conv2d(input_size, num_conv_out_channels, 1),
                nn.BatchNorm2d(num_conv_out_channels),
                nn.ReLU(inplace=True)))

        idx_levels = 0
        for idx_branches in range(self.num_branches):
            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            fc = nn.Linear(num_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            pyramid_fc_list.append(fc)

    @staticmethod
    def pyramid_forward(self, feat,
                        pyramid_conv_list,
                        pyramid_fc_list,
                        feat_list,
                        logits_list):

        basic_stripe_size = int(feat.size(2) / self.num_stripes)

        idx_levels = 0
        used_branches = 0
        for idx_branches in range(self.num_branches):

            if idx_branches >= sum(self.num_in_each_level[0:idx_levels+1]):
                idx_levels += 1

            if self.used_levels[idx_levels] == 0:
                continue

            idx_in_each_level = idx_branches - \
                sum(self.num_in_each_level[0:idx_levels])

            stripe_size_in_level = basic_stripe_size * (idx_levels+1)

            st = idx_in_each_level * basic_stripe_size
            ed = st + stripe_size_in_level

            local_feat = F.avg_pool2d(feat[:, :, st: ed, :],
                                      (stripe_size_in_level, feat.size(-1))) + F.max_pool2d(feat[:, :, st: ed, :],
                                                                            (stripe_size_in_level, feat.size(-1)))

            local_feat = pyramid_conv_list[used_branches](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)

            local_logits = pyramid_fc_list[used_branches](
                self.dropout_layer(local_feat))
            logits_list.append(local_logits)

            used_branches += 1
