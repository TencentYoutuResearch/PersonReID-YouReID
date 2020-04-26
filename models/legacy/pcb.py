from models.legacy.resnet import *
import torch.nn as nn
import torch.nn.functional as F
class PCB(nn.Module):
    def __init__(self, class_num=1000, backbone=resnet50, pretrained=True, is_for_test=False, stripe=6):
        super(PCB,self).__init__()
        self.pretrained = pretrained
        self.is_for_test = is_for_test
        self.backbone = backbone(pretrained=pretrained, remove=True, last_stride=1)
        self.stripe = stripe
        self.new = nn.ModuleList()

        down = nn.Sequential(nn.Conv2d(2048, 256, 1),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU(inplace=True))


        self.new.add_module('down', down)

        if self.is_for_test is False:
            fc_list = nn.ModuleList([nn.Linear(256, class_num) for _ in range(stripe)])
            self.new.add_module('fc_list', fc_list)

    def _init_parameters(self):
        for m in self.new.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_param(self, lr):
        new_param = list(self.new.parameters())
        # return new_param
        new_param_id = [id(p) for p in new_param]
        finetuned_params = []
        for p in self.parameters():
            if id(p) not in new_param_id:
                finetuned_params.append(p)
        return [{'params': new_param, 'lr': lr},
                {'params': finetuned_params, 'lr': 1e-1 * lr}]

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        assert x.size(2) % self.stripe == 0
        stripe_h = x.size(2) // self.stripe
        local_feat_list = []
        logits_list = []

        for i in range(self.stripe):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
                x[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                (stripe_h, x.size(-1)))
            # shape [N, c, 1, 1]

            local_feat = self.new.down(local_feat)  # independent conv
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if self.is_for_test is False:
                logits_list.append(self.new.fc_list[i](local_feat))

        if self.is_for_test is True:
            return local_feat_list

        return logits_list, local_feat_list

