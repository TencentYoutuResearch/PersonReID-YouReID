import copy

import torch
from torch import nn

from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.resnet_ibn_a import resnet101_ibn_a
from .backbones.senet import se_resnext101_32x4d
from torchvision.models.resnet import resnet50, resnet101

def make_model(args):
    return MGNv1(args)

class MGNv3(nn.Module):
    def __init__(self, class_num=1000):
        super(MGNv3, self).__init__()
        num_classes = class_num

        # resnet = se_resnet101_ibn_a(pretrained=True)
        resnet = resnet101_ibn_a(pretrained=True)
        # resnet = se_resnext101_32x4d()

        # Modifiy the stride of last conv layer
        resnet.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        resnet.layer4[0].downsample = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                                                    nn.BatchNorm2d(2048))
        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2[0],
        )

        res_conv3 = nn.Sequential(*resnet.layer2[1:])


        self.p1 = nn.Sequential(copy.deepcopy(res_conv3), copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv3), copy.deepcopy(resnet.layer3), copy.deepcopy(resnet.layer4))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.avgpool_zp1 = nn.AvgPool2d(kernel_size=(12, 8)) #384x128
        # self.avgpool_zp1 = nn.AvgPool2d(kernel_size=(8, 8))  # 256x128

        reduction = nn.Sequential(nn.BatchNorm2d(2048), nn.Conv2d(2048, 384, 1, bias=False), nn.BatchNorm2d(384))#, nn.ReLU())
        self._init_reduction(reduction)

        self.reduction_g_0 = copy.deepcopy(reduction)
        self.reduction_g_1 = copy.deepcopy(reduction)

        self.reduction_l_0_0 = copy.deepcopy(reduction)
        self.reduction_l_0_1 = copy.deepcopy(reduction)

        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(384, num_classes))
        self._init_fc(fc_layer)

        self.fc_id_g_0 = copy.deepcopy(fc_layer)
        self.fc_id_g_1 = copy.deepcopy(fc_layer)

        self.fc_id_l_0_0 = copy.deepcopy(fc_layer)
        self.fc_id_l_0_1 = copy.deepcopy(fc_layer)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        if len(reduction) == 3:
            # bn
            nn.init.constant_(reduction[0].weight, 1.)
            nn.init.constant_(reduction[0].bias, 0.)
            #
            nn.init.kaiming_normal_(reduction[1].weight, mode='fan_in')
            # bn
            nn.init.constant_(reduction[2].weight, 1.)
            nn.init.constant_(reduction[2].bias, 0.)
        elif len(reduction) == 2:
            nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
            # bn
            nn.init.constant_(reduction[1].weight, 1.)
            nn.init.constant_(reduction[1].bias, 0.)
        else:
            # bn
            nn.init.constant_(reduction[0].weight, 1.)
            nn.init.constant_(reduction[0].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x):
        '''
        ('input.shape:', (64, 3, 384, 128))
        ('x:', (64, 1024, 24, 8))
        ('p1:', (64, 2048, 12, 4))
        ('p2:', (64, 2048, 24, 8))
        ('p3:', (64, 2048, 24, 8))
        ('zg_p1:', (64, 2048, 1, 1))
        ('zg_p2:', (64, 2048, 1, 1))
        ('zg_p3:', (64, 2048, 1, 1))
        ('zp2:', (64, 2048, 2, 1))
        ('zp3:', (64, 2048, 3, 1))
        '''
        x = self.backone(x) #(64, 1024, 24, 8)
        h = x.size(2)
        y1 = torch.cat([x[:, :, :h//2, :], x[:, :, h//2:, :]], 0)
        y1 = self.p1(y1)  # (64, 2048, 24, 8)
        n1 = y1.size(0) // 2
        y1 = torch.cat([y1[:n1, :, :, :], y1[n1:, :, :, :]], 2)

        y2 = self.p2(x)

        zg_p1 = self.gap(y1)  # (64, 2048, 1, 1)
        zg_p2 = self.gap(y2)

        zp1 = self.avgpool_zp1(y1)
        z0_p1 = zp1[:, :, 0:1, :]
        z1_p1 = zp1[:, :, 1:2, :]


        fg_p1 = self.reduction_g_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_g_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        global_triplet = [fg_p1, fg_p2]

        #print(z0_p1.size(), z1_p1.size())
        f0_p1 = self.reduction_l_0_0(z0_p1).squeeze(dim=3).squeeze(dim=2)
        f1_p1 = self.reduction_l_0_1(z1_p1).squeeze(dim=3).squeeze(dim=2)

        local_triplet = [
            torch.cat([f0_p1, f1_p1], 1),
        ]

        #
        l_p1 = self.fc_id_g_0(fg_p1)
        l_p2 = self.fc_id_g_1(fg_p2)
        global_softmax = [l_p1, l_p2]

        local_softmaxs = []
        l0_p1 = self.fc_id_l_0_0(f0_p1)
        l1_p1 = self.fc_id_l_0_1(f1_p1)
        local_softmaxs.append([l0_p1, l1_p1])

        return global_softmax, local_softmaxs, global_triplet, local_triplet




