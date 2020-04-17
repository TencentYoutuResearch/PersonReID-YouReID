import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck, resnet101, resnet152
from .backbones import senet
def make_model(args):
    return MGN(args)

class MGN2(nn.Module):
    def __init__(self, class_num=1000):
        super(MGN2, self).__init__()
        num_classes = class_num

        resnet = resnet101(pretrained=True)

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
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        #res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(resnet.layer4))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(resnet.layer4))

        pool2d = nn.AvgPool2d


        self.gap = nn.AdaptiveAvgPool2d(1)


        #self.maxpool_zg_p1 = pool2d(kernel_size=(24, 8))
        #self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        #self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.avgpool_zp1 = pool2d(kernel_size=(12, 8))
        self.avgpool_zp2 = pool2d(kernel_size=(8, 8))
        self.avgpool_zp3 = pool2d(kernel_size=(6, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))#, nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)

        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)

        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)

        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)
        self.reduction_8 = copy.deepcopy(reduction)

        self.reduction_9 = copy.deepcopy(reduction)
        self.reduction_10 = copy.deepcopy(reduction
)
        self.reduction_11 = copy.deepcopy(reduction)
        self.reduction_12 = copy.deepcopy(reduction)
        self.reduction_13 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        fc_layer = nn.Linear(256, num_classes)
        self._init_fc(fc_layer)

        self.fc_id_2048_0 = copy.deepcopy(fc_layer)
        self.fc_id_2048_1 = copy.deepcopy(fc_layer)
        self.fc_id_2048_2 = copy.deepcopy(fc_layer)
        self.fc_id_2048_3 = copy.deepcopy(fc_layer)

        self.fc_id_256_0_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_0_1 = copy.deepcopy(fc_layer)

        self.fc_id_256_1_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_1_1 = copy.deepcopy(fc_layer)
        self.fc_id_256_1_2 = copy.deepcopy(fc_layer)

        self.fc_id_256_2_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_2_1 = copy.deepcopy(fc_layer)

        self.fc_id_256_3_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_3_1 = copy.deepcopy(fc_layer)
        self.fc_id_256_3_2 = copy.deepcopy(fc_layer)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        #nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

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
        x = self.backone(x) # (64, 2048, 24, 8)

        p1 = self.p1(x)  # (64, 2048, 24, 8)
        p2 = self.p2(x)

        y1 = torch.cat([x[:, :, :12, :], x[:, :, 12:, :]], 0)
        y1 = self.p1(y1)
        n = y1.size(0) / 2
        y1 = torch.cat([y1[:n, :, :, :], y1[n:, :, :, :]], 2)

        y2 = torch.cat([x[:, :, :8, :], x[:, :, 8:16, :], x[:, :, 16:, :]], 0)
        y2 = self.p2(y2)
        n = y2.size(0) / 3
        y2 = torch.cat([y2[:n, :, :, :], y2[n:(2*n), :, :, :], y2[(2*n):, :, :, :]], 2)

        zg_p1 = self.gap(p1)  # (64, 2048, 1, 1)
        zg_p2 = self.gap(p2)

        zg_p3 = self.gap(y1)
        zg_p4 = self.gap(y2)    

        #print(zg_p1.shape)

        zp1 = self.avgpool_zp1(p1)
        #print(zp1.shape)
        z0_p1 = zp1[:, :, 0:1, :]
        z1_p1 = zp1[:, :, 1:2, :]

        zp2 = self.avgpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]
        z2_p2 = zp2[:, :, 2:3, :]

        zp3 = self.avgpool_zp1(y1)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]

        zp4 = self.avgpool_zp2(y2)
        z0_p4 = zp4[:, :, 0:1, :]
        z1_p4 = zp4[:, :, 1:2, :]
        z2_p4 = zp4[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)

        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        fg_p4 = self.reduction_3(zg_p4).squeeze(dim=3).squeeze(dim=2)

        f0_p1 = self.reduction_4(z0_p1).squeeze(dim=3).squeeze(dim=2)
        f1_p1 = self.reduction_5(z1_p1).squeeze(dim=3).squeeze(dim=2)

        f0_p2 = self.reduction_6(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_7(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f2_p2 = self.reduction_8(z2_p2).squeeze(dim=3).squeeze(dim=2)

        f0_p3 = self.reduction_9(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_10(z1_p3).squeeze(dim=3).squeeze(dim=2)

        f0_p4 = self.reduction_11(z0_p4).squeeze(dim=3).squeeze(dim=2)
        f1_p4 = self.reduction_12(z1_p4).squeeze(dim=3).squeeze(dim=2)
        f2_p4 = self.reduction_13(z2_p4).squeeze(dim=3).squeeze(dim=2)

        cat_p1 = torch.cat([f0_p1, f1_p1], 1)
        cat_p2 = torch.cat([f0_p2, f1_p2, f2_p2], 1)
        
        cat_p3 = torch.cat([f0_p3, f1_p3], 1)
        cat_p4 = torch.cat([f0_p4, f1_p4, f2_p4], 1)
        #print('cat_p1:', cat_p1.shape)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        l_p4 = self.fc_id_2048_3(fg_p4)

        l0_p1 = self.fc_id_256_0_0(f0_p1)
        l1_p1 = self.fc_id_256_0_1(f1_p1)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l2_p2 = self.fc_id_256_1_2(f2_p2)

        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)

        l0_p4 = self.fc_id_256_3_0(f0_p4)
        l1_p4 = self.fc_id_256_3_1(f1_p4)
        l2_p4 = self.fc_id_256_3_2(f2_p4)

        #predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        #predict = torch.cat([fg_p3, f0_p3, f1_p3, f2_p3], dim=1)
        #print('predict.shape:', predict.shape)
        #print('predict2.shape:', predict2.shape)
        predict = torch.cat([fg_p1, fg_p2, fg_p3, fg_p4, f0_p1, f1_p1, f0_p2, f1_p2, f2_p2, f0_p3, f1_p3, f0_p4, f1_p4, f2_p4], 1)
        #print('predict:', predict.shape)

        return predict, cat_p1, cat_p2, cat_p3, cat_p4, fg_p1, fg_p2, fg_p3, fg_p4, l_p1, l_p2, l_p3, l_p4, l0_p1, l1_p1, l0_p2, l1_p2, l2_p2, l0_p3, l1_p3, l0_p4, l1_p4, l2_p4
        # return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        #return predict, fg_p3, l_p3, l0_p3, l1_p3, l2_p3

