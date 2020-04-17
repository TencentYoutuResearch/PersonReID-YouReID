import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck, resnet101

from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck, se_resnet101




class MGN_SENet(nn.Module):
    def __init__(self, class_num=1000):
        super(MGN_SENet, self).__init__()
        num_classes = class_num


        base = se_resnet101()



        start_pos = -2
        modules = list(base.children())
        self.backone = nn.Sequential(
            base.layer0,
            base.layer1,
            base.layer2,
            base.layer3,
        )
        res_conv4 = nn.Sequential(
            base.layer4,
        )

        self.p1 = copy.deepcopy(res_conv4)
        self.p2 = copy.deepcopy(res_conv4)
        self.p3 = copy.deepcopy(res_conv4)


        pool2d = nn.MaxPool2d


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
        self.reduction_10 = copy.deepcopy(reduction)
        self.reduction_11 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        fc_layer = nn.Linear(256, num_classes)
        self._init_fc(fc_layer)

        self.fc_id_2048_0 = copy.deepcopy(fc_layer)
        self.fc_id_2048_1 = copy.deepcopy(fc_layer)
        self.fc_id_2048_2 = copy.deepcopy(fc_layer)

        self.fc_id_256_0_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_0_1 = copy.deepcopy(fc_layer)
        self.fc_id_256_1_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_1_1 = copy.deepcopy(fc_layer)
        self.fc_id_256_1_2 = copy.deepcopy(fc_layer)
        self.fc_id_256_2_0 = copy.deepcopy(fc_layer)
        self.fc_id_256_2_1 = copy.deepcopy(fc_layer)
        self.fc_id_256_2_2 = copy.deepcopy(fc_layer)
        self.fc_id_256_2_3 = copy.deepcopy(fc_layer)

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
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
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
        x = self.backone(x)
        #print('x.shape:', x.shape)

        p1 = self.p1(x)  # (64, 2048, 24, 8)
        p2 = self.p2(x)
        p3 = self.p3(x)
        #print('p1.shape:', p1.shape)

        zg_p1 = self.gap(p1)  # (64, 2048, 1, 1)
        zg_p2 = self.gap(p2)
        zg_p3 = self.gap(p3)
        #print('zg_p1.shape:', zg_p1.shape)

        zp1 = self.avgpool_zp1(p1)
        z0_p1 = zp1[:, :, 0:1, :]
        z1_p1 = zp1[:, :, 1:2, :]

        zp2 = self.avgpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]
        z2_p2 = zp2[:, :, 2:3, :]

        zp3 = self.avgpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        z3_p3 = zp3[:, :, 3:4, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p1 = self.reduction_3(z0_p1).squeeze(dim=3).squeeze(dim=2)
        f1_p1 = self.reduction_4(z1_p1).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_5(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_6(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f2_p2 = self.reduction_7(z2_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_8(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_9(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_10(z2_p3).squeeze(dim=3).squeeze(dim=2)
        f3_p3 = self.reduction_11(z3_p3).squeeze(dim=3).squeeze(dim=2)

        cat_p1 = torch.cat([f0_p1, f1_p1], 1)
        cat_p2 = torch.cat([f0_p2, f1_p2, f2_p2], 1)
        cat_p3 = torch.cat([f0_p3, f1_p3, f2_p3, f3_p3], 1)
        #print('cat_p1:', cat_p1.shape)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p1 = self.fc_id_256_0_0(f0_p1)
        l1_p1 = self.fc_id_256_0_1(f1_p1)
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l2_p2 = self.fc_id_256_1_2(f2_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        l3_p3 = self.fc_id_256_2_3(f3_p3)

        #predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        #predict = torch.cat([fg_p3, f0_p3, f1_p3, f2_p3], dim=1)
        #print('predict.shape:', predict.shape)
        #print('predict2.shape:', predict2.shape)
        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p1, f1_p1, f0_p2, f1_p2, f2_p2, f0_p3, f1_p3, f2_p3, f3_p3], 1)
        #print('predict:', predict.shape)

        return predict, cat_p1, cat_p2, cat_p3, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p1, l1_p1, l0_p2, l1_p2, l2_p2, l0_p3, l1_p3, l2_p3, l3_p3
        # return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        #return predict, fg_p3, l_p3, l0_p3, l1_p3, l2_p3
    def load_param(self, trained_path):
        print('[baseline]  Imagenet pretrained')
        param_dict = torch.load(trained_path)
        #print('loading from ', trained_path)
        #print('++++++++++++++++\n\n')
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
