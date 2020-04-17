import copy

import torch
from torch import nn

from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.senet import se_resnext101_32x4d
from torchvision.models.resnet import resnet50

def make_model(args):
    return BA(args)

class BA(nn.Module):
    def __init__(self, class_num=1000):
        super(BA, self).__init__()
        num_classes = class_num

        resnet = se_resnet101_ibn_a(pretrained=True)

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
            resnet.layer3,
            resnet.layer4
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.reduction = nn.Sequential(nn.BatchNorm2d(2048), nn.Dropout(), nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512))#, nn.ReLU())
        self._init_reduction(self.reduction)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        if len(reduction) == 4:
            # bn
            nn.init.constant_(reduction[0].weight, 1.)
            nn.init.constant_(reduction[0].bias, 0.)
            #
            nn.init.kaiming_normal_(reduction[2].weight, mode='fan_in')
            # bn
            nn.init.constant_(reduction[-1].weight, 1.)
            nn.init.constant_(reduction[-1].bias, 0.)


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
        x = self.gap(x)
        
        logit = self.reduction(x).squeeze(dim=3).squeeze(dim=2)
        #print(x.size())

        return logit




