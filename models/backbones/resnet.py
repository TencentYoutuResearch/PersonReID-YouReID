#conding=utf-8
# @Time  : 2020/4/26 15:28
# @Author: fufuyu
# @Email:  fufuyu@tencent.com

import math
import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.ops.deform_conv import DeformConv2d
from core.layers import NonLocal

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 use_non_local=False
                 ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # GCNet

        self.use_non_local = use_non_local
        if self.use_non_local:
            self.non_local_block = NonLocal(planes * self.expansion,
                                       planes * self.expansion // 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.use_non_local:
            out = self.non_local_block(out)

        return out



class DCNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 use_non_local=False
                 ):
        super(DCNBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # GCNet

        self.use_non_local = use_non_local
        if self.use_non_local:
            self.non_local_block = NonLocal(planes * self.expansion,
                                       planes * self.expansion // 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.use_non_local:
            out = self.non_local_block(out)

        return out

class ResNet(nn.Module):
    def __init__(self, last_stride,
                 block, layers, model_name, use_non_local=False,
                 groups=1, width_per_group=64):

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self._model_name = model_name
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_non_local=use_non_local)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=last_stride, use_non_local=use_non_local)

    def _make_layer(self, block, planes, blocks, stride=1,
                    use_non_local=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride,
                            downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_non_local_flag = use_non_local and i == blocks - 2
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                use_non_local = use_non_local_flag
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_pretrain(self, model_path=''):
        with_model_path = (model_path is not '')
        if not with_model_path:  # resnet pretrain
            print('Download from', model_urls[self._model_name])
            state_dict = model_zoo.load_url(model_urls[self._model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.load_state_dict(state_dict, strict=False)
        else:
            # ibn pretrain
            print('load from ', model_path)
            state_dict = torch.load(model_path)['state_dict']
            state_dict.pop('module.fc.weight')
            state_dict.pop('module.fc.bias')
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])  # remove module in name
                if self.state_dict()[new_k].shape == state_dict[k].shape:
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            self.load_state_dict(state_dict, strict=False)


    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _resnet(pretrained, last_stride, block,
            layers, model_name, model_path='',**kwargs):
    """"""
    model = ResNet(last_stride,
                   block, layers, model_name,**kwargs)
    if pretrained:
        model.load_pretrain(model_path)
    return model

def resnet50(pretrained=False, last_stride=1, use_non_local=False, model_path=''):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 4, 6, 3], use_non_local=use_non_local,
                   model_path=model_path, model_name='resnet50')

def resnet101(pretrained=False, last_stride=1, use_non_local=False, model_path=''):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet(pretrained=pretrained, last_stride=last_stride, use_non_local=use_non_local,
                   block=Bottleneck, layers=[3, 4, 23, 3], model_path=model_path, model_name='resnet101')

def resnet152(pretrained=False, last_stride=1, model_path=''):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 8, 36, 3], model_path=model_path, model_name='resnet152')


def resnext101_32x8d(pretrained=False, last_stride=1, model_path='', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(pretrained=pretrained, last_stride=last_stride,
                   block=Bottleneck, layers=[3, 4, 23, 3],
                   model_path=model_path, model_name='resnext101_32x8d', **kwargs)