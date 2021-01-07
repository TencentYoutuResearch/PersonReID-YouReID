import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .resnet import Bottleneck


__all__ = ['ResNet', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50_ibn_a': os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet50_ibn_a.pth.tar'),
    'resnet101_ibn_a': os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet101_ibn_a.pth.tar')
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, last_stride=2, for_test=True, **kwargs):
        scale = 64
        self.inplanes = scale
        self.for_test = for_test
        super(ResNet, self).__init__()
        self.use_non_local = kwargs.get('use_non_local', False)

        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2, use_non_local=self.use_non_local)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=last_stride, use_non_local=self.use_non_local)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.InstanceNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_non_local=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_non_local_flag = use_non_local and i == blocks - 2
            layers.append(block(self.inplanes, planes, ibn, use_non_local = use_non_local_flag))

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
        if not self.for_test:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def resnet50_ibn_a(pretrained=False, last_stride=1, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], last_stride=last_stride, **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['resnet50_ibn_a'])
        new_state_dict = OrderedDict()
        for k in state_dict['state_dict']:
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = state_dict['state_dict'][k]
        model.load_state_dict(new_state_dict, strict=False)
    return model


def resnet101_ibn_a(pretrained=False, last_stride=1, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], last_stride=last_stride, **kwargs)
    if pretrained:
        if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK']:
            print('map weight to cuda: %s' % str(os.environ['LOCAL_RANK']))
            state_dict = torch.load(model_urls['resnet101_ibn_a'],
                                    map_location="cuda:" + str(os.environ['LOCAL_RANK']))
        else:
            state_dict = torch.load(model_urls['resnet101_ibn_a'])
        new_state_dict = OrderedDict()
        for k in state_dict['state_dict']:
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = state_dict['state_dict'][k]
        model.load_state_dict(new_state_dict, strict=False)
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
