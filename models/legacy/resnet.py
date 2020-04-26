import torch.nn as nn
from torch import load
import torchvision.models.resnet
__all__ = ['ResNet',  'resnet50', 'resnet18']

model_maps = {
    'resnet18': '../../pretrained_models/resnet18-5c106cde.pth',
    'resnet34': '../../pretrained_models/resnet34-333f7ec4.pth',
    'resnet50': '../../pretrained_models/resnet50-19c8e357.pth',
    'resnet101': '../../pretrained_models/resnet101-5d3b4d8f.pth',
    'resnet152': '../../pretrained_models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


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

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, class_num=1000, last_stride=2, is_for_test=False, norm=False):
        self.inplanes = 64
        self.is_for_test = is_for_test
        self.norm = norm
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, is_last=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.is_for_test is False:
            self.fc = nn.Linear(512 * block.expansion, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.is_for_test:
            return x
        if self.norm:
            x = x.renorm(2, 0, 1e-5).mul(1e5)
            class_centroid = self.fc.weight.renorm(2, 0, 1e-5).mul(1e5)
            logit = x.mm(class_centroid.t())
            return logit, x, class_centroid
        else:
            x = self.fc(x)
            return x


    def get_param(self, lr):
        new_param = self.fc.parameters()
        # return new_param
        new_param_id = [id(p) for p in new_param]
        finetuned_params = []
        for p in self.parameters():
            if id(p) not in new_param_id:
                finetuned_params.append(p)
        return [{'params': new_param, 'lr': lr},
                {'params': finetuned_params, 'lr': 1e-1 * lr}]




def resnet50(pretrained=True, remove=False, **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        init_pretrained_weights(model, load(model_maps['resnet50']))

        print("using ImageNet pre-trained model to initialize the weight")
    if remove:
        del model.fc
    return model

def resnet18(pretrained=True, remove=False, **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        init_pretrained_weights(model, load(model_maps['resnet18']))

        print("using ImageNet pre-trained model to initialize the weight")
    if remove:
        del model.fc
    return model

def init_pretrained_weights(model, pretrain_dict):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # for k in pretrain_dict.keys():
    #     print(k)
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)



