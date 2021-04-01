from .resnet import resnet50, resnet101, resnext101_32x8d
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .senet import se_resnext101_32x4d

model_zoo = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'se_resnext101_32x4d': se_resnext101_32x4d
}
