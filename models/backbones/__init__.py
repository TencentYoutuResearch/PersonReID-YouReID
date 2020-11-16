from .resnet import resnet50, resnet101, resnext101_32x8d
from .resnest import resnest50, resnest101, resnest200, resnest269
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .resnet_bcn import resnet50_bcn, resnet101_bcn

model_zoo = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnext101_32x8d': resnext101_32x8d,
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet50_bcn': resnet50_bcn,
    'resnet101_bcn': resnet101_bcn
}