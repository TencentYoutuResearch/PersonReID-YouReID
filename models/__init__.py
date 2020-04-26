from models.legacy.resnet import *
from models.legacy.pcb import *
from models.legacy.resnet_enhanced import resnet50e, resnet152e
from .mgn import MGN
from models.legacy.mgn_v2 import MGNv2
from .baseline import Baseline
# from models.legacy.baseline_arcface import BA

__all__ = ['PCB', 'resnet50e', 'resnet152e', 'MGN', 'MGNv2', 'Baseline']
