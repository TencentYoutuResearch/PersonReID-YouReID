from models.legacy.resnet import *
# from models.legacy.pcb import *
# from models.legacy.resnet_enhanced import resnet50e, resnet152e
from .mgn import MGN
from .baseline import Baseline
from .baseline_split import BaselineSplit
from .baseline_shuffle import BaselineShuffle
from .pcb import PCB
from .mpcb import MPCB
from .pgfa import PGFA

# from models.legacy.baseline_arcface import BA

__all__ = ['PCB', 'MPCB', 'MGN', 'Baseline', 'PGFA', 'BaselineSplit', 'BaselineShuffle']
