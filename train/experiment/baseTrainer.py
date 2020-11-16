# conding=utf-8
# @Time  : 2020/8/21 10:20
# @Author: fufuyu
# @Email:  fufuyu@tencent.com


import os
import torch
import torch.distributed as dist
from utils.logger import create_logger
import numpy as np

class BaseTrainer(object):
    def __init__(self):
        """"""
        # self.local_rank = local_rank

    def init_distirbuted_mode(self):
        """"""
        env_dict = {
            key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        self.logger.info(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl", world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    def fix_random_seed(self, seed=31):
        """"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def init_logger(self):
        """"""
        self.logger = create_logger(filepath=os.environ['RPF_LOG_DIR'], rank=os.environ['RANK'])
        self.logger.info('============ Initialized logger ============')
        # logger.info(
        #     "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
        # )
        # logger.info(info"The experiment will be stored in %s\n" % params.dump_path)
        self.logger.info("")

    def run(self):
        self.init_logger()
        self.init_distirbuted_mode()
        self.fix_random_seed()
        self.train()
        dist.destroy_process_group()

    def make_dataset(self):
        """"""
        raise NotImplementedError()

    def make_model(self):
        raise NotImplementedError()

    def make_optimizer(self, model):
        raise NotImplementedError()

    def restore_from_ckpt(self, ckp_path, run_variables=None, **kwargs):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
