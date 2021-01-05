#conding=utf-8
# @Time  : 2020/3/30 21:31
# @Author: fufuyu
# @Email:  fufuyu@tencent.com


import yaml
import os
import argparse

class Config(object):
    def __init__(self, args):
        if os.path.isfile(args.yaml):
            with open(args.yaml) as f:
                config = f.read()
                self._config = yaml.load(config)
            with open(self._config['yaml']) as f:
                super_config = f.read()
                self._super_config = yaml.load(super_config)
                self._config.update(self._super_config)
        else:
            self._config = {}
        self._default_config = self.default_config()

    def default_config(self):
        return {
            'seed': 1234,
            'print_freq': 10
        }

    def get(self, attr, default=None):
        default = self._default_config.get(attr, default)
        return self._config.get(attr, default)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--yaml', default='../config.yaml',
                    help='yaml path')
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
config = Config(args)