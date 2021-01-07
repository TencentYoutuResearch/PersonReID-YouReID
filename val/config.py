import yaml
import os

class Config(object):
    """"""
    def __init__(self):
        if os.path.isfile('eval_config.yaml'):
            with open('eval_config.yaml') as f:
                config = f.read()
                self._config = yaml.load(config)
        else:
            self._config = {}
    def get(self, attr):
        return self._config.get(attr, None)


config = Config()