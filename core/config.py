import yaml
import os

class Config(object):
    def __init__(self):
        if os.path.isfile('config.yaml'):
            with open('config.yaml') as f:
                config = f.read()
                self._config = yaml.load(config)
            with open(self._config['yaml']) as f:
                self._super_config = yaml.load(f.read())
            self._config.update(self._super_config)
        else:
            self._config = {}
        self._default_config = self.default_config()

    def default_config(self):
        return {
            'root_path': '/data1/home/fufuyu/checkpoint/',
            'save_max_to_keep': 2,
            'log_every_n_steps': 10,
            'save_every_n_steps': 1000,
        }

    def get(self, attr):
        default = self._default_config.get(attr, None)
        return self._config.get(attr, default)


config = Config()