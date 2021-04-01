# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Logger(object):
    def __init__(self, fpath=None, rank='0'):
        self.console = sys.stdout
        print()
        self.file = None
        self.rank = rank
        if fpath is not None:
            if not os.path.exists(fpath):
                os.makedirs(fpath, exist_ok=True)
            self.file = open(os.path.join(fpath, 'log_%s.txt' % rank), 'a+')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.rank == '0':
            print(msg)
        if self.file is not None:
            print(msg, file=self.file)
        self.flush()

    def __call__(self, msg):
        self.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
