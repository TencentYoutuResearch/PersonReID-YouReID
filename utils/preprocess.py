#conding=utf-8

import tensorflow as tf
from core.preprocessim import PreProcessIm
from core.config import config

class SampleProcessor:
    def __init__(self):
        self._resize_size = 256
        self._crop_size = 248

    def proc(self, image_buffer, is_aug=True):
        im = tf.image.decode_jpeg(image_buffer, channels=3, dct_method='INTEGER_ACCURATE')
        if config.get('preprocess_config'):
            pre_process_im = PreProcessIm(**config.get('preprocess_config'))
        else:
            pre_process_im = PreProcessIm()

        image=pre_process_im(im)

        return image

class SingleSampleProcessor(SampleProcessor):
    def __call__(self, file_name, label_idx, is_aug=True):
        image_buffer = tf.read_file(file_name)
        image = self.proc(image_buffer, is_aug=is_aug)
        return (file_name, image, {'label': label_idx})
