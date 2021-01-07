import tensorflow as tf
import numpy as np
import random
import math

def do_random(prob, func, image):
    do_a_random = tf.greater(prob, tf.random_uniform([]))
    image = tf.cond(do_a_random, lambda: func(image), lambda: image)
    return image

def random_horizontal_flip(image, seed=None):
    """Randomly flips the image and detections horizontally.

    The probability of flipping the image is 50%.

    Args:
      image: rank 3 float32 tensor with shape [height, width, channels].
      seed: random seed

    Returns:
      image: image which is the same shape as input image.
    Raises:
      ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image]):
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        return image



class PreProcessIm(object):
    def __init__(
        self,
        im_mean=[0.486, 0.459, 0.408],
        im_std=[0.229, 0.224, 0.225],
        random_erasing_prob=0.5,
        do_pad_and_crop=False,
        sl=0.02,
        sh=0.4,
        r1=0.33,
        height=384,
        width=128
    ):
        self.im_mean = im_mean
        self.im_std = im_std
        self.prng = np.random
        self.im_mean = im_mean
        self.im_std = im_std
        self.random_erasing_prob = random_erasing_prob
        self.do_pad_and_crop = do_pad_and_crop
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.height = height
        self.width = width

    def __call__(self, im):
        return self.pre_process_im(im)


    def generate_box_size_randomly(self, seed=None):
        # erase
        area = self.height * self.width #384*128

        h_f = tf.constant(0)
        w_f = tf.constant(0)
        for attempt in range(10):
            target_area = tf.random_uniform([], minval=self.sl, maxval=self.sh, seed=seed) * area
            aspect_ratio = tf.random_uniform([], minval=self.r1, maxval=1 / self.r1, seed=seed)

            # [h,w] is box size
            h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)  # h143
            w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)  # w69

            get_h_w = tf.logical_and(tf.less(h, self.height), tf.less(w, self.width))

            h = tf.cond(get_h_w, lambda:h, lambda:0)
            w = tf.cond(get_h_w, lambda:w, lambda:0)

            has_h_w = tf.logical_and(tf.greater(h_f, 0), tf.greater(w_f, 0))
            h_f = tf.cond(has_h_w, lambda:h_f, lambda:h)
            w_f = tf.cond(has_h_w, lambda:w_f, lambda:w)

        h1_t = tf.random_uniform([], minval=0, maxval=self.height - h_f, dtype=tf.int32, seed=seed)  # 168
        w1_t = tf.random_uniform([], minval=0, maxval=self.width - w_f, dtype=tf.int32, seed=seed)  # 44

        has_h_w = tf.logical_and(tf.greater(h_f, 0), tf.greater(w_f, 0))
        h1 = tf.cond(has_h_w, lambda:h1_t, lambda:0) 
        w1 = tf.cond(has_h_w, lambda:w1_t, lambda:0) 

        return h_f, w_f, h1, w1

    def erasing(self, image):
        h, w, h1, w1 = self. generate_box_size_randomly()
  
        def erase_image(image, h, w, h1, w1):
            # 1.erase
            black_box = tf.ones([h, w, 3], dtype=tf.float32)
            mask = 1.0 - tf.image.pad_to_bounding_box(black_box, h1, w1, self.height, 128)
            image = tf.multiply(image, mask)

            # 2.add mean
            image_rgb = []
            for i in range(3):
                mean_0 = tf.fill([h, w, 1], self.im_mean[i])
                mean_0_mask = tf.image.pad_to_bounding_box(mean_0, h1, w1, self.height, 128)
                image_rgb.append(tf.expand_dims(tf.add(image[:, :, i], mean_0_mask[:, :, 0]), 2))

            image = tf.concat(image_rgb, axis=2)
            return image

        has_h_w = tf.logical_and(tf.greater(h, 0), tf.greater(w, 0))

        image = tf.cond(has_h_w, lambda: erase_image(image, h, w, h1, w1), lambda: image)

        return image


    def pad_and_crop(self, image):
        image = tf.pad(image, paddings=[[10, 10], [10, 10], [0, 0]])
        image = tf.random_crop(image, size=[self.height, self.width, 3])
        return image

    def pre_process_im(self, image):
        image = tf.image.resize_images(image, [self.height, self.width])

        if self.do_pad_and_crop:
            image = self.pad_and_crop(image)

        image = tf.divide(image, 255)
        if self.random_erasing_prob > 0:
            image = do_random(1-self.random_erasing_prob, self.erasing, image)
        # normalize
        image = tf.subtract(image, self.im_mean)
        image = tf.divide(image, self.im_std)
        # flip
        image = random_horizontal_flip(image)

        return image

