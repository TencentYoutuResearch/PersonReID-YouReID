import numpy as np
from PIL import Image
import random
import numbers
import math
from torchvision.transforms.functional import pad



def rotate_img(img, rot):
    if isinstance(img, Image.Image):
        img = np.array(img)

    if rot == 0: # 0 degrees rotation
        return Image.fromarray(img)
    elif rot == 90: # 90 degrees rotation
        return Image.fromarray(np.flipud(np.transpose(img, (1,0,2))))
    elif rot == 180: # 90 degrees rotation
        return Image.fromarray(np.fliplr(np.flipud(img)))
    elif rot == 270: # 270 degrees rotation / or -90
        return Image.fromarray(np.transpose(np.flipud(img), (1,0,2)))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')



class RandomCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, range=0.7, interpolation=Image.BILINEAR):
        self.range =  range

        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(100):
            area = img.size[0] * img.size[1]
            if isinstance(self.range, tuple):
                l, h = self.range
                ratio = random.uniform(l, h)
            else:
                assert isinstance(self.range, numbers.Number)
                ratio = self.range
            w = int(random.uniform(round(ratio*img.size[0]), img.size[0]))
            h = int(ratio* area/w)


            if w <= img.size[0] and h <= img.size[1]:
                left = random.randint(0, img.size[0] - w)
                upper = random.randint(0, img.size[1] - h)

                img = img.crop((left, upper, left + w, upper + h))
                assert (img.size == (w, h))

                return img

        # Fallback


        return img


class CameraMixup(object):

    def __init__(self, min_alpha=0.7, random_mix=False):
        self.min_alpha = min_alpha
        self.random_mix = random_mix

    def __call__(self, tensor, tensor_of_other_camera):

        return camera_mixup(tensor, tensor_of_other_camera, self.min_alpha, self.random_mix)


def camera_mixup(tensor, tensor_of_other_camera, min_alpha=0.7, random_mix=False):
    if random_mix:
        alpha = random.uniform(min_alpha, 1)
    else:
        alpha = min_alpha
    if random.uniform(0, 1) > 0.7:
        return alpha * tensor + (1 - alpha) * tensor_of_other_camera
    else:
        return tensor


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomPadding(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, max_pad_rate=0.4):
        self.probability = probability
        self.max_pad_rate = max_pad_rate

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        h, w = img.height, img.width
        pad_left = random.randint(0, int(w * self.max_pad_rate / 2))
        pad_right = random.randint(0, int(w * self.max_pad_rate / 2))
        pad_top = random.randint(0, int(h * self.max_pad_rate / 2))
        pad_bottom = random.randint(0, int(h * self.max_pad_rate / 2))
        paddings = (pad_left, pad_top, pad_right, pad_bottom)
        img = pad(img, paddings, fill=127)

        return img



class PositionCrop(object):

    def __init__(self, position):
        self.position = position

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
            positiono: The crop rectangle, as a (left, upper, right, lower)-tuple.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        x1, y1, x2, y2 = self.position

        return img.crop((x1, y1, x2, y2))


class RandomCropping(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, max_pad_rate=0.2):
        self.probability = probability
        self.max_pad_rate = max_pad_rate

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        h, w = img.height, img.width
        crop_left = random.randint(0, int(w * self.max_pad_rate / 2))
        crop_right = random.randint(0, int(w * self.max_pad_rate / 2))
        crop_top = random.randint(0, int(h * self.max_pad_rate / 2))
        crop_bottom = random.randint(0, int(h * self.max_pad_rate / 2))

        return img.crop((crop_left, crop_top, w-crop_right, h-crop_bottom))
