from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
from PIL import Image
import torch
import numpy as np



def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, root='../../snapshot/', flag='checkpoint.pth', logger=None):

    if not os.path.exists(root):
        os.makedirs(root)
    filename = os.path.join(root, flag)
    torch.save(state, filename)
    if logger:
        logger.write('Save checkpoint at %s' % filename)
    else:
        print('Save checkpoint at %s' % filename)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.MAT'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_image(path, mask_p=None):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not os.path.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    if not is_image_file(path):
        raise IOError('"{}" is not an image file'.format(path))
    while not got_img:
        try:
            # if path.endswith('jpg'):
            img = Image.open(path).convert('RGB')
            # else:
            #     img = cv2.imread(path)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
            pass

    if mask_p is None:
        return img
    else:
        mask = np.load(mask_p)
        return img, mask
