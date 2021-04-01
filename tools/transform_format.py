"""Refactor file directories, save/rename images and partition the
train/val/test set, in order to support the unified dataset interface.
borrowed from https://github.com/huanghoujing/person-reid-triplet-loss-baseline
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os
import os.path as osp
import numpy as np
import pickle
import glob
import shutil
from collections import defaultdict

def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)

def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret

def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret

def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set.
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use,
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`,
                      `val_query_im_names`,
                      `val_gallery_im_names`)
  """
  np.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  np.random.shuffle(im_names)
  ids = np.array([parse_im_name(n, 'id') for n in im_names])
  cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(im_names))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  train_inds = np.sort(train_inds)
  query_inds = np.sort(query_inds)
  gallery_inds = np.sort(gallery_inds)

  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions


new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def parse_new_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed

def move_ims(ori_im_paths, new_im_dir, parse_im_name, new_im_name_tmpl):
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    shutil.copy(im_path, osp.join(new_im_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names

def parse_original_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = -1 if im_name.startswith('-1') else int(im_name[:4])
  else:
    parsed = int(im_name[4]) if im_name.startswith('-1') \
      else int(im_name[6])
  return parsed


def save_images(zip_file, save_dir=None, train_test_split_file=None):
  """Rename and move all used images to a directory."""

  print("Extracting zip file")
  root = osp.dirname(osp.abspath(zip_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(osp.abspath(save_dir))
  with ZipFile(zip_file) as z:
    z.extractall(path=save_dir)
  print("Extracting zip file done")

  new_im_dir = osp.join(save_dir, 'images')
  may_make_dir(osp.abspath(new_im_dir))
  raw_dir = osp.join(save_dir, osp.basename(zip_file)[:-4])

  im_paths = []
  nums = []

  im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_train'),
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))

  im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_test'),
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths_ = [p for p in im_paths_ if not osp.basename(p).startswith('-1')]
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))

  im_paths_ = get_im_names(osp.join(raw_dir, 'query'),
                           return_path=True, return_np=False)
  im_paths_.sort()
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))
  q_ids_cams = set([(parse_original_im_name(osp.basename(p), 'id'),
                     parse_original_im_name(osp.basename(p), 'cam'))
                    for p in im_paths_])

  im_paths_ = get_im_names(osp.join(raw_dir, 'gt_bbox'),
                           return_path=True, return_np=False)
  im_paths_.sort()
  # Only gather images for those ids and cams used in testing.
  im_paths_ = [p for p in im_paths_
               if (parse_original_im_name(osp.basename(p), 'id'),
                   parse_original_im_name(osp.basename(p), 'cam'))
               in q_ids_cams]
  im_paths += list(im_paths_)
  nums.append(len(im_paths_))

  im_names = move_ims(
    im_paths, new_im_dir, parse_original_im_name, new_im_name_tmpl)

  split = dict()
  keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names', 'mq_im_names']
  inds = [0] + nums
  inds = np.cumsum(np.array(inds))
  for i, k in enumerate(keys):
    split[k] = im_names[inds[i]:inds[i + 1]]

  save_pickle(split, train_test_split_file)
  print('Saving images done.')
  return split


def transform(zip_file, save_dir=None):
  """Refactor file directories, rename images and partition the train/val/test
  set.
  """

  train_test_split_file = osp.join(save_dir, 'train_test_split.pkl')
  train_test_split = save_images(zip_file, save_dir, train_test_split_file)
  # train_test_split = load_pickle(train_test_split_file)

  # partition train/val/test set

  trainval_ids = list(set([parse_new_im_name(n, 'id')
                           for n in train_test_split['trainval_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  trainval_ids.sort()
  trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
  partitions = partition_train_val_set(
    train_test_split['trainval_im_names'], parse_new_im_name, num_val_ids=100)
  train_im_names = partitions['train_im_names']
  train_ids = list(set([parse_new_im_name(n, 'id')
                        for n in partitions['train_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  train_ids.sort()
  train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

  # A mark is used to denote whether the image is from
  #   query (mark == 0), or
  #   gallery (mark == 1), or
  #   multi query (mark == 2) set

  val_marks = [0, ] * len(partitions['val_query_im_names']) \
              + [1, ] * len(partitions['val_gallery_im_names'])
  val_im_names = list(partitions['val_query_im_names']) \
                 + list(partitions['val_gallery_im_names'])

  test_im_names = list(train_test_split['q_im_names']) \
                  + list(train_test_split['mq_im_names']) \
                  + list(train_test_split['gallery_im_names'])
  test_marks = [0, ] * len(train_test_split['q_im_names']) \
               + [2, ] * len(train_test_split['mq_im_names']) \
               + [1, ] * len(train_test_split['gallery_im_names'])

  partitions = {'trainval_im_names': train_test_split['trainval_im_names'],
                'trainval_ids2labels': trainval_ids2labels,
                'train_im_names': train_im_names,
                'train_ids2labels': train_ids2labels,
                'val_im_names': val_im_names,
                'val_marks': val_marks,
                'test_im_names': test_im_names,
                'test_marks': test_marks}
  partition_file = osp.join(save_dir, 'partitions.pkl')
  save_pickle(partitions, partition_file)
  print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform Market1501 Dataset")
  parser.add_argument('--zip_file', type=str,
                      default='~/Dataset/market1501/Market-1501-v15.09.15.zip')
  parser.add_argument('--save_dir', type=str,
                      default='~/Dataset/market1501')
  args = parser.parse_args()
  zip_file = osp.abspath(osp.expanduser(args.zip_file))
  save_dir = osp.abspath(osp.expanduser(args.save_dir))
  transform(zip_file, save_dir)