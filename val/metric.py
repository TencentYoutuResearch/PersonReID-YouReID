#conding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import average_precision_score
import pickle
import os.path as osp
import sys
ospj = osp.join

import os


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

def normcc(fff):
    fff = np.array(fff).astype("float32")
    fff /= (fff**2).sum()**0.5
    return fff


def load_feat(feat_path):
    feats = []
    ids = []
    cams = []
    paths = []
    with open(feat_path) as f:
        for line in f.readlines():
            line = line.strip()
            path = line.split(" ")[0]
            strfeature = line.split(" ")[1:]
            feat = normcc(strfeature)
            id = parse_im_name(path,'id')
            cam = parse_im_name(path,'cam')

            feats.append(feat)
            ids.append(id)
            cams.append(cam)
            paths.append(path)
    #print (feats)
    feats = np.stack(feats, axis=0)
    ids = np.stack(ids,axis=0)
    cams = np.stack(cams, axis=0)

    return feats, ids, cams, paths


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean', 'weight']
  if type == 'cosine':
    #array1 = normalize(array1, axis=1)
    #array2 = normalize(array2, axis=1)
    #dist = np.matmul(array1, array2.T)

    #boyyang cosine dist
    print('cosine dist')
    dist = 1 - np.matmul(array1, array2.T)
    dist_cos = dist - np.min(dist)
    return dist_cos

  elif type == 'euclidean':
    print('euclidean dist')
    # shape [m1, 1] (after normalize still square1??)
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    #print(square1)
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    #print(square2)
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist
  elif type == 'weight':
    print('weight dist')
    a1 = array1[:, : 1536].reshape((-1, 1, 6, 256))
    a2 = array2[:, : 1536].reshape((1, -1, 6, 256))
    s1 = array1[:, 1536: ].reshape((-1, 1, 6))
    s2 = array2[:, 1536: ].reshape((1, -1, 6))
    s1_s2 = s1 * s2
    dists = []
    begin, end = 0, 0
    nums = 20
    gap = int(a1.shape[0] / nums)
    for i in range(nums):
        end = (i + 1) * gap
        a_i = a1[begin:end, :]
        begin = end
        dist_i = np.sqrt(np.sum(np.square(a_i - a2), axis=-1) + 1e-6)
        dists.append(dist_i)
    a_i = a1[begin:, :]
    dist_i = np.sqrt(np.sum(np.square(a_i - a2), axis=-1) + 1e-6)
    dists.append(dist_i)
    dist = np.concatenate(dists, axis=0)
    # stop = int(a1.shape[0] / 2)
    # dist = np.sqrt(np.sum(np.square(a1-a2), axis=-1) + 1e-6)
    # dist2 = np.sqrt(np.sum(np.square(a1[stop:, :] - a2), axis=-1) + 1e-6)

    dist = np.sum(dist * s1_s2, axis=-1) / np.sum(s1_s2, axis=-1)
    return dist


def cmc(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    topk=100,
    separate_camera_set=False,
    first_match_break=False,
    filter_same_id_and_cam=True,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape
  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1)
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute CMC for each query
  ret = np.zeros([m, topk])
  is_valid_query = np.zeros(m)
  num_valid_queries = 0
  for i in range(m):
    # Filter out the same id and same camera
    if filter_same_id_and_cam:
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
    else:
        valid = np.ones(n).astype(np.bool)
    if separate_camera_set:
      # Filter out samples from same camera
      valid &= (gallery_cams[indices[i]] != query_cams[i])
    if not np.any(matches[i, valid]): continue
    is_valid_query[i] = 1
    repeat = 1
    for _ in range(repeat):
      index = np.nonzero(matches[i, valid])[0]
      delta = 1. / (len(index) * repeat)
      for j, k in enumerate(index):
        if k - j >= topk: break
        if first_match_break:
          ret[i, k - j] += 1
          break
        ret[i, k - j] += delta
    num_valid_queries += 1
  if num_valid_queries == 0:
    raise RuntimeError("No valid query")
  ret = ret.cumsum(axis=1)
  if average:
    return np.sum(ret, axis=0) / num_valid_queries, ret, is_valid_query
  return ret, is_valid_query


def mean_ap(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    filter_same_id_and_cam=True,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """

  # -------------------------------------------------------------------------
  # The behavior of method `sklearn.average_precision` has changed since version
  # 0.19.
  # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
  # (https://github.com/zhunzhong07/person-re-ranking/
  # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
  # (http://www.liangzheng.org/Project/project_reid.html).
  # My current awkward solution is sticking to this older version.
  import sklearn
  cur_version = sklearn.__version__
  required_version = '0.18.1'
  if cur_version != required_version:
    print('User Warning: Version {} is required for package scikit-learn, '
          'your current version is {}. '
          'As a result, the mAP score may not be totally correct. '
          'You can try `pip uninstall scikit-learn` '
          'and then `pip install scikit-learn=={}`'.format(
      required_version, cur_version, required_version))
  # -------------------------------------------------------------------------

  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape #5005,69000
  # TODO find valid query
  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1) #TODO small->big index a query sim
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute AP for each query
  aps = np.zeros(m)
  is_valid_query = np.zeros(m)
  for i in range(m):
    # Filter out the same id and same camera
    if filter_same_id_and_cam:
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    else:
        valid = np.ones(n).astype(np.bool)
    y_true = matches[i, valid]
    y_score = -distmat[i][indices[i]][valid]
    if not np.any(y_true): continue
    is_valid_query[i] = 1
    aps[i] = average_precision_score(y_true, y_score) #average
  if len(aps) == 0:
    raise RuntimeError("No valid query")
  if average:
    return float(np.sum(aps)) / np.sum(is_valid_query), None
  return aps, is_valid_query


def print_scores(mAP, cmc_scores):
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
            .format(mAP, *cmc_scores[[0, 4, 9]]))


def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    #parsed = int(im_name[:8])
    parsed = int(os.path.basename(im_name).split('_')[0])
  else:
    #parsed = int(im_name[9:13])
    parsed = int(os.path.basename(im_name).split('_')[1])
  return parsed


