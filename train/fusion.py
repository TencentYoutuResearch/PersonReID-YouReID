import os
import torch
from scipy import io as io

import sys
import numpy
import json
sys.path.append("..")
# from utils.metric import cosine


def cosine(query, gallery):
    query = torch.from_numpy(query)
    gallery = torch.from_numpy(gallery)

    m, n = query.size(0), gallery.size(0)
    dist = 1 - torch.mm(query, gallery.t())/((torch.norm(query, 2, dim=1, keepdim=True).expand(m, n)
                                              *torch.norm(gallery, 2, dim=1, keepdim=True).expand(n, m).t()))
    return dist.numpy()

def fusing(code, code2, data, path='../../result'):

    gallery = '../../feature/' +code + '/'+ data +'_gallery.mat'
    query = '../../feature/'+ code + '/'+ data+'_query.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)

    dist = cosine(query['feature'], gallery['feature'])


    gallery = '../../feature/' +code2 + '/'+ data +'_gallery.mat'
    query = '../../feature/'+ code2 + '/'+ data+'_query.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)

    dist = cosine(query['feature'], gallery['feature']) + dist
    # qq_dis = utils.metric.cosine(query['feature'], query['feature'])
    # gg_dis = utils.metric.cosine(gallery['feature'], gallery['feature'])
    #
    # dist = utils.re_ranking(dist, qq_dis, gg_dis)

    result  = get_kesci_result(dist, query['path'], gallery['path'], max_rank=200)

    path = os.path.join(path, str(code+code2))
    if not os.path.exists(path):
        os.makedirs(path)
    write_json(path, result)




def get_kesci_result(distmat, q_paths, g_paths, max_rank=200):
    result = {}
    num_q, num_g = distmat.shape

    q_paths = numpy.array([q.replace(' ','') for q in list(q_paths)])
    g_paths = numpy.array([g.replace(' ', '') for g in list(g_paths)])

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = numpy.argsort(distmat, axis=1)

    for q_idx in range(num_q):

        q_path = q_paths[q_idx]
        order = indices[q_idx, 0:200]
        sorted_list = list(g_paths[order])
        result[q_path]  = sorted_list


    return result

def write_json(path, data):
    with open(os.path.join(path, 'fusion_result.json'), 'w') as f:
        json.dump(data, f)


def main1():
    fusing(code='2019111102', code2='2019111203', data='KESCI')

if __name__ == '__main__':
    main1()
