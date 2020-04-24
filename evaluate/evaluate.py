import sys
import os
import numpy as np
sys.path.append("..")
sys.path.append("../config/pycharm-debug-py3k.egg")
import utils.measure
from  scipy import io
import utils.metric
from collections import defaultdict
import json
import numpy
import time

def eval_result(data, root, use_metric_cuhk03=False,
                use_rerank=False, use_pcb_format=True):

    gallery = os.path.join(root, data +'_gallery.mat')
    query = os.path.join(root, data +'_query.mat')
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)

    dist = utils.metric.cosine(query['feature'], gallery['feature'])
    if use_rerank:
        qq_dis = utils.metric.cosine(query['feature'], query['feature'])
        gg_dis = utils.metric.cosine(gallery['feature'], gallery['feature'])

        dist = utils.re_ranking(dist, qq_dis, gg_dis)

    query_label = []
    query_cam = []
    for q in query['path']:
        if use_pcb_format:
            pid, cam, _ = q.split('_')
            query_label.append(int(pid))
            query_cam.append(int(cam))
        else:
            query_label.append(int(q.split('.')[0].split('_')[0]))
            query_cam.append(0)
 
    gallery_label = []
    gallery_cam = []
    for g in gallery['path']:
        if use_pcb_format:
            pid, cam, _ = g.split('_')
            gallery_label.append(int(pid))
            gallery_cam.append(int(cam))
        else:
            gallery_label.append(int(g.split('.')[0].split('_')[0]))
            gallery_cam.append(1)

    cmc_scores, mAP  = utils.measure.evaluate_rank(dist, np.array(query_label), np.array(gallery_label),
                                np.array(query_cam), np.array(gallery_cam), max_rank=50, use_metric_cuhk03=use_metric_cuhk03)
    print('Mean AP: {:4.2%}'.format(mAP))


    if mAP>=0 and mAP<=100:
        if use_pcb_format:
            cmc_topk = (1, 5, 10, 15, 20)
        else:
            cmc_topk = (1, 3, 5)
        for k in cmc_topk:
            print('  top-{:<4}{:12.2%}'
                  .format(k,cmc_scores[k - 1]))



def part_re_ranking(code, data, path='../../result',
                    signature='',
                    top_k=2000,
                    k1=20, k2=2, lambda_value=1):

    gallery = '../../feature/' + code + '/' + data + '_gallery.mat'
    query = '../../feature/' + code + '/' + data + '_query.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)
    result = {}
    q_feat = query['feature']
    g_feat = gallery['feature']
    q_paths = query['path']
    g_paths = gallery['path']

    q_paths = np.array([q.replace(' ', '') for q in list(q_paths)])
    g_paths = np.array([g.replace(' ', '') for g in list(g_paths)])
    q_g_dist = utils.metric.cosine(q_feat, g_feat)
    t0 = time.time()
    dist_top_k_indices = np.argpartition(q_g_dist, top_k, axis=-1)[:, :top_k]
    print('dist', time.time()-t0)
    for i in range(q_g_dist.shape[0]):
        t1 = time.time()
        interest_q_g_dist = q_g_dist[i:i+1, dist_top_k_indices[i]] # 1, top_k
        print(np.sort(interest_q_g_dist))
        interest_q_q_dist = [[0]] # 1,1
        interest_g_g_dist = utils.metric.cosine(g_feat[dist_top_k_indices[i], :],
                                   g_feat[dist_top_k_indices[i], :])
        interest_q_path = q_paths[i]
        interest_g_paths = g_paths[dist_top_k_indices[i]]
        final_dist = utils.re_ranking(interest_q_g_dist,
                                interest_q_q_dist,
                                interest_g_g_dist,
                                k1=k1, k2=k2,
                                lambda_value=lambda_value)    # 1, top_k
        print(np.sort(final_dist))
        top_200_indices = np.argsort(final_dist, axis=-1)[0, :200]
        interest_g_paths = interest_g_paths[top_200_indices]
        result[interest_q_path] =list(interest_g_paths)
        print(i, time.time() - t1)
        if i >=5 :
            break

    path = os.path.join(path, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    write_json(path, code + signature + '_part_rerank', result)

def part_re_ranking_group(code, data, path='../../result',
                    signature='',
                    top_k=300, group_size=50,
                    k1=30, k2=2, lambda_value=0.5):

    gallery = '../../feature/' + code + '/' + data + '_gallery.mat'
    query = '../../feature/' + code + '/' + data + '_query.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)
    result = {}
    q_feat = query['feature']
    g_feat = gallery['feature']
    q_paths = query['path']
    g_paths = gallery['path']

    q_paths = np.array([q.replace(' ', '') for q in list(q_paths)])
    g_paths = np.array([g.replace(' ', '') for g in list(g_paths)])
    t0 = time.time()
    q_g_dist = utils.metric.cosine(q_feat, g_feat)
    q_q_dist = utils.metric.cosine(q_feat, q_feat)
    dist_top_k_indices = np.argpartition(q_g_dist, top_k, axis=-1)[:, :top_k]
    
    print('dist', time.time()-t0)
    for i in range(q_g_dist.shape[0] // group_size):
        t1 = time.time()
        part_g_indices = np.unique(dist_top_k_indices[i * group_size: (i + 1) * group_size])
        interest_q_g_dist = q_g_dist[i * group_size: (i+1) * group_size, part_g_indices] # 1, top_k
        interest_q_q_dist = q_q_dist[i * group_size: (i+1) * group_size, i * group_size: (i+1) * group_size]
        interest_g_g_dist = utils.metric.cosine(g_feat[part_g_indices, :], g_feat[part_g_indices, :])
        print(i, time.time() - t1, interest_g_g_dist.shape) 
        interest_q_path = q_paths[i * group_size: (i+1) * group_size]
        interest_g_paths = g_paths[part_g_indices]
        final_dist = utils.re_ranking(interest_q_g_dist,
                                interest_q_q_dist,
                                interest_g_g_dist,
                                k1=k1, k2=k2,
                                lambda_value=lambda_value)    # 1, top_k
        top_200_indices = np.argsort(final_dist, axis=-1)[:, :200]
        for j in range(interest_q_path.shape[0]):
            result[interest_q_path[j]] =list(interest_g_paths[top_200_indices[j]])
        
        print(i, time.time() - t1)
        if i >= 4:
            break

    path = os.path.join(path, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    write_json(path, code + signature + '_part_rerank_test', result)

def eval_kesci(code, data, path='../../result', rerank=False, signature=''):

    gallery = '../../feature/' +code + '/'+ data +'_gallery.mat'
    query = '../../feature/'+ code + '/'+ data+'_query.mat'
    train = '../../feature/'+ code + '/'+ data+'_val.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)
    train = io.loadmat(train)

    dist = utils.metric.cosine(query['feature'], gallery['feature'])
    dist_train_query = utils.metric.cosine(query['feature'], train['feature'])
    q_g_min = dist.min(axis=-1)
    q_t_min = dist_train_query.min(axis=-1)
    print( len(q_t_min), len(q_g_min) )
    print( q_g_min.sum( ) / q_t_min.sum( ) )
    print( np.exp(q_g_min).sum( ) / np.exp( q_t_min ).sum() )
    print( 'q_t dist smaller than q_g dist', (q_t_min < q_g_min ).sum( ) )   
    print(q_g_min) 
    result  = get_kesci_result(dist, query['path'], gallery['path'], max_rank=200, with_dis=True)

    path = os.path.join(path, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    write_json(path, code+signature+'_dist', result)

    #dist_qq = utils.metric.cosine(query['feature'], query['feature'])
    #result_qq  = get_kesci_result(dist_qq, query['path'], query['path'], max_rank=10) 
    #write_json(path, code+signature+'_qq', result_qq)

    #valid_ind = np.argpartition(dist, 100, axis=1)[:, :100]
    #valid_ind = np.sort(np.unique(np.reshape(valid_ind, (-1))))
    #print(valid_ind.shape)
    #print(valid_ind)
    #valid_dist = dist[:, valid_ind]
 
    if rerank:
        qq_dis = utils.metric.cosine(query['feature'], query['feature'])
        gg_dis = utils.metric.cosine(gallery['feature'], gallery['feature'])

        dist = utils.re_ranking(dist, qq_dis, gg_dis, k1=30, k2=2, lambda_value=0.3)
    
        result  = get_kesci_result(dist, query['path'], gallery['path'], max_rank=200)
        write_json(path, code+signature+'rerank30-2-0.3-b', result)


def eval_kesci_multi_crop(code, data, path='../../result', signature=''):
    for t in range(5):
        gallery = '../../feature/' +code + '/'+ data +'_gallery' + '_'+ str(t) + '.mat'
        query = '../../feature/'+ code + '/'+ data+'_query' + '_'+ str(t) + '.mat'
        gallery = io.loadmat(gallery)
        query = io.loadmat(query)
        if t == 0:
            dist = utils.metric.cosine(query['feature'], gallery['feature'])
        else:
            dist += utils.metric.cosine(query['feature'], gallery['feature'])
    # qq_dis = utils.metric.cosine(query['feature'], query['feature'])
    # gg_dis = utils.metric.cosine(gallery['feature'], gallery['feature'])
    #
    # dist = utils.re_ranking(dist, qq_dis, gg_dis)

    result  = get_kesci_result(dist, query['path'], gallery['path'], max_rank=200)

    path = os.path.join(path, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    write_json(path, signature, result)




def get_kesci_result(distmat, q_paths, g_paths, max_rank=200, with_dis=False):
    result = {}
    num_q, num_g = distmat.shape

    q_paths = numpy.array([q.replace(' ','') for q in list(q_paths)])
    g_paths = numpy.array([g.replace(' ', '') for g in list(g_paths)])

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):

        q_path = q_paths[q_idx]
        order = indices[q_idx, 0:200]
        sorted_list = list(g_paths[order])
        if with_dis:
            dis_qg = distmat[q_idx, order]
            result[q_path]  = (sorted_list, list(map(float, list(dis_qg))))
        else:
            result[q_path]  = sorted_list


    return result

def write_json(path, signature, data):
    with open(os.path.join(path, signature + 'result.json'), 'w') as f:
        json.dump(data, f)
