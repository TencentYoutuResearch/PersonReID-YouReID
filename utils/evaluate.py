import sys
import os
import numpy as np
sys.path.append("..")
sys.path.append("../config/pycharm-debug-py3k.egg")
import utils.measure
from scipy import io
import utils.metric
import json


def eval_result(data, root,
                use_rerank=False, use_pcb_format=True, logger=None):

    gallery = os.path.join(root, data + '_gallery.mat')
    query = os.path.join(root, data + '_query.mat')
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
            query_cam.append(int(cam.split('c')[-1]))
        else:
            query_label.append(int(q.split('.')[0].split('_')[0]))
            query_cam.append(0)
 
    gallery_label = []
    gallery_cam = []
    for g in gallery['path']:
        if use_pcb_format:
            pid, cam, _ = g.split('_')
            gallery_label.append(int(pid))
            gallery_cam.append(int(cam.split('c')[-1]))
        else:
            gallery_label.append(int(g.split('.')[0].split('_')[0]))
            gallery_cam.append(1)

    cmc_scores, mAP  = utils.measure.evaluate_rank(dist,
                                np.array(query_label), np.array(gallery_label),
                                np.array(query_cam), np.array(gallery_cam),
                                max_rank=50)
    if logger is None:
        print('Mean AP: {:4.2%}'.format(mAP))
    else:
        logger.write('Mean AP: {:4.2%}'.format(mAP))

    if 0 <= mAP <= 100:
        if use_pcb_format:
            cmc_topk = (1, 5, 10, 15, 20)
        else:
            cmc_topk = (1, 3, 5)
        for k in cmc_topk:
            if logger is None:
                logger.write('  top-{:<4}{:12.2%}'.format(k, cmc_scores[k - 1]))
            else:
                logger.write('  top-{:<4}{:12.2%}'.format(k, cmc_scores[k - 1]))

    return mAP, cmc_scores[0]

def write_json(path, signature, data):
    with open(os.path.join(path, signature + 'result.json'), 'w') as f:
        json.dump(data, f)
