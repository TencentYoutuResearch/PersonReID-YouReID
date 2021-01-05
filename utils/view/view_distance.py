import numpy as np
from  scipy import io
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns

def compute_distance(matrix_a, matrix_b, type='cosine'):
    if type == 'cosine':
        return np.matmul(matrix_a, matrix_b.T)

def compute_distance_by_torch(matrix_a, matrix_b, type='cosine'):
    if type == 'cosine':
        if not isinstance(matrix_a, torch.Tensor):
            matrix_a = torch.from_numpy(matrix_a)
        if not isinstance(matrix_b, torch.Tensor):
            matrix_b = torch.from_numpy(matrix_b)
        return torch.matmul(matrix_a, matrix_b.t())

def test(root, data):
    gallery = os.path.join(root, data + '_gallery.mat')
    query = os.path.join(root, data + '_query.mat')
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)

    q_q_dist = compute_distance(query['feature'], query['feature'])
    # q_g_dist = compute_distance(query['feature'], gallery['feature'])
    # g_g_dist = compute_distance(gallery['feature'], gallery['feature'])

    query_label = []
    for q in query['path']:
        pid, cam, _ = q.split('_')
        query_label.append(int(pid))

    # gallery_label = []
    # for g in gallery['path']:
    #     pid, cam, _ = g.split('_')
    #     gallery_label.append(int(pid))

    query_label = np.array(query_label)
    # gallery_label = np.array(gallery_label)


    q_q_mask = np.equal(np.expand_dims(query_label, axis=1), np.expand_dims(query_label, axis=0))
    # q_g_mask = np.equal(np.expand_dims(query_label, axis=1), np.expand_dims(gallery_label, axis=0))
    # g_g_mask = np.equal(np.expand_dims(gallery_label, axis=1), np.expand_dims(gallery_label, axis=0)) #- np.eye(gallery_label.shape[0])

    q_size, g_size = q.shape[0], g.shape[0]
    pos_dist, neg_dist = [], []
    for i in range(q_size):
        for j in range(q_size):
            if i == j:
                continue
            if q_q_mask[i, j]:
                pos_dist.append(q_q_dist[i, j])
            else:
                neg_dist.append(q_q_dist[i, j])

    # for i in range(g_size):
    #     for j in range(g_size):
    #         if i == j:
    #             continue
    #         if g_g_mask[i, j]:
    #             pos_dist.append(g_g_dist[i, j])
    #         else:
    #             neg_dist.append(g_g_dist[i, j])
    #
    # for i in range(q_size):
    #     for j in range(g_size):
    #         if q_g_mask[i, j]:
    #             pos_dist.append(q_g_dist[i, j])
    #         else:
    #             neg_dist.append(q_g_dist[i, j])

    plt.hist(pos_dist, label='pos')
    plt.hist(neg_dist, label='neg')
    plt.legend(loc='upper right')
    plt.show()


def view_train(root, data):
    trainmatpath = os.path.join(root, data + '_train.mat')
    trainmat = io.loadmat(trainmatpath)

    dist = compute_distance(trainmat['feature'], trainmat['feature'])
    path = trainmat['path']
    label = trainmat['label'].squeeze()

    centers = {}
    for i in range(label.shape[0]):
        if label[i] not in centers:
            centers[label[i]] = [[trainmat['feature'][i]], [path[i]]]
        else:
            centers[label[i]][0].append(trainmat['feature'][i])
            centers[label[i]][1].append(path[i])

    c_dises, q_dises = [], []
    for j in centers:
        feas = centers[j][0]
        feas = np.stack(feas)
        cent = feas.mean(axis=0)
        cent /= np.linalg.norm(cent, ord=2)
        centers[j].append(cent)
        c_dis = compute_distance(feas, np.expand_dims(cent, axis=0))
        centers[j].append(c_dis.squeeze())
        c_dises.extend(c_dis.squeeze().tolist())
        q_dis = compute_distance(feas, feas)
        for k in range(q_dis.shape[0]):
            for l in range(0, k):
                q_dises.append(q_dis[k, l])



    plt.hist(q_dises, label='c_dises', bins=200)
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    view_train('/raid/home/fufuyu/snapshot/distribute/baseline_distribute_2', 'market1501')