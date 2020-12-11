import numpy as np
from  scipy import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

def compute_distance(matrix_a, matrix_b, type='cosine'):
    if type == 'cosine':
        return np.matmul(matrix_a, matrix_b.T)

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




if __name__ == '__main__':
    test('/raid/home/fufuyu/snapshot/distribute/baseline_l8_b160_5x32_2', 'market1501')