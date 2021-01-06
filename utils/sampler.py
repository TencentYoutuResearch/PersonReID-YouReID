
from collections import defaultdict
import numpy as np
import math
import copy
import random
import torch.distributed as dist
from torch.utils.data.sampler import Sampler, RandomSampler

class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances, use_tf_sample=False, use_all_sample=False):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.use_tf_sample = use_tf_sample
        self.use_all_sample = use_all_sample

        self.index_dic = defaultdict(list)
        for index, imginfo in enumerate(self.data_source.imgs):
            self.index_dic[imginfo[1]].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):

        if not self.use_tf_sample:
            batch_idxs_dict = defaultdict(list)

            for pid in self.pids:
                idxs = copy.deepcopy(self.index_dic[pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []
            
            if not self.use_all_sample:
                while len(avai_pids) >= self.num_pids_per_batch:
                    selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                    for pid in selected_pids:
                        batch_idxs = batch_idxs_dict[pid].pop(0)
                        final_idxs.extend(batch_idxs)
                        if len(batch_idxs_dict[pid]) == 0:
                            avai_pids.remove(pid)
            #else:
            #    final_idxs.
            #    for pid in selected_pids:
            #        final_idxs.
            return iter(final_idxs)

        else:
            avai_pids = copy.deepcopy(self.pids)
            index_dict = copy.deepcopy(self.index_dic)
            for pid in index_dict:
                random.shuffle(index_dict[pid])
            index_dict_temp = copy.deepcopy(index_dict)
            final_idxs = []
            ncount = self.length // self.batch_size + 1
            for i in range(ncount):
                select_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in select_pids:
                    if len(index_dict_temp[pid]) < self.num_instances:
                        pid_images = copy.deepcopy(index_dict[pid])
                        random.shuffle(pid_images)
                        index_dict_temp[pid] = pid_images
                    for i in range(self.num_instances):
                        idx = index_dict_temp[pid].pop()
                        final_idxs.append(idx)

            return iter(final_idxs)


    def __len__(self):
        return self.length


class DistributeRandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances,
                 use_tf_sample=False, use_all_sample=False, rnd_select_nid=0,
                 shuffle=True, seed=0):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.use_tf_sample = use_tf_sample
        self.use_all_sample = use_all_sample
        self.rnd_select_nid = rnd_select_nid

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        # print(self.num_replicas, self.rank)
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle

        self.index_dic = defaultdict(list)
        for index, imginfo in enumerate(self.data_source.imgs):
            self.index_dic[imginfo[1]].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - (0 if self.use_tf_sample else num % self.num_instances)

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.epoch + self.seed)
        if not self.use_tf_sample:
            batch_idxs_dict = defaultdict(list)
            for pid in self.pids:
                idxs = copy.deepcopy(self.index_dic[pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            if not self.use_all_sample:
                while len(avai_pids) >= (self.num_pids_per_batch - self.rnd_select_nid):
                    selected_pids = np.random.choice(avai_pids, size=self.num_pids_per_batch- self.rnd_select_nid,
                                                     replace=False)
                    for pid in selected_pids:
                        batch_idxs = batch_idxs_dict[pid].pop(0)
                        final_idxs.append(batch_idxs)
                        if len(batch_idxs_dict[pid]) == 0:
                            avai_pids.remove(pid)

                    for i in range(self.rnd_select_nid):
                        final_idxs.append(
                            np.random.choice(self.length, size=self.num_instances, replace=False).tolist()
                        )


            num_samples = int(math.ceil(len(final_idxs) * 1.0 / (self.num_replicas * self.num_pids_per_batch)))
            total_size = num_samples * self.num_replicas * self.num_pids_per_batch
            final_idxs += final_idxs[:(total_size - len(final_idxs))]
            rank_indices = final_idxs[self.rank:total_size:self.num_replicas]
            rank_idxs = []
            for r in rank_indices:
                rank_idxs.extend(r)

            return iter(rank_idxs)

        else:
            print('use mgn sample')
            avai_pids = copy.deepcopy(self.pids)
            index_dict = copy.deepcopy(self.index_dic)
            for pid in index_dict:
                random.shuffle(index_dict[pid])
            index_dict_temp = copy.deepcopy(index_dict)
            final_idxs = []
            num_samples = int(math.ceil(self.length * 1.0 / (self.num_replicas * self.batch_size)))
            total_size = num_samples * self.num_replicas * self.num_pids_per_batch
            # print('total', total_size)
            for i in range(total_size):
                select_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in select_pids:
                    if len(index_dict_temp[pid]) < self.num_instances:
                        pid_images = copy.deepcopy(index_dict[pid])
                        random.shuffle(pid_images)
                        index_dict_temp[pid] = pid_images
                    batch_idxs = []
                    for i in range(self.num_instances):
                        idx = index_dict_temp[pid].pop()
                        batch_idxs.append(idx)
                    final_idxs.append(batch_idxs)
            rank_indices = final_idxs[self.rank:total_size:self.num_replicas]
            # print('rank_indices', rank_indices)
            rank_idxs = []
            for r in rank_indices:
                rank_idxs.extend(r)
            # print('rank_idxs', len(rank_idxs))
            return iter(rank_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch





