import math
import random
from collections import defaultdict

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from ..builder import SAMPLERS


@SAMPLERS.register_module()
class IdentitySampler(Sampler):

    def __init__(self, dataset, num_instances=4, **kwargs):
        self.dataset = dataset
        self.num_instances = num_instances

        self.init_data()

    def init_data(self):
        self.index_pid = defaultdict(int)
        self.pid_inds = defaultdict(list)
        self.pid_cams = defaultdict(list)

        for index, (_, pid, camid) in enumerate(self.dataset.img_items):
            self.index_pid[index] = pid
            self.pid_cams[pid].append(camid)
            self.pid_inds[pid].append(index)

        self.pids = list(self.pid_inds.keys())
        self.num_samples = len(self.pids)
        self.total_size = self.num_samples

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        yield from self._gen_iter_list()

    def _gen_iter_list(self):
        indices = torch.randperm(len(self.pids)).tolist()
        indices += indices[:self.total_size - len(indices)]

        yield from self._sample_list(indices)

    def _sample_list(self, indices):
        ret = []

        for p_idx in indices:
            i = random.choice(self.pid_inds[self.pids[p_idx]])
            _, i_pid, i_camid = self.dataset.img_items[i]
            ret.append(i)

            cams = self.pid_cams[i_pid]
            inds = self.pid_inds[i_pid]
            select_cams = No_index(cams, i_camid)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(
                        select_cams,
                        size=self.num_instances - 1,
                        replace=False)
                else:
                    cam_indexes = np.random.choice(
                        select_cams, size=self.num_instances - 1, replace=True)

                for _idx in cam_indexes:
                    ret.append(inds[_idx])
            else:
                select_indexes = No_index(inds, i)
                if not select_indexes:
                    continue
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(
                        select_indexes,
                        size=self.num_instances - 1,
                        replace=False)
                else:
                    ind_indexes = np.random.choice(
                        select_indexes,
                        size=self.num_instances - 1,
                        replace=True)

                for _idx in ind_indexes:
                    ret.append(inds[_idx])

        return ret


@SAMPLERS.register_module()
class FixedStepsIdentitySampler(IdentitySampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances=4,
                 steps=400,
                 seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.steps = steps

        self.num_samples = self.steps * self.batch_size
        self.total_size = self.num_samples

    def init_data(self):
        self.index_pid = defaultdict(int)
        self.pid_inds = defaultdict(list)
        self.pid_cams = defaultdict(list)

        for index, (_, pid, camid) in enumerate(self.dataset.img_items):
            self.index_pid[index] = pid
            self.pid_cams[pid].append(camid)
            self.pid_inds[pid].append(index)

        self.pids = list(self.pid_inds.keys())

        indices = []
        while len(indices) < self.num_samples:
            _inds = torch.randperm(len(self.pids)).tolist()
            indices.extend(self._sample_list(_inds))

        self.indices = indices[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.init_data()
        return iter(self.indices)


@SAMPLERS.register_module()
class DistributedFixedStepsIdentitySampler(FixedStepsIdentitySampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances=4,
                 steps=400,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _world_size = get_dist_info()
        if num_replicas is None:
            num_replicas = _world_size
        if rank is None:
            rank = _rank
        self.num_replicas = num_replicas
        self.rank = rank
        super().__init__(dataset, batch_size, num_instances, steps)

        self.num_samples = int(
            math.ceil(self.batch_size * self.steps / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.seed = seed

    def __iter__(self):
        self.init_data()
        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]
