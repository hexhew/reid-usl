import math
import random
from collections import defaultdict

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from ..builder import SAMPLERS
from ._identity_deprecated import (  # noqa
    DistributedFixedStepsIdentitySampler, FixedStepsIdentitySampler,
    IdentitySampler)


@SAMPLERS.register_module()
class FixedStepIdentitySampler(Sampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances=4,
                 step=400,
                 with_camid=True,
                 seed=0,
                 **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.step = step
        self.with_camid = with_camid

        self.seed = seed
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        self.num_samples = self.batch_size * self.step
        self.total_size = self.num_samples

    def init_data(self):
        self.index_pid = defaultdict(int)  # index -> pid
        self.pid_cams = defaultdict(list)  # pid -> [camids]
        self.pid_inds = defaultdict(list)  # pid -> [indexes]

        for index, (_, pid, camid) in enumerate(self.dataset.img_items):
            self.index_pid[index] = pid
            self.pid_cams[pid].append(camid)
            self.pid_inds[pid].append(index)

        self.pids = list(self.pid_inds.keys())

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.init_data()

        indices = []
        while len(indices) < self.num_samples:
            _indices = torch.randperm(
                len(self.pids), generator=self.g).tolist()
            indices.extend(self._sample_list(_indices))

        indices = indices[:self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

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
class DistributedFixedStepIdentitySampler(FixedStepIdentitySampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances=4,
                 step=400,
                 with_camid=True,
                 seed=0,
                 num_replicas=None,
                 rank=None,
                 **kwargs):
        _rank, _world_size = get_dist_info()
        if num_replicas is None:
            num_replicas = _world_size
        if rank is None:
            rank = _rank
        self.num_replicas = num_replicas
        self.rank = rank
        # batch_size of single process -> total batch size
        batch_size = batch_size * self.num_replicas
        super().__init__(dataset, batch_size, num_instances, step, with_camid)

        num_samples = self.batch_size * self.step
        self.num_samples = int(math.ceil(num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        self.init_data()

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        num_ids = len(self.pids)
        num_ids_per_proc = int(math.ceil(num_ids / self.num_replicas))
        num_ids = num_ids_per_proc * self.num_replicas

        indices = []
        while len(indices) < self.total_size:
            _indices = torch.randperm(
                len(self.pids), generator=self.g).tolist()
            _indices += _indices[:num_ids - len(self.pids)]
            # subsample
            indices.extend(
                self._sample_list(
                    _indices[self.rank:num_ids:self.num_replicas]))
        indices = indices[:self.num_samples]

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]
