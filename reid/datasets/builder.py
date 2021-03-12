import platform
import random
from functools import partial

import numpy as np
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATA_SOURCES = Registry('data source')
DATASETS = Registry('dataset')
SAMPLERS = Registry('sampler')


def build_data_source(cfg):
    return build_from_cfg(cfg, DATA_SOURCES)


def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)


def build_sampler(sampler_cfg,
                  dataset,
                  batch_size,
                  shuffle=True,
                  dist=True,
                  seed=1):
    """Build sampler for data loader.
    """
    if sampler_cfg is not None:
        sampler = sampler_cfg['type']
        if dist and 'Distributed' not in sampler:
            sampler_cfg['type'] = f'Distributed{sampler}'
        sampler_cfg['dataset'] = dataset
        sampler_cfg['batch_size'] = batch_size
        sampler_cfg['seed'] = seed

        return build_from_cfg(sampler_cfg, SAMPLERS)
    else:
        if dist:
            sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        else:
            sampler = RandomSampler(dataset) if shuffle else None

    return sampler


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     sampler=None,
                     seed=1,
                     **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    sampler = build_sampler(
        sampler, dataset, batch_size, shuffle=shuffle, dist=dist, seed=seed)

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
