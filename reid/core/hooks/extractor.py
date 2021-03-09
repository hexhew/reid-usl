import torch
import torch.distributed as dist

from reid.datasets import build_dataloader, build_dataset
from ..evaluation import multi_gpu_extract, single_gpu_extract


class Extractor(object):

    def __init__(self,
                 dataset,
                 samples_per_gpu=32,
                 workers_per_gpu=4,
                 feat_dim=2048):
        self.dataset = build_dataset(dataset)
        self.distributed = dist.is_available() and dist.is_initialized()
        self.data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            dist=self.distributed,
            shuffle=False)

        self.feat_dim = feat_dim

    @torch.no_grad()
    def _dist_extract_feats(self, model):
        results = multi_gpu_extract(model, self.data_loader)

        if dist.get_rank() == 0:
            feats = results['feats'].cuda()
        else:
            feats = torch.zeros((len(self.dataset), self.feat_dim),
                                dtype=torch.float).cuda()
        dist.broadcast(feats, 0)

        return feats

    @torch.no_grad()
    def _non_dist_extract_feats(self, model):
        return single_gpu_extract(model, self.data_loader)['feats'].cuda()

    @torch.no_grad()
    def extract_feats(self, model):
        if self.distributed:
            return self._dist_extract_feats(model)
        else:
            return self._non_dist_extract_feats(model)
