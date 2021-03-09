import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from mmcv.utils import print_log
from tabulate import tabulate

from .rank import evaluate_rank
from .extract import multi_gpu_extract, single_gpu_extract
from ..distances import cosine_distance, euclidean_distance


class Evaluator:

    def __init__(self,
                 metric='cosine',
                 feat_norm=True,
                 max_rank=50,
                 topk=(1, 5, 10),
                 progbar=False):
        if metric == 'cosine':
            self.metric = cosine_distance
        elif metric == 'euclidean':
            self.metric = euclidean_distance

        self.feat_norm = feat_norm
        self.max_rank = max_rank
        self.topk = topk
        self.progbar = progbar

    def extract(self, model, data_loader):
        if dist.is_available() and dist.is_initialized():
            return multi_gpu_extract(model, data_loader, progbar=self.progbar)
        else:
            return single_gpu_extract(model, data_loader, progbar=self.progbar)

    def evaluate(self, model, data_loader, logger=None):
        dataset = data_loader.dataset
        assert dataset.test_mode
        num_query = dataset.num_query
        self.DATA_SOURCE = dataset.DATA_SOURCE

        # extract
        results = self.extract(model, data_loader)
        rank, _ = get_dist_info()
        if rank == 0:
            # only run evaluation on rank 0
            self.reid_eval(results, num_query, logger=logger)

    def reid_eval(self, results, num_query, logger=None):
        feats = results['feats']
        pids = results['pids']
        camids = results['camids']

        if self.feat_norm:
            feats = F.normalize(feats, dim=1)

        # query
        q_feats = feats[:num_query]
        q_pids = pids[:num_query]
        q_camids = camids[:num_query]

        # gallery
        g_feats = feats[num_query:]
        g_pids = pids[num_query:]
        g_camids = camids[num_query:]

        # compute distance
        dist_mat = self.metric(q_feats, g_feats)

        cmc, mAP = evaluate_rank(
            dist_mat.cpu().numpy(),
            q_pids.cpu().numpy(),
            g_pids.cpu().numpy(),
            q_camids.cpu().numpy(),
            g_camids.cpu().numpy(),
            max_rank=self.max_rank,
            cmc_topk=self.topk)

        _results = {}
        _results['mAP'] = mAP
        for k in self.topk:
            _results[f'Rank-{k}'] = cmc[k - 1]

    def report(self, results, logger=None):
        headers = ['dataset']
        headers.extend(list(results.keys()))

        outputs = [self.DATA_SOURCE]
        outputs += [val for _, val in results.items()]

        table = tabulate([outputs],
                         tablefmt='github',
                         floatfmt='.2%',
                         headers=headers,
                         numalign='left')
        print_log('\n====> Results:\n' + table, logger=logger)
