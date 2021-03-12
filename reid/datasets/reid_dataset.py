from mmcv.runner import get_dist_info
from torch.utils.data import Dataset

from .builder import DATASETS, build_data_source
from .pipelines import build_pipeline


@DATASETS.register_module()
class ReIDDataset(Dataset):

    def __init__(self, data_source, pipeline=None, test_mode=False):
        self.data_source = build_data_source(data_source)
        self.DATA_SOURCE = self.data_source.DATA_SOURCE
        self.test_mode = test_mode

        rank, _ = get_dist_info()
        verbose = True if rank == 0 else False
        self.img_items, self.pids, self.camids = self.data_source.get_data(
            test_mode=self.test_mode, verbose=verbose)

        if not self.test_mode:
            # pid -> label
            self.pid_dict = {p: i for i, p in enumerate(self.pids)}

        if self.test_mode:
            # number of query, gallery for evaluation
            self.num_query = len(self.data_source.query)
            self.num_gallery = len(self.data_source.gallery)

        if pipeline is not None:
            self.pipeline = build_pipeline(pipeline, dataset=self)
        else:
            self.pipeline = None

    def __len__(self):
        return len(self.img_items)

    def get_sample(self, idx):
        img, pid, camid = self.img_items[idx]

        return img, pid, camid

    def __getitem__(self, idx):
        img, pid, camid = self.get_sample(idx)
        label = self.pid_dict[pid] if not self.test_mode else pid
        results = dict(img=img, label=label, pid=pid, camid=camid, idx=idx)

        return self.pipeline(results)
