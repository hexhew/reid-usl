import copy

import torch

from .builder import DATASETS
from .pseudo_label import PseudoLabelDataset


@DATASETS.register_module()
class ContrastiveDataset(PseudoLabelDataset):

    def __getitem__(self, idx):
        img, pid, camid = self.get_sample(idx)
        label = self.pid_dict[pid] if not self.test_mode else pid
        results = dict(img=img, label=label, pid=pid, camid=camid, idx=idx)

        img1 = self.pipeline(copy.deepcopy(results))['img']
        img2 = self.pipeline(copy.deepcopy(results))['img']
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        results['img'] = img_cat

        return results
