import torch

from .builder import DATASETS
from .reid_dataset import ReIDDataset


@DATASETS.register_module()
class ContrastiveDataset(ReIDDataset):

    def __getitem__(self, idx):
        img, _, _ = self.get_sample(idx)

        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)

        return dict(img=img_cat, idx=idx)
