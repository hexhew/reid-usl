import torch
import torch.nn.functional as F

from ..utils import GatherLayer
from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class TMP(Baseline):

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'

        N, _, C, H, W = img.shape
        img = img.reshape(N * 2, C, H, W)
        z = self.neck(self.backbone(img))[0]
        z = F.normalize(z, dim=1)
        z = torch.cat(GatherLayer.apply(z), dim=0)

        assert z.shape[0] % 2 == 0
        batch_size = int(z.shape[0] / 2)
        z = z.reshape(batch_size, 2, -1)
        losses = self.head(z, **kwargs)  # SupContrastHead

        return losses
