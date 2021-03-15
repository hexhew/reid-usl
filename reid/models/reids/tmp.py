import torch
import torch.nn.functional as F

from ..builder import REIDS
from ..utils import GatherLayer
from .baseline import Baseline


@REIDS.register_module()
class TMP(Baseline):

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        batch_size = img.shape[0]

        img = torch.cat(torch.unbind(img, dim=1), dim=0)
        z = self.neck(self.backbone(img))[0]
        z = F.normalize(z, dim=1)

        z1, z2 = torch.split(z, [batch_size, batch_size], dim=0)
        z = torch.cat((z1.unsqueeze(1), z2.unsqueeze(1)), dim=1)
        z = torch.cat(GatherLayer.apply(z), dim=0)
        losses = self.head(z, **kwargs)

        return losses
