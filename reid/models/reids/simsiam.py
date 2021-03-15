import torch

from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class SimSiam(Baseline):

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        batch_size = img.shape[0]

        img = torch.cat(torch.unbind(img, dim=1), dim=0)
        z = self.neck(self.backbone(img))[0]

        z1, z2 = torch.split(z, [batch_size, batch_size], dim=0)
        loss = self.head(z1, z2)['loss'] + self.head(z2, z1)['loss']

        return dict(loss=loss)
