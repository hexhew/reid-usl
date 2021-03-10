import torch

from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class SimSiam(Baseline):

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        N, _, C, H, W = img.shape

        img = img.reshape(N * 2, C, H, W)
        z = self.neck(self.backbone(img))[0]

        z1, z2 = torch.unbind(z.reshape(N, 2, -1), dim=1)
        loss = self.head(z1, z2)['loss'] + self.head(z2, z1)['loss']

        return dict(loss=loss)
