from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class BoT(Baseline):

    def forward_train(self, img, **kwargs):
        f_t, f_i = self.neck(self.backbone(img), loc='both')
        losses = self.head(f_t, f_i, **kwargs)

        return losses
