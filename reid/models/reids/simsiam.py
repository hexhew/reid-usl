from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class SimSiam(Baseline):

    def forward_train(self, img, **kwargs):
        pass
