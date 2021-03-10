import torch
import torch.nn.functional as F

from reid.core import build_label_generator
from ..builder import REIDS
from ..utils import MemoryLayer
from .baseline import Baseline


@REIDS.register_module()
class MMCL(Baseline):

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 pretrained=None,
                 feat_dim=2048,
                 memory_size=65536,
                 base_momentum=0.5,
                 start_epoch=6,
                 label_generator=dict(type='MPLP', t=0.6)):
        super().__init__(backbone, neck, head, pretrained=pretrained)
        self.feat_dim = feat_dim
        self.memory_size = memory_size
        self.register_buffer(
            'features', torch.zeros((memory_size, feat_dim),
                                    dtype=torch.float))

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.start_epoch = start_epoch
        self._epoch = 0

        self.label_generator = build_label_generator(label_generator)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def forward_test(self, img, **kwargs):
        # MMCL uses pooling-5 features
        return self.neck(self.backbone(img), loc='both')[0]

    def forward_train(self, img, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Inputed images of shape (N, C, H, W).
            idx (Tensor): Indices of images in the dataset.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        z = self.neck(self.backbone(img))[0]
        z = F.normalize(z, dim=1)
        logits = MemoryLayer.apply(z, idx, self.features, self.momentum)

        with torch.no_grad():
            if self._epoch >= self.start_epoch - 1:
                labels = self.label_generator.gen_labels(self.features, idx)
            else:
                labels = logits.new_zeros(logits.size(), dtype=torch.long)
                labels.scatter_(1, idx.unsqueeze(1), 1)
        losses = self.head(logits, labels)

        return losses
