from ..builder import REIDS, build_backbone, build_head, build_neck
from .base import BaseModel


@REIDS.register_module()
class Baseline(BaseModel):

    def __init__(self, backbone, neck, head=None, pretrained=None):
        super(Baseline, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(Baseline, self).init_weights(pretrained=pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.head.init_weights()

    def forward_test(self, img, **kwargs):
        return self.neck(self.backbone(img))[0]

    def forward_train(self, img, **kwargs):
        z = self.neck(self.backbone(img))[0]
        losses = self.head(z, **kwargs)

        return losses
