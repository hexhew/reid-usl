import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import MemoryLayer
from ..builder import HEADS


@HEADS.register_module()
class HybridMemoryHead(nn.Module):

    def __init__(self,
                 temperature=0.05,
                 momentum=0.2,
                 feat_dim=2048,
                 memory_size=65536):
        super(HybridMemoryHead, self).__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.feat_dim = feat_dim
        self.memory_size = memory_size

        self.register_buffer(
            'features', torch.zeros((memory_size, feat_dim),
                                    dtype=torch.float))
        self.register_buffer('labels',
                             torch.zeros(memory_size, dtype=torch.long))

    def init_weights(self, **kwargs):
        pass

    @torch.no_grad()
    def update_features(self, features):
        features = F.normalize(features, dim=1)
        self.features.data.copy_(features.data)

    @torch.no_grad()
    def update_labels(self, labels):
        self.labels.data.copy_(labels.data)

    def forward(self, inputs, idx, **kwargs):
        inputs = F.normalize(inputs, dim=1)

        # inputs: B*2048, features: L*2048
        inputs = MemoryLayer.apply(inputs, idx, self.features, self.momentum)
        inputs /= self.temperature
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[idx].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels,
                        torch.ones(self.memory_size, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(),
                                    mask.t().contiguous())
        loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets)

        return dict(loss=loss)
