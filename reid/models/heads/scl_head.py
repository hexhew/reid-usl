import torch
import torch.nn as nn

from reid.utils import concat_all_gather
from ..builder import HEADS


@HEADS.register_module()
class SCLHead(nn.Module):

    def __init__(self, temperature=0.1, size_average=True):
        super(SCLHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def init_weights(self, **kwargs):
        pass

    def forward(self, features, label, **kwargs):
        N = features.shape[0]
        features = features.reshape(2 * N, -1)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        label = concat_all_gather(label)
        # label: N -> 2N
        label = label.unsqueeze(1).expand(N, 2).reshape(-1)
        # 2N x 1 -> 2N x 2N
        label = label.unsqueeze(1).expand(2 * N, 2 * N)
        is_pos = label.eq(label.t())
        is_neg = label.ne(label.t())
        # 2N x (2N - 1)
        pos_mask = torch.masked_select(is_pos, mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg, mask == 1).reshape(2 * N, -1)

        n = logit.size(0)
        loss = []

        for i in range(n):
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                negative = logit[i, neg_inds].unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature
                _label = _logit.new_zeros((1, ), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss = sum(loss)
        loss /= logit.size(0)

        return dict(loss=loss)
