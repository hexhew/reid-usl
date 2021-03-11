import torch
import torch.nn as nn

from reid.utils import concat_all_gather
from ..builder import HEADS


# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
@HEADS.register_module()
class SupContrastHead(nn.Module):

    def __init__(self, temperature=0.2, contrast_mode='all', with_label=False):
        super(SupContrastHead, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.with_label = with_label

    def init_weights(self, **kwargs):
        pass

    def forward(self, features, label=None, mask=None, **kwargs):
        assert features.dim() == 3, \
            f'features must be 3 dims, but got: {features.dim()}'

        batch_size = features.shape[0]
        if not self.with_label and label is not None:
            # set label to None if using no labels
            label = None

        if label is not None and mask is not None:
            raise ValueError('cannot define both label and mask')
        elif label is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float).cuda()
        elif label is not None:
            label = concat_all_gather(label)
            label = label.contiguous().view(-1, 1)
            if label.shape[0] != batch_size:
                raise ValueError(
                    'Number of labels does not match number of features')
            mask = torch.eq(label, label.T).float().cuda()
        else:
            mask = concat_all_gather(mask)
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unsupported mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likehood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = (-mean_log_prob_pos).view(anchor_count, batch_size).mean()

        return dict(loss=loss)
