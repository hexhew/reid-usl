import torch.nn as nn
from mmcv.cnn import is_norm, kaiming_init, normal_init

from ..builder import NECKS, build_neck


@NECKS.register_module()
class NonLinearPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=False,
                 avgpool=dict(type='AvgPoolNeck')):
        super(NonLinearPredictor, self).__init__()
        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avgpool = build_neck(avgpool)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear=init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)

        return [self.mlp(x[0])]


@NECKS.register_module()
class Projection(nn.Module):

    def __init__(self,
                 in_channels=2048,
                 hid_channels=2048,
                 out_channels=2048,
                 num_layers=3,
                 with_avg_pool=False,
                 avgpool=dict(type='AvgPoolNeck')):
        super(Projection, self).__init__()
        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avgpool = build_neck(avgpool)

        self.num_layers = num_layers
        self.mlp = nn.ModuleList()
        for i in range(self.num_layers - 1):
            channels = in_channels if i == 0 else hid_channels
            self.mlp.append(nn.Linear(channels, hid_channels))
            self.mlp.append(nn.BatchNorm1d(hid_channels))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(hid_channels, out_channels))
        self.mlp.append(nn.BatchNorm1d(out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear=init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)

        x = x[0]
        for layer in self.mlp:
            x = layer(x)

        return [x]


def _init_weights(module, init_linear='normal', std=0.01, bias=0):
    assert init_linear in ['normal', 'kaiming']

    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif is_norm(m):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
