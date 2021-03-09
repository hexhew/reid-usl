import copy
import re

from mmcv.runner.optimizer import OPTIMIZERS
from mmcv.utils import build_from_cfg, print_log

from reid.utils import get_root_logger


def build_optimizer(model, cfg):
    logger = get_root_logger()

    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = copy.deepcopy(cfg)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_cfg is None:
        optimizer_cfg['params'] = model.parameters()
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)

    # set param-wise lr and weight decay
    assert isinstance(paramwise_cfg, dict)
    params = []
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        for regexp, options in paramwise_cfg.items():
            if re.search(regexp, name):
                for key, value in options.items():
                    if key.endswith('_mult'):  # a multiplier
                        key = key[:-5]
                        assert key in optimizer_cfg, \
                            f'{key} not in optimizer_cfg'
                        value = optimizer_cfg[key] * value
                    param_group[key] = value
                    print_log(
                        f'optimizer -- {name}: {key} == {value}',
                        logger=logger)

        # otherwise use the global settings
        params.append(param_group)

    optimizer_cfg['params'] = params

    return build_from_cfg(optimizer_cfg, OPTIMIZERS)
