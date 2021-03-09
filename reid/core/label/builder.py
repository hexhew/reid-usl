from mmcv.utils import Registry, build_from_cfg

LABEL_GENERATORS = Registry('label generator')


def build_label_generator(cfg):
    return build_from_cfg(cfg, LABEL_GENERATORS)
