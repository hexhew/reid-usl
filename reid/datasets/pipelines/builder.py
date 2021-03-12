from mmcv.utils import Registry, build_from_cfg

from .compose import Compose

PIPELINES = Registry('pipeline')


def build_pipeline(pipeline, dataset=None):
    transforms = []

    if pipeline[0]['type'] == 'RandomCamStyle':
        assert dataset is not None
        camstyle = pipeline.pop(0)
        camstyle['dataset'] = dataset
        transforms.append(build_from_cfg(camstyle, PIPELINES))

    for _pipe in pipeline:
        transforms.append(build_from_cfg(_pipe, PIPELINES))

    return Compose(transforms)
