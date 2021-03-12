import inspect
import random

import numpy as np
import torchvision.transforms as T
from mmcv.utils import build_from_cfg
from PIL import ImageFilter

from ..builder import PIPELINES

_excluded_transforms = ['RandomApply']
for m in inspect.getmembers(T, inspect.isclass):
    if m[0] not in _excluded_transforms:
        PIPELINES.register_module(name=m[0], module=m[1])


@PIPELINES.register_module()
class RandomApply(T.RandomApply):

    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = [build_from_cfg(t, PIPELINES) for t in transforms]

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)

        return img


@PIPELINES.register_module()
class GaussianBlur(object):

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return img
