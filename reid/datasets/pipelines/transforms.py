import inspect
import math
import random

import numpy as np
import torchvision.transforms as T
from mmcv.utils import build_from_cfg
from PIL import Image, ImageFilter

from ..builder import PIPELINES

_excluded_transforms = ['RandomErasing', 'RandomApply']
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


class RectScale(object):

    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


@PIPELINES.register_module()
class RandomSizedRectCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.height = size[0]
        self.width = size[1]
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.width, self.height),
                                  self.interpolation)

        # Fallback
        scale = RectScale(
            self.height, self.width, interpolation=self.interpolation)
        return scale(img)
