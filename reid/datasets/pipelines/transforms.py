import inspect

import torchvision.transforms as T

from ..builder import PIPELINES

for m in inspect.getmembers(T, inspect.isclass):
    if m[0] != 'RandomErasing':
        PIPELINES.register_module(name=m[0], module=m[1])
