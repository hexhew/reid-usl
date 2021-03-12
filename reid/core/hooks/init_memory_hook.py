import torch
from mmcv.runner import HOOKS, Hook

from .extractor import Extractor


@HOOKS.register_module()
class InitMemoryHook(Hook):
    """A hook that extracts features and initialize memory of the model
    before the start of training.
    """

    def __init__(self, extractor):
        self.extractor = Extractor(**extractor)

    def before_run(self, runner):
        with torch.no_grad():
            feats = self.extractor.extract_feats(runner.model)

        if hasattr(runner.model.module, 'init_memory'):
            runner.model.module.init_memory(feats)
        elif hasattr(runner.model.module.head, 'init_memory'):
            runner.model.module.head.init_memory(feats)
        else:
            raise NotImplementedError(
                'model or its head should has a method init_memory')
        del feats
