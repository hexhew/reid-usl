from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochHook(Hook):

    def before_train_epoch(self, runner):
        # set epoch
        runner.model.module.set_epoch(
            runner.epoch, max_epochs=runner.max_epochs)
