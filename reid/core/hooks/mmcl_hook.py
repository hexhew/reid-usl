from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MMCLHook(Hook):

    def before_train_epoch(self, runner):
        # set epoch
        runner.model.module.set_epoch(runner.epoch)

        m = runner.model.module.base_momentum * (
            runner.epoch / runner.max_epochs)
        runner.model.module.momentum = m
