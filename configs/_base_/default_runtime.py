evaluation = dict(
    interval=5,
    evaluator=dict(
        metric='cosine',
        feat_norm=True,
        max_rank=50,
        topk=(1, 5, 10),
        progbar=False))

optimizer_config = dict(grad_clip=None)
# yapf: disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf: enable
checkpoint_config = dict(interval=5)
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
