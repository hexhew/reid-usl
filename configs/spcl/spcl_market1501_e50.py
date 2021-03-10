_base_ = '../_base_/default_runtime.py'

model = dict(
    type='Baseline',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='BN'),
        norm_eval=False),
    neck=dict(
        type='BNNeck',
        feat_dim=2048,
        norm_cfg=dict(type='BN1d'),
        with_bias=False,
        with_avg_pool=True,
        avgpool=dict(type='AvgPoolNeck')),
    head=dict(
        type='HybridMemoryHead',
        temperature=0.05,
        momentum=0.2,
        feat_dim=2048,
        memory_size=12936))

data_source = dict(type='Market1501', data_root='/data/datasets/market1501')
dataset_type = 'PseudoLabelDataset'
train_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='Pad', padding=10),
    dict(type='RandomCrop', size=(256, 128)),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='RandomErasing')
]
test_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
]
data = dict(
    samples_per_gpu=32,  # 32 x 2 = 64
    workers_per_gpu=4,
    sampler=dict(type='FixedStepIdentitySampler', num_instances=4, steps=400),
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=data_source,
        pipeline=test_pipeline,
        test_mode=True))

custom_hooks = [
    dict(
        type='SpCLHook',
        extractor=dict(
            dataset=dict(
                type='ReIDDataset',
                data_source=data_source,
                pipeline=test_pipeline),
            samples_per_gpu=32,
            workers_per_gpu=4),
        label_generator=dict(
            type='SelfPacedGenerator',
            eps=[0.58, 0.6, 0.62],
            min_samples=4,
            k1=30,
            k2=6))
]
optimizer = dict(type='Adam', lr=0.00035, weight_decay=5e-4)
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
