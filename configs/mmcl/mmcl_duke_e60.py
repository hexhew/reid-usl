_base_ = '../_base_/default_runtime.py'

model = dict(
    type='MMCL',
    pretrained='torchvision://resnet50',
    feat_dim=2048,
    memory_size=16522,
    base_momentum=0.5,
    start_epoch=6,
    label_generator=dict(type='MPLP', t=0.6),
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
    head=dict(type='MMCLHead', delta=5.0, r=0.01))

data_source = dict(type='DukeMTMC', data_root='/data/datasets/duke')
dataset_type = 'ReIDDataset'
train_pipeline = [
    dict(
        type='RandomCamStyle',
        camstyle_root='bounding_box_train_camstyle',
        p=0.5),
    dict(
        type='RandomResizedCrop',
        size=(256, 128),
        scale=(0.64, 1.0),
        ratio=(0.33, 0.5),
        interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=10),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='RandomErasing', value=[0.485, 0.456, 0.406])
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
    samples_per_gpu=32,  # 32 x 4 = 128
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=data_source,
        pipeline=test_pipeline,
        test_mode=True))

custom_hooks = [dict(type='MMCLHook')]
paramwise_cfg = {'backbone': dict(lr_mult=0.1)}
optimizer = dict(
    type='SGD',
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    paramwise_cfg=paramwise_cfg)
lr_config = dict(policy='step', step=[40])
total_epochs = 60
