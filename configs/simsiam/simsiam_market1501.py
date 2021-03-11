_base_ = '../_base_/default_runtime.py'

model = dict(
    type='SimSiam',
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
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearPredictor',
            in_channels=2048,
            hid_channels=4096,
            out_channels=2048)))

data_source = dict(type='Market1501', data_root='/data/datasets/market1501')
dataset_type = 'ContrastiveDataset'
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
    samples_per_gpu=64,  # 64 x 8 = 512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=data_source,
        pipeline=test_pipeline,
        test_mode=True))

optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9)
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
