_base_ = [
    '../../_base_/datasets/star.py', '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='H2RBoxV2PDetectorCrop',
    backbone=dict(
        type='SwinLoRand',
        pretrain_img_size=224,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        patch_size=4,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrain/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth' # https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth
        )),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxV2PHead',
        num_classes=48,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        square_cls=[4, 44],
        # resize_cls=[1],
        scale_angle=False,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_ss_symmetry=dict(
            type='SmoothL1Loss', loss_weight=0.2, beta=0.1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = 'data/STAR/'
data = dict(
    train=dict(type='STARWSOODDataset', pipeline=train_pipeline,
               ann_file=data_root + 'train/annfiles/',
               img_prefix=data_root + 'train/images/',
               version=angle_version),
    val=dict(type='STARWSOODDataset', pipeline=test_pipeline,
             ann_file=data_root + 'test/annfiles/',
             img_prefix=data_root + 'test/images/',
             version=angle_version),
    test=dict(type='STARWSOODDataset', pipeline=test_pipeline,
              ann_file=data_root + 'test/annfiles/',
              img_prefix=data_root + 'test/images/',
              version=angle_version))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=6, metric='mAP')

