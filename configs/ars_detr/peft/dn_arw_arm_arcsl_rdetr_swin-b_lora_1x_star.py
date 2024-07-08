# -*- coding: utf-8 -*-
_base_ = 'dn_arw_arm_arcsl_rdetr_swin-b_mona_1x_star.py'
model = dict(
    type='ARSDETRCrop',
    backbone=dict(
        type='SwinLoRA',
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
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)