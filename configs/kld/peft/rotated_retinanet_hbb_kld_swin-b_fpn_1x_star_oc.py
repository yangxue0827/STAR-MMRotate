_base_ = ['./rotated_retinanet_hbb_kld_swin-b_mona_fpn_1x_star_oc.py']

model = dict(
    type='RotatedRetinaNetCrop',
    backbone=dict(
        type='SwinTransformer',
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
        add_extra_convs='on_input',
        num_outs=5),
)