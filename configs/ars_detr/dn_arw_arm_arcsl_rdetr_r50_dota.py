# -*- coding: utf-8 -*-
_base_ = 'arcsl_detr_r50_dota.py'
model = dict(
    bbox_head=dict(
        type='DNARSDeformableDETRHead',
        as_two_stage=True,
        rotate_deform_attn=True,
        aspect_ratio_weighting=True,
        dn_cfg=dict(
            type='DnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4, angle=0.02),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DNARSRotatedDeformableDetrTransformer',
            decoder=dict(type='DNARSDeformableDetrTransformerDecoder',
                    transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='RotatedMultiScaleDeformableAttention',
                            embed_dims=256)
                    ])
            )
        )
    ),
)