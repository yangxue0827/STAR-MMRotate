# -*- coding: utf-8 -*-
_base_ = 'csl_detr_swin-b_mona_1x_star.py'
model = dict(
    bbox_head=dict(
        angle_coder=dict(
            window='aspect_ratio',
            normalize=True,
        )
    )
)