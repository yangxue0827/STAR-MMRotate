# -*- coding: utf-8 -*-
_base_ = 'csl_detr_r50_dota.py'
model = dict(
    bbox_head=dict(
        angle_coder=dict(
            window='aspect_ratio',
            normalize=True,
        )
    )
)