_base_ = ['./rotated_retinanet_obb_r50_fpn_1x_star_le90.py']

fp16 = dict(loss_scale='dynamic')
