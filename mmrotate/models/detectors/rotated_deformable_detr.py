# Copyright (c) OpenMMLab. All rights reserved.
# from ..builder import DETECTORS
# from mmdet.models.builder import DETECTORS
from ..builder import ROTATED_DETECTORS
from .rotated_detr_crop import RotatedDETRCrop
from .rotated_detr import RotatedDETR


@ROTATED_DETECTORS.register_module()
class RotatedDeformableDETR(RotatedDETR):

    def __init__(self, *args, **kwargs):
        super(RotatedDETRCrop, self).__init__(*args, **kwargs)
        

@ROTATED_DETECTORS.register_module()
class RotatedDeformableDETRCrop(RotatedDETRCrop):

    def __init__(self, *args, **kwargs):
        super(RotatedDETRCrop, self).__init__(*args, **kwargs)