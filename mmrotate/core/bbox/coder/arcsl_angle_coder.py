# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
import numpy as np


@ROTATED_BBOX_CODERS.register_module()
class ARCSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6,
                 refine_range=None, weight=1.0, normalize=False):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version == 'le90', 'this model uses le90'
        assert window in ['gaussian', 'triangle', 'rect', 'pulse', 'aspect_ratio']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        assert omega == 1, 'the angle range is used 180 in this model'
        assert self.angle_range % (2 *omega) == 0, \
        'angle_range % 2omega should be 0'
        self.omega = omega
        self.window = window
        self.radius = radius
        self.coding_len = int(self.angle_range // omega)
        self.refine_range = self.coding_len // 2 if refine_range is None else refine_range
        self.weight = weight
        self.normalize = normalize

    def __copy__(self):
        return ARCSLCoder(self.angle_version, self.omega, self.window,
                          self.radius, self.refine_range, self.normalize)

    def encode(self, angle_targets, aspect_ratio=None):
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, coding_len)
        """
        # assert if()
        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.coding_len
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        elif self.window == 'aspect_ratio':
            assert aspect_ratio != None, \
            'aspect_ratio should be not a None'
            base_radius_range = torch.arange(
                -self.angle_range // (2 * self.omega),
                self.angle_range // (2 * self.omega),
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len

            smooth_value = self.aspect_ratio_smooth(aspect_ratio)


        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor) and self.window != 'aspect_ratio':
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds, angle_bias=None):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (num_anchors * H * W, coding_len)

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
        """
        angle_cls_inds = torch.argmax(angle_preds, dim=2)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        angle_pred = angle_pred * (math.pi / 180)
        if angle_bias is not None:
            angle_pred = angle_pred + angle_bias
            angle_pred = self.check_angle(angle_pred)
        return angle_pred

    def aspect_ratio_smooth(self, aspect_ratio):
        '''
        count the smoothed angle-cls labels of different aspect-ratio
        :param aspect_ratio:
        :return:
        '''
        angle = torch.arange(-self.angle_range // 2, self.angle_range // 2,
                             self.omega, device = aspect_ratio.device)
        angle = torch.abs(angle) * math.pi / 180  # to radian
        angle[angle.shape[0] // 2] = 1e-6  # prevent nan
        inds = (2 * (math.pi / 2 - torch.arctan(aspect_ratio)) /
                math.pi * 180 / self.omega).long()[:, 0].to(aspect_ratio.device)  # count the critical angle
        # x = torch.ones((aspect_ratio.shape[0], angle.shape[0])).to(aspect_ratio.device)
        # y = x.clone() * aspect_ratio
        #
        # # count the IoU in situation 1
        # inter1 = 4 * x * x / torch.sin(angle)
        # union1 = 2 * 4 * (x * y) - inter1
        # iou1 = inter1 / union1
        #
        # res = torch.zeros_like(iou1).to(aspect_ratio.device)
        #
        # # count the IoU in situation 2
        # k = (x / torch.tan(angle / 2) - y) * torch.tan(angle / 2)
        # inter2 = 4 * x * y - k * k * torch.tan(angle) - \
        #          torch.pow((2 * x - k / torch.cos(angle) - k), 2) \
        #          * torch.tan(math.pi / 2 - angle)
        # uinon2 = 2 * 4 * x * y - inter2
        # iou2 = inter2 / uinon2

        k = torch.ones((aspect_ratio.shape[0], angle.shape[0])).to(aspect_ratio.device) \
            * aspect_ratio

        iou1 = 4 / (8 * k * torch.sin(angle) - 4)

        x = (1 - k * torch.tan(angle / 2)) * torch.tan(angle)
        y = ((-2 * torch.sin(angle / 2) * torch.sin(angle / 2) +
              k * torch.sin(angle)) / torch.cos(angle))
        iou2 = (4 * k * torch.tan(angle) - x * x - y * y) / \
               (4 * k * torch.tan(angle) + x * x + y * y)

        res = torch.zeros_like(iou1).to(aspect_ratio.device)

        for i, ind in enumerate(inds):
            # replace the value of iou1 within the range of ind with
            # the value of iou2 in corresponding positions
            iou1[i, angle.shape[0] // 2 - ind: angle.shape[0] // 2 + ind + 1] = \
                iou2[i, angle.shape[0] // 2 - ind: angle.shape[0] // 2 + ind + 1]

        refine_range = self.refine_range // self.omega

        # wheter to normalize the smoothing values
        if self.normalize:
            value_min = torch.min(iou1, dim=1).values.view(-1, 1)
            iou1 = (iou1 - value_min) / (1 - value_min)
        # get the value within the refine_range
        res[:, angle.shape[0] // 2 - refine_range: angle.shape[0] // 2 + refine_range + 1] += \
            iou1[:, angle.shape[0] // 2 - refine_range: angle.shape[0] // 2 + refine_range + 1]

        return res

    @staticmethod
    def check_angle(angle):
        period =  math.pi

        angle = (angle + period / 2) % period - period / 2

        return angle