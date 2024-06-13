# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Tuple, Union

import torch
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from mmrotate.models.detectors.single_stage_crop import RotatedSingleStageDetectorCrop
from torch.nn.functional import grid_sample
from mmrotate.core import rbbox2result
from torchvision import transforms


@ROTATED_DETECTORS.register_module()
class H2RBoxV2PDetector(RotatedSingleStageDetector):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 prob_rot=0.95,
                 view_range=(0.25, 0.75),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained, init_cfg)
        self.prob_rot = prob_rot
        self.view_range = view_range

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Add an id to each annotation to match objects in different views
        offset = 1
        for i, bboxes in enumerate(gt_bboxes):
            bids = torch.arange(
                0, len(bboxes), 1, device=bboxes.device) + offset
            gt_bboxes[i] = torch.cat((bboxes, bids[:, None]), dim=-1)
            offset += len(bboxes)

        # Concat original/rotated/flipped images and gts in the batch dim
        if torch.rand(1) < self.prob_rot:
            rot = math.pi * (
                torch.rand(1).item() *
                (self.view_range[1] - self.view_range[0]) + self.view_range[0])
            img_ss = transforms.functional.rotate(img, -rot / math.pi * 180)
            
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, :2] = (bboxes[..., :2] - ctr).matmul(tf.T) + ctr
                bboxes[:, 4] = bboxes[:, 4] + rot
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            img = torch.cat((img, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_labels = gt_labels + gt_labels
            for m in img_metas:
                m['ss'] = ('rot', rot)
        else:
            img_ss = transforms.functional.vflip(img)
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, 1] = img.shape[-2] - bboxes[:, 1]
                bboxes[:, 4] = -bboxes[:, 4]
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            img = torch.cat((img, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_labels = gt_labels + gt_labels
            for m in img_metas:
                m['ss'] = ('flp', 0)
        
        # from mmrotate.core import imshow_det_rbboxes
        # import numpy as np
        # for i, bboxes in enumerate(gt_bboxes):
        #     _img = img[i].detach().permute(1, 2, 0)[
        #         ..., [2, 1, 0]].cpu().numpy()
        #     _img = (_img * np.array([58.395, 57.12, 57.375]) + np.array(
        #         [123.675, 116.28, 103.53])).clip(0, 255).astype(
        #         np.uint8)
        #     imshow_det_rbboxes(_img, bboxes=bboxes[:, :-1].detach().cpu().numpy(),
        #                        labels=np.arange(len(bboxes)), class_names=bboxes[:, -1], out_file=f'{i}.png')

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)  
        return losses


@ROTATED_DETECTORS.register_module()
class H2RBoxV2PDetectorCrop(RotatedSingleStageDetectorCrop):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 prob_rot=0.95,
                 view_range=(0.25, 0.75),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained, init_cfg)
        self.prob_rot = prob_rot
        self.view_range = view_range

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Add an id to each annotation to match objects in different views
        offset = 1
        for i, bboxes in enumerate(gt_bboxes):
            bids = torch.arange(
                0, len(bboxes), 1, device=bboxes.device) + offset
            gt_bboxes[i] = torch.cat((bboxes, bids[:, None]), dim=-1)
            offset += len(bboxes)

        # Concat original/rotated/flipped images and gts in the batch dim
        if torch.rand(1) < self.prob_rot:
            rot = math.pi * (
                torch.rand(1).item() *
                (self.view_range[1] - self.view_range[0]) + self.view_range[0])
            img_ss = transforms.functional.rotate(img, -rot / math.pi * 180)
            
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, :2] = (bboxes[..., :2] - ctr).matmul(tf.T) + ctr
                bboxes[:, 4] = bboxes[:, 4] + rot
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            img = torch.cat((img, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_labels = gt_labels + gt_labels
            for m in img_metas:
                m['ss'] = ('rot', rot)
        else:
            img_ss = transforms.functional.vflip(img)
            gt_bboxes_ss = copy.deepcopy(gt_bboxes)
            for bboxes in gt_bboxes_ss:
                bboxes[:, 1] = img.shape[-2] - bboxes[:, 1]
                bboxes[:, 4] = -bboxes[:, 4]
                bboxes[:, 5] = bboxes[:, 5] + 0.5

            img = torch.cat((img, img_ss), 0)
            gt_bboxes = gt_bboxes + gt_bboxes_ss
            gt_labels = gt_labels + gt_labels
            for m in img_metas:
                m['ss'] = ('flp', 0)
        
        # from mmrotate.core import imshow_det_rbboxes
        # import numpy as np
        # for i, bboxes in enumerate(gt_bboxes):
        #     _img = img[i].detach().permute(1, 2, 0)[
        #         ..., [2, 1, 0]].cpu().numpy()
        #     _img = (_img * np.array([58.395, 57.12, 57.375]) + np.array(
        #         [123.675, 116.28, 103.53])).clip(0, 255).astype(
        #         np.uint8)
        #     imshow_det_rbboxes(_img, bboxes=bboxes[:, :-1].detach().cpu().numpy(),
        #                        labels=np.arange(len(bboxes)), class_names=bboxes[:, -1], out_file=f'{i}.png')

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)  
        return losses
