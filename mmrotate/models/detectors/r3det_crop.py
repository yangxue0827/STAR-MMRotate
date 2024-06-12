# Copyright (c) SJTU. All rights reserved.
import warnings

from mmcv.runner import ModuleList

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import FeatureRefineModule

from .single_stage_img_split_bridge_tools import *
from mmdet.utils import get_device

def resize_bboxes(bboxes,scale):
    """Resize bounding boxes with scales."""

    orig_shape = bboxes.shape
    out_boxxes=bboxes.clone().reshape((-1, 5))
    # bboxes = bboxes.reshape((-1, 5))
    w_scale = scale
    h_scale = scale
    out_boxxes[:, 0] *= w_scale
    out_boxxes[:, 1] *= h_scale
    out_boxxes[:, 2:4] *= np.sqrt(w_scale * h_scale)

    return out_boxxes

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0)
    return inputs

def FullImageCrop(self, imgs, bboxes, labels, patch_shape,
                  gaps,
                  jump_empty_patch=False,
                  mode='train'):
    """
    Args:
        imgs (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        bboxes (list[Tensor]): Each item are the truth boxes for each
            image in [tl_x, tl_y, br_x, br_y] format.
        labels (list[Tensor]): Class indices corresponding to each box
    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    out_imgs=[]
    out_bboxes=[]
    out_labels=[]
    out_metas=[]
    device = get_device()
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值

    if mode == 'train':
        # for i in range(imgs.shape[0]):
        for img, bbox, label in zip(imgs, bboxes, labels):
            p_imgs = []
            p_bboxes = []
            p_labels = []
            p_metas = []
            img = img.cpu()
            # patch
            info = dict()
            info['labels'] = np.array(torch.tensor(label, device='cpu',requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[1]
            info['height'] = img.shape[2]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, self.version))

            sizes = [patch_shape[0]]
            # gaps=[0]
            windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
            window_anns = get_window_obj(info, windows, iof_thr)
            patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
                                                    img,
                                                    no_padding=True,
                                                    padding_value=[104, 116, 124])

            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    if patch_info['labels'] == [-1]:
                        continue                

                obj = patch_info['ann']
                tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), self.version)
                p_bboxes.append(tmp_boxes.to(device))
                p_labels.append(torch.tensor(patch_info['labels'], device=device))
                p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                                'y_start': torch.tensor(patch_info['y_start'], device=device),
                                'shape': patch_shape, 'trunc': torch.tensor(obj['trunc'], device=device)})

                patch = patchs[i]
                p_imgs.append(patch.to(device))

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

    elif mode =='test':
        p_imgs = []
        p_metas = []
        img = imgs.cpu().squeeze(0)
        # patch
        info = dict()
        info['labels'] = np.array(torch.tensor([], device='cpu'))
        info['ann'] = {'bboxes': {}}
        info['width'] = img.shape[1]
        info['height'] = img.shape[2]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows,img,
                                                  no_padding=False,
                                                  padding_value=[104, 116, 124])

        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape,'img_shape':patch_shape, 'scale_factor':1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas


    return out_imgs, out_bboxes,out_labels, out_metas

def get_single_img(fea_g_necks, i):
    fea_g_neck=[]
    for idx in range(len(fea_g_necks)):
        fea_g_neck.append(fea_g_necks[idx][i])

    return tuple(fea_g_neck)

def relocate(idx, local_bboxes, patch_meta):
    # put patches' local bboxes to full img via patch_meta
    meta=patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    for i in range(len(local_bboxes)):
        bbox = local_bboxes[i]
        if bbox.size()[0] == 0:
            continue
        bbox[0] += left
        bbox[1] += top

    return



@ROTATED_DETECTORS.register_module()
class R3Det(RotatedBaseDetector):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 num_refine_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(R3Det, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.num_refine_stages = num_refine_stages
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg['s0'])
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.feat_refine_module = ModuleList()
        self.refine_head = ModuleList()
        for i, (frm_cfg,
                refine_head) in enumerate(zip(frm_cfgs, refine_heads)):
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg['sr'][i])
            refine_head.update(test_cfg=test_cfg)
            self.refine_head.append(build_head(refine_head))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f's0.{name}'] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'sr{i}.{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        return losses

    # def simple_test(self, img, img_meta, rescale=False):
    #     """Test function without test time augmentation.

    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes. \
    #             The outer list corresponds to each image. The inner list \
    #             corresponds to each class.
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     rois = self.bbox_head.filter_bboxes(*outs)
    #     # rois: list(indexed by images) of list(indexed by levels)
    #     for i in range(self.num_refine_stages):
    #         x_refine = self.feat_refine_module[i](x, rois)
    #         outs = self.refine_head[i](x_refine)
    #         if i + 1 in range(self.num_refine_stages):
    #             rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
    #     bbox_results = [
    #         rbbox2result(det_bboxes, det_labels,
    #                      self.refine_head[-1].num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        gaps = [200]
        patch_shape = (1024, 1024)
        p_bs = 4  # patch batchsize
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')
        local_bboxes_lists=[]
        for i in range(img.shape[0]):
            j = 0
            patches = list2tensor(p_imgs[i])
            patches_meta = p_metas[i]
            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs] 
                with torch.no_grad():
                    patch=patch.cuda() 
                    x = self.extract_feat(patch)
                    outs = self.bbox_head(x)
                    rois = self.bbox_head.filter_bboxes(*outs)
                    # rois: list(indexed by images) of list(indexed by levels)
                    for i in range(self.num_refine_stages):
                        x_refine = self.feat_refine_module[i](x, rois)
                        outs = self.refine_head[i](x_refine)
                        if i + 1 in range(self.num_refine_stages):
                            rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
                    bbox_inputs = outs + (patch_meta, self.test_cfg, rescale)
                    local_bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
              
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes, det_labels = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)

                j = j+p_bs

        bbox_list = [merge_results([local_bboxes_lists],iou_thr=0.4)]
        bbox_results = [
            rbbox2result(det_bboxes, det_labels,
                         self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
