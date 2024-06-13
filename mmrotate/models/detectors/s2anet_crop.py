# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import AlignConvModule

from mmrotate.models.detectors.single_stage_img_split_bridge_tools import *
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
        inputs.append(img.cpu())  # 转移到cpu上,否则torch.stack内存不足
    inputs = torch.stack(inputs, dim=0)
    # inputs = torch.stack(inputs, dim=0).to(get_device())
    
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
    out_imgs = []
    out_bboxes = []
    out_labels = []
    out_metas = []
    device = get_device()
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值
    padding_value = [0.0081917211329, -0.004901960784, 0.0055655449953]  # 归一化后的padding值

    if mode == 'train':
        # for i in range(imgs.shape[0]):
        for img, bbox, label in zip(imgs, [bboxes], [labels]):
            p_imgs = []
            p_bboxes = []
            p_labels = []
            p_metas = []
            img = img.cpu()
            # patch
            info = dict()
            info['labels'] = np.array(torch.tensor(label, device='cpu', requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[2]
            info['height'] = img.shape[1]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, self.version))  # 这里将OBB转换为8点表示形式
            bbbox = info['ann']['bboxes']
            sizes = [patch_shape[0]]
            # gaps=[0]
            windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
            window_anns = get_window_obj(info, windows, iof_thr)
            patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
                                                    img,
                                                    no_padding=True,
                                                    # no_padding=False,
                                                    padding_value=padding_value)

            # 对每张大图分解成的子图集合中的每张子图遍历
            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    # 如果该patch中不含有效标签,将其跳过不输出,可在训练时使用

                    if patch_info['labels'] == [-1]:
                        # print('Patch does not contain box.\n')
                        continue
                obj = patch_info['ann']
                if min(obj['bboxes'].shape) == 0:  # 张量为空
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), 'oc')  # oc转化可以处理空张量
                else:
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), self.version)  # 转化回5参数
                p_bboxes.append(tmp_boxes.to(device))
                # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # 是否截断,box全部在win内部时为false
                ## 若box超出win范围则trunc为true
                p_labels.append(torch.tensor(patch_info['labels'], device=device))
                p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                                'y_start': torch.tensor(patch_info['y_start'], device=device),
                                'shape': patch_shape, 'trunc': torch.tensor(obj['trunc'], device=device),'img_shape': patch_shape, 'scale_factor': 1})

                patch = patchs[i]
                p_imgs.append(patch.to(device))

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

            #### change for sgdet
            # poly2obb(out_bboxes, self.version)
            return out_imgs, out_bboxes, out_labels, out_metas

    elif mode == 'test':
        p_imgs = []
        p_metas = []
        img = imgs.cpu().squeeze(0)
        # patch
        info = dict()
        info['labels'] = np.array(torch.tensor([], device='cpu'))
        info['ann'] = {'bboxes': {}}
        info['width'] = img.shape[2]
        info['height'] = img.shape[1]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                  no_padding=False,
                                                  padding_value=padding_value)

        # 对每张大图分解成的子图集合中的每张子图遍历
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas

    return out_imgs, out_bboxes, out_labels, out_metas



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
class S2ANetCrop(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 fam_head=None,
                 align_cfgs=None,
                 odm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetCrop, self).__init__()

        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            fam_head.update(train_cfg=train_cfg['fam_cfg'])
        fam_head.update(test_cfg=test_cfg)
        self.fam_head = build_head(fam_head)

        self.align_conv_type = align_cfgs['type']
        self.align_conv_size = align_cfgs['kernel_size']
        self.feat_channels = align_cfgs['channels']
        self.featmap_strides = align_cfgs['featmap_strides']

        if self.align_conv_type == 'AlignConv':
            self.align_conv = AlignConvModule(self.feat_channels,
                                              self.featmap_strides,
                                              self.align_conv_size)

        if train_cfg is not None:
            odm_head.update(train_cfg=train_cfg['odm_cfg'])
        odm_head.update(test_cfg=test_cfg)
        self.odm_head = build_head(odm_head)

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
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.fam_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_refine = self.odm_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value

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
    #     outs = self.fam_head(x)
    #     rois = self.fam_head.refine_bboxes(*outs)
    #     # rois: list(indexed by images) of list(indexed by levels)
    #     align_feat = self.align_conv(x, rois)
    #     outs = self.odm_head(align_feat)

    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)
    #     bbox_results = [
    #         rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results


    def simple_test(self, img, img_metas, rescale=False):
        # 为了使用裁剪小图策略推理标准模型
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
            p_imgs[i]=torch.stack(p_imgs[i], dim=0)
            patches=p_imgs[i]
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                with torch.no_grad():
                    fea_l_neck = self.extract_feat(patch)
                    outs_local = self.fam_head(fea_l_neck)
                    rois = self.fam_head.refine_bboxes(*outs_local)
                    align_feat = self.align_conv(fea_l_neck, rois)
                    outs = self.odm_head(align_feat)

                    bbox_inputs = outs + (patch_meta, self.test_cfg, rescale)
                    local_bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)

                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes, det_labels = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)

                j = j+p_bs
        
        bbox_list = [merge_results([local_bboxes_lists],iou_thr=0.4)]
        # print(' bbox_list', bbox_list)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
