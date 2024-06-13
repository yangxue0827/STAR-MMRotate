import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, images_to_levels,
                        multi_apply, unmap)
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import select_single_mlvl
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core import (gaussian2bbox, multiclass_nms_rotated, obb2poly,
                           poly2obb)
from mmrotate.core import GaussianMixture as GMM

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


def levels_to_images(mlvl_tensor, flatten=False):
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    split_inside_flags = torch.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside


@HEADS.register_module()
class KLDRepPointsHead(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_bias='auto',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 center_init=True,
                 transform_method='rotrect',
                 use_reassign=False,
                 topk=6,
                 anti_factor=0.75,
                 version='v1',
                 covariance_type='diag',
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(KLDRepPointsHead, self).__init__(init_cfg)
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.use_reassign = use_reassign
        self.topk = topk
        self.anti_factor = anti_factor
        self.version = version
        self.covariance_type = covariance_type
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def points2gmm2rotrect(self, pts, y_first=True):
        if y_first:
            pts = pts.reshape(-1, self.num_points, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts = torch.cat([pts_dx, pts_dy],
                            dim=2).reshape(-1, 2 * self.num_points)

        pts = pts.reshape(-1, self.num_points, 2)
        if self.transform_method == 'rotrect':
            gmm = GMM(n_components=1)
            gmm.fit(pts)
            rotrect_pred = gaussian2bbox(gmm)
            return rotrect_pred
        else:
            raise NotImplementedError

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl, _ in enumerate(self.point_strides):
            pts_lvl = []
            for i_img, _ in enumerate(center_list):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             num_level_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             overlaps,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        num_level_anchors_inside = get_num_level_anchors_inside(
            num_level_proposals, inside_flags)

        # convert gt from obb to poly
        gt_bboxes = obb2poly(gt_bboxes, self.version)

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
            assign_result = assigner.assign(
                proposals, gt_bboxes, overlaps, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
            if self.train_cfg.refine.assigner.type not in ('ATSSKldAssigner',
                                                           'SASAssigner'):
                assign_result = assigner.assign(
                    proposals, gt_bboxes, overlaps, gt_bboxes_ignore,
                    None if self.sampling else gt_labels)
            else:
                assign_result = assigner.assign(
                    proposals, num_level_anchors_inside, gt_bboxes,
                    gt_bboxes_ignore, None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros(num_valid_proposals)
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * 4
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             num_level_proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)

        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, None)

    def get_cfa_targets(self,
                        proposals_list,
                        valid_flag_list,
                        gt_bboxes_list,
                        img_metas,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        stage='init',
                        label_channels=1,
                        unmap_outputs=True):

        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs
        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * 4
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             num_level_proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        pos_inds = []
        # pos_gt_index = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = (0 <= single_labels) & (
                single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]

        return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                all_proposal_weights, pos_inds, gt_inds)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, rbbox_gt_init, convex_weights_init,
                    rbbox_gt_refine, convex_weights_refine, stride,
                    num_total_samples_refine):
        normalize_term = self.point_base_scale * stride
        if self.use_reassign:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            return 0, loss_pts_init, 0
        else:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            # init points loss
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            # refine points loss
            rbbox_gt_refine = rbbox_gt_refine.reshape(-1, 8)
            pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
            convex_weights_refine = convex_weights_refine.reshape(-1)
            pos_ind_refine = (convex_weights_refine > 0).nonzero(
                as_tuple=False).reshape(-1)
            pts_pred_refine_norm = pts_pred_refine[pos_ind_refine]
            rbbox_gt_refine_norm = rbbox_gt_refine[pos_ind_refine]
            convex_weights_pos_refine = convex_weights_refine[pos_ind_refine]
            loss_pts_refine = self.loss_bbox_refine(
                pts_pred_refine_norm / normalize_term,
                rbbox_gt_refine_norm / normalize_term,
                convex_weights_pos_refine)
            # classification loss
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples_refine)
            return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        device = cls_scores[0].device

        # target for initial stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.use_reassign:  # get num_proposal_each_lvl and lvl_num
            num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                        for featmap in cls_scores]
            num_level = len(featmap_sizes)
            assert num_level == len(pts_coordinate_preds_init)
        if self.train_cfg.init.assigner['type'] in [
                'ConvexAssigner', 'ConvexGTAssigner'
        ]:
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, rbbox_gt_list_init, candidate_list_init, convex_weights_list_init,
         num_total_pos_init, num_total_neg_init, _) = cls_reg_targets_init
        # target for refinement stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)
        if self.use_reassign:
            cls_reg_targets_refine = self.get_cfa_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine, _,
             convex_weights_list_refine, pos_inds_list_refine,
             pos_gt_index_list_refine) = cls_reg_targets_refine
            cls_scores = levels_to_images(cls_scores)
            cls_scores = [
                item.reshape(-1, self.cls_out_channels) for item in cls_scores
            ]
            pts_coordinate_preds_init_cfa = levels_to_images(
                pts_coordinate_preds_init, flatten=True)
            pts_coordinate_preds_init_cfa = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_init_cfa
            ]
            pts_coordinate_preds_refine = levels_to_images(
                pts_coordinate_preds_refine, flatten=True)
            pts_coordinate_preds_refine = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_refine
            ]
            with torch.no_grad():
                pos_losses_list, = multi_apply(
                    self.get_pos_loss, cls_scores,
                    pts_coordinate_preds_init_cfa, labels_list,
                    rbbox_gt_list_refine, label_weights_list,
                    convex_weights_list_refine, pos_inds_list_refine)
                labels_list, label_weights_list, convex_weights_list_refine, \
                    num_pos, pos_normalize_term = multi_apply(
                        self.cfa_reassign,
                        pos_losses_list,
                        labels_list,
                        label_weights_list,
                        pts_coordinate_preds_init_cfa,
                        convex_weights_list_refine,
                        gt_bboxes,
                        pos_inds_list_refine,
                        pos_gt_index_list_refine,
                        num_proposals_each_level=num_proposals_each_level,
                        num_level=num_level
                    )
                num_pos = sum(num_pos)
            # convert all tensor list to a flatten tensor
            cls_scores = torch.cat(cls_scores, 0).view(-1,
                                                       cls_scores[0].size(-1))
            pts_preds_refine = torch.cat(pts_coordinate_preds_refine, 0).view(
                -1, pts_coordinate_preds_refine[0].size(-1))
            labels = torch.cat(labels_list, 0).view(-1)
            labels_weight = torch.cat(label_weights_list, 0).view(-1)
            rbbox_gt_refine = torch.cat(rbbox_gt_list_refine, 0).view(
                -1, rbbox_gt_list_refine[0].size(-1))
            convex_weights_refine = torch.cat(convex_weights_list_refine,
                                              0).view(-1)
            pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
            pos_inds_flatten = ((0 <= labels) &
                                (labels < self.num_classes)).nonzero(
                                    as_tuple=False).reshape(-1)
            if num_pos:
                assert len(pos_normalize_term) == len(pos_inds_flatten)
                losses_cls = self.loss_cls(
                    cls_scores, labels, labels_weight, avg_factor=num_pos)
                pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
                pos_rbbox_gt_refine = rbbox_gt_refine[pos_inds_flatten]
                pos_convex_weights_refine = convex_weights_refine[
                    pos_inds_flatten]
                losses_pts_refine = self.loss_bbox_refine(
                    pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                    pos_rbbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                    pos_convex_weights_refine)
            else:
                losses_cls = cls_scores.sum()
                losses_pts_refine = pts_preds_refine.sum() * 0

            if losses_cls.numel() == 1:
                losses_cls = losses_cls.unsqueeze(0)
                losses_pts_refine = losses_pts_refine.unsqueeze(0)
            None_list = [None] * num_level
            _, losses_pts_init, _ = multi_apply(
                self.loss_single,
                None_list,
                pts_coordinate_preds_init,
                None_list,
                None_list,
                None_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                None_list,
                None_list,
                self.point_strides,
                num_total_samples_refine=None,
            )
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all
        else:
            cls_reg_targets_refine = self.get_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine,
             candidate_list_refine, convex_weights_list_refine,
             num_total_pos_refine, num_total_neg_refine,
             _) = cls_reg_targets_refine
            num_total_samples_refine = (
                num_total_pos_refine + num_total_neg_refine
                if self.sampling else num_total_pos_refine)

            losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
                self.loss_single,
                cls_scores,
                pts_coordinate_preds_init,
                pts_coordinate_preds_refine,
                labels_list,
                label_weights_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                rbbox_gt_list_refine,
                convex_weights_list_refine,
                self.point_strides,
                num_total_samples_refine=num_total_samples_refine)
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all

    def get_pos_loss(self, cls_score, pts_pred, label, bbox_gt, label_weight,
                     convex_weight, pos_inds):
        if len(pos_inds) == 0:
            return label.new_zeros(1)
        pos_scores = cls_score[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_cls
        return pos_loss,

    def cfa_reassign(self,
                     pos_losses,
                     label,
                     label_weight,
                     pts_pred_init,
                     convex_weight,
                     gt_bbox,
                     pos_inds,
                     pos_gt_inds,
                     num_proposals_each_level=None,
                     num_level=None):
        if len(pos_inds) == 0 or len(pos_gt_inds) == 0:
            return label, label_weight, convex_weight, 0, torch.tensor(
                []).type_as(convex_weight)
        num_gt = pos_gt_inds.max() + 1
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        pos_inds_after_cfa = [label.new_tensor([])]
        ignore_inds_after_cfa = [label.new_tensor([])]

        for gt_ind in range(num_gt):
            pos_inds_cfa = []
            pos_loss_cfa = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                if sum(pos_gt_inds) == 0:
                    level_gt_mask = level_mask
                else:
                    level_gt_mask = level_mask & gt_mask
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_cfa.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_cfa.append(value)
            pos_inds_cfa = torch.cat(pos_inds_cfa)
            pos_loss_cfa = torch.cat(pos_loss_cfa)

            if len(pos_inds_cfa) < 2:
                continue

            if len(pos_inds_cfa) == 2:
                if pos_loss_cfa[0] == pos_loss_cfa[1]:
                    continue

            device = pos_loss_cfa.device
            pos_loss_cfa, sort_inds = pos_loss_cfa.sort()
            pos_inds_cfa = pos_inds_cfa[sort_inds]
            pos_loss_cfa = pos_loss_cfa.reshape(-1, 1).cpu().numpy()
            min_loss, max_loss = pos_loss_cfa.min(), pos_loss_cfa.max()

            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)

            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])

            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type)
            gmm.fit(pos_loss_cfa)
            gmm_assignment = gmm.predict(pos_loss_cfa)
            scores = gmm.score_samples(pos_loss_cfa)
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)

            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_cfa)
            pos_inds_after_cfa.append(pos_inds_temp)
            ignore_inds_after_cfa.append(ignore_inds_temp)

        pos_inds_after_paa = torch.cat(pos_inds_after_cfa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_cfa)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        convex_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)

        pos_level_mask_after_cfa = []
        for i in range(num_level):
            mask = (pos_inds_after_paa >= inds_level_interval[i]) & (
                pos_inds_after_paa < inds_level_interval[i + 1])
            pos_level_mask_after_cfa.append(mask)
        pos_level_mask_after_cfa = torch.stack(pos_level_mask_after_cfa,
                                               0).type_as(label)
        pos_normalize_term = pos_level_mask_after_cfa * (
            self.point_base_scale *
            torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(convex_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_paa)

        return label, label_weight, convex_weight, num_pos, pos_normalize_term

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper instead of (b).
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero(as_tuple=False).numel():
            _, pos_thr_ind = scores[fgs].topk(1)
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            ignore_inds_temp = pos_inds_gmm.new_tensor([])

        return pos_inds_temp, ignore_inds_temp

    @force_fp32(apply_to=('cls_scores', 'pts_preds_init', 'pts_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        assert len(cls_scores) == len(pts_preds_refine)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].device,
            device=cls_scores[0].device)

        result_list = []

        for img_id, _ in enumerate(img_metas):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            point_pred_list = select_single_mlvl(pts_preds_refine, img_id)

            results = self._get_bboxes_single(cls_score_list, point_pred_list,
                                              mlvl_priors, img_meta, cfg,
                                              rescale, with_nms, **kwargs)
            result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           point_pred_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into rbbox predictions."""

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(point_pred_list)
        scale_factor = img_meta['scale_factor']

        mlvl_bboxes = []
        mlvl_scores = []
        for level_idx, (cls_score, points_pred, points) in enumerate(
                zip(cls_score_list, point_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            points_pred = points_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            poly_pred = self.points2gmm2rotrect(points_pred, y_first=True)
            bbox_pos_center = points[:, :2].repeat(1, 4)
            polys = poly_pred * self.point_strides[level_idx] + bbox_pos_center
            bboxes = poly2obb(polys, self.version)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes[..., :4].new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)

            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores