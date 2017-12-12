# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Modified by Yuanshun Cui

import numpy as np

from ...utils.config import cfg
from ...utils.nms_wrapper import nms
from .generate_anchors import generate_anchors
from ...utils.transform_bbox import bbox_transform_inv, clip_boxes


def gen_shift(height, width, _feat_stride):
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.vstack((shift_x.ravel(), shift_y.ravel(),
                       shift_x.ravel(), shift_y.ravel())).transpose()
    return shift


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, feat_stride=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2]):
    """
    Parameters
    ----------
    rpn_cls_prob:  (B, Ax2, H, W) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (B, Ax4, H, W), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    anchor_ratios: the aspect ratios to the anchor windows (default is [0.5, 1, 2])
    ----------
    Returns
    ----------
    rpn_rois : [B, (H x W x A, 5)] e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    """

    # Generate anchors
    _anchors = generate_anchors(
        base_size=feat_stride, ratios=anchor_ratios, scales=np.array(anchor_scales))
    # Get anchors numbers
    _num_anchors = _anchors.shape[0]

    # Get config parameters
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    min_size = cfg[cfg_key].RPN_MIN_SIZE

    # assert rpn_cls_prob_reshape.shape[0] == 1, \
    #    'Only single item batches are supported

    # Get shape of features
    heights, widths = rpn_cls_prob.shape[-2:]

    # Get postive scores. Size: B x (h*w*A) x 1
    # scores are (B, A, H, W) format
    # transpose to (B, H, W, A)
    # reshape to (B, H * W * A, 1) where rows are ordered by (h, w, a)
    scores = rpn_cls_prob.transpose(0, 2, 3, 1).reshape(
        [-1, heights, widths, _num_anchors, 2])[:, :, :, :, 1].reshape([-1, heights * widths * _num_anchors, 1])
    bbox_deltas = rpn_bbox_pred

    # Get shifts
    shifts = gen_shift(height=heights, width=widths, _feat_stride=feat_stride)

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # K = heights * widths
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (B, 4 * A, H, W) format
    # transpose to (B, H, W, 4 * A)
    # reshape to (B, H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose(
        (0, 2, 3, 1)).reshape((-1, heights * widths * A, 4))

    # Convert anchors into proposals via bbox transformations
    # NOTICE: Attention the batch size B
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # Clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[0, :2])

    # remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keeps = [_filter_boxes(proposals[i, :, :], min_size * im_info[i, 2])
             for i in range(len(im_info))]
    proposals_list = [proposals[i, keeps[i], :] for i in range(len(keeps))]
    scores_list = [scores[i, keeps[i], :] for i in range(len(keeps))]

    # # remove irregular boxes, too fat too tall
    # keeps = [_filter_irregular_boxes(proposals[i, :, :], min_size * im_info[i, 2])
    #          for i in range(len(im_info))]
    # proposals_list = [proposals[i, keeps[i], :] for i in range(len(keeps))]
    # scores_list = [scores[i, keeps[i], :] for i in range(len(keeps))]

    # Sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN (e.g. 6000)
    orders = [score.ravel().argsort()[::-1] for score in scores_list]
    if pre_nms_topN > 0:
        orders = [order[:pre_nms_topN] for order in orders]
    # else:
    #     raise NotImplementedError(
    #         'You must set up the para of RPN_PRE_NMS_TOP_N')
    proposals = [proposal[order, :]
                 for proposal, order in zip(proposals_list, orders)]
    # proposals = np.array(proposals)  # (B, N, 4)
    scores = [score[order] for score, order in zip(scores_list, orders)]
    # scores = np.array(scores)  # (B, N, 1)

    # apply nms (e.g. threshold = 0.7)
    # take after_nms_topN (e.g. 300)
    # return the top proposals (-> RoIs top)
    keeps = [nms(np.hstack((proposal.astype(np.float32), score.astype(np.float32))), nms_thresh)
             for proposal, score in zip(proposals, scores)]
    if post_nms_topN > 0:
        keeps = [keep[:post_nms_topN] for keep in keeps]
    proposals = [proposal[keep, :] for proposal, keep in zip(proposals, keeps)]
    scores = [score[keep] for score, keep in zip(scores, keeps)]  # [B, (N, 1)]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0,...,0,1,...,1,.....
    # batch_inds = np.array([[i] * proposals.shape[1]
    #                        for i in range(proposals.shape[0])], dtype=np.float32).reshape(-1, 1)
    batch_inds = [np.array([i] * proposals[i].shape[0],
                           dtype=np.float32).reshape(-1, 1) for i in range(len(proposals))]  # [B, (N, 1)]
    blobs = [np.hstack((batch_ind, proposal.astype(
        np.float32, copy=False).reshape(-1, 4))) for batch_ind, proposal in zip(batch_inds, proposals)]  # [B, (N, 5)]
    return blobs


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1  # n
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


# if __name__ == '__main__':
#     rpn_cls_prob = np.random.rand(15, 9 * 2, 64, 86)
#     rpn_bbox_pred = np.random.rand(15, 9 * 4, 64, 86)
#     im_info = [480, 1200, 0.5] * 15
#     blobs = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, 'TRAIN',
#                            feat_stride=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2])
