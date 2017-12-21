# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Modified by Yuanshun Cui

import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ...utils.bbox import bbox_overlaps, bbox_intersections
from ...utils.config import cfg
from ...utils.transform_bbox import bbox_transform


def gen_shift(height, width, _feat_stride):
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.vstack((shift_x.ravel(), shift_y.ravel(),
                       shift_x.ravel(), shift_y.ravel())).transpose()
    return shift


def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, feat_stride=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: for pytorch (B, Ax2, H, W) bg/fg scores of previous conv layer
    gt_boxes: [B, (G, 5)] vstack of [x1, y1, x2, y2, class]
    gt_ishard: [B, (G, 1)], 1 or 0 indicates difficult or not
    dontcare_areas: [B, (D, 4)], some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : [B, (HxWxA, 1)], for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    # Generate anchors
    _anchors = generate_anchors(
        base_size=cfg.ANCHOR_BASE_SIZE, ratios=anchor_ratios, scales=np.array(anchor_scales))
    # Get anchors numbers
    _num_anchors = _anchors.shape[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = cfg.ALLOWED_BORDER

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    height, width = rpn_cls_score.shape[2:]

    # Get shifts
    shifts = gen_shift(height=height, width=width, _feat_stride=feat_stride)

    # Enumerate all shifted anchors:
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    B = len(gt_boxes)
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image (with a little pixels out of edges)
    inds_insides = [np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[i, 1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[i, 0] + _allowed_border)  # height
    )[0] for i in range(B)]

    # keep only inside anchors
    anchors = [all_anchors[inds_inside, :]
               for inds_inside in inds_insides]  # [B, (A, 4)]

    # NOTE: Attention the batch size follow
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = [np.ones(len(inds_inside), dtype=np.float32) * (-1)
              for inds_inside in inds_insides]  # [B, (A)]
    # labels.fill(-1)  # (B, A)

    # overlaps between the anchors and the gt boxes
    # overlaps [B, (ex, gt)], shape is [B, (A, G)]
    overlaps = [bbox_overlaps(np.ascontiguousarray(anchor, dtype=np.float),
                              np.ascontiguousarray(gt_box, dtype=np.float)) for anchor, gt_box in zip(anchors, gt_boxes)]
    argmax_overlaps = [overlap.argmax(axis=1)
                       for overlap in overlaps]  # [B, (A)]
    max_overlaps = [overlap[np.arange(len(inds_inside)), argmax_overlap]
                    for overlap, inds_inside, argmax_overlap in zip(overlaps, inds_insides, argmax_overlaps)]
    gt_argmax_overlaps = [overlap.argmax(axis=0)
                          for overlap in overlaps]  # [B, (G)]
    gt_max_overlaps = [overlap[gt_argmax_overlap, np.arange(
        overlap.shape[1])] for overlap, gt_argmax_overlap in zip(overlaps, gt_argmax_overlaps)]
    gt_argmax_overlaps = [np.where(overlap == gt_max_overlap)[
        0] for overlap, gt_max_overlap in zip(overlaps, gt_max_overlaps)]
    # -----------------------------------------------------------------------------------------------
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        for i in range(B):
            labels[i][max_overlaps[i] < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    for i in range(B):
        labels[i][gt_argmax_overlaps[i]] = 1
    # fg label: above threshold IOU
    for i in range(B):
        labels[i][max_overlaps[i] >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        for i in range(B):
            labels[i][max_overlaps[i] < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # preclude dontcare areas
    if dontcare_areas is not None:  # and dontcare_areas[0].shape[0] > 0:
        assert len(dontcare_areas) == len(
            gt_boxes), 'The number of dontcare_areas is not equal image numbers.'
        # intersec shape is [B, (D x A)]
        intersecs = [bbox_intersections(
            np.ascontiguousarray(dontcare_area, dtype=np.float),  # D x 4
            np.ascontiguousarray(anchor, dtype=np.float)  # A x 4
        ) if dontcare_area.shape[0] > 0 else None for anchor, dontcare_area in zip(anchor, dontcare_areas)]
        intersecs_ = [intersec.sum(axis=0)
                      if intersec is not None else None for intersec in intersecs]  # [B, (A, 1)]
        for i in range(len(intersecs_)):
            if intersecs_[i] is not None:
                labels[i][intersecs_[i] >
                          cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and all([gt_ishard[i].shape[0] > 0 for i in range(len(gt_ishard))]):
        assert all([gt_ishard_.shape[0] == gt_box.shape[0]
                    for gt_ishard_, gt_box in zip(gt_ishard, gt_boxes)]), 'gt_ishard shape is not equal gt_boxes'' shape'
        gt_ishard = [gt_ishard_.astype(int) for gt_ishard_ in gt_ishard]
        gt_hardboxes = [gt_box[gt_ishard_ == 1, :]
                        for gt_ishard_, gt_box in zip(gt_ishard, gt_boxes)]  # [B, (N, 5)]
        for i in range(B):  # i in range(B)
            if gt_hardboxes[i].shape[0] > 0:
                # H x A
                hard_overlaps = bbox_overlaps(
                    np.ascontiguousarray(
                        gt_hardboxes[i], dtype=np.float),  # H x 4
                    np.ascontiguousarray(anchors[i], dtype=np.float))  # A x 4
                hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
                labels[i, hard_max_overlaps >=
                       cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
                max_intersec_label_inds = hard_overlaps.argmax(axis=1)  # H x 1
                labels[i][max_intersec_label_inds] = -1  #

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = [np.where(label == 1)[0] for label in labels]  # [B, X]
    for i in range(B):
        if len(fg_inds[i]) > num_fg:
            disable_inds = npr.choice(
                fg_inds[i], size=(len(fg_inds[i]) - num_fg), replace=False)
            labels[i][disable_inds] = -1

    # subsample negative labels if we have too many
    for i in range(B):
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels[i] == 1)
        bg_inds = np.where(labels[i] == 0)[0]
        # ----------------------------------------------------------------------------------------
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[i][disable_inds] = -1

    # Compute bbox_targets, which is pre rpn_bbox_targets
    bbox_targets = [_compute_targets(anchor, gt_box[argmax_overlap, :])
                    for anchor, gt_box, argmax_overlap in zip(anchors, gt_boxes, argmax_overlaps)]
    # Compute the inside & outside weights
    bbox_inside_weights = [np.zeros(
        (len(inds_inside), 4), dtype=np.float32) for inds_inside in inds_insides]  # (B, A, 4)
    for i in range(B):
        bbox_inside_weights[i][labels[i] == 1, :] = np.array(
            cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = [np.zeros(
        (len(inds_inside), 4), dtype=np.float32) for inds_inside in inds_insides]
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = [np.sum(label >= 0) + 1 for label in labels]
        positive_weights = [
            np.ones((1, 4)) * 1.0 / num_example for num_example in num_examples]
        negative_weights = [
            np.ones((1, 4)) * 1.0 / num_example for num_example in num_examples]
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = [(cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                             (np.sum(label == 1)) + 1) for label in labels]
        negative_weights = [((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                             (np.sum(label == 0)) + 1) for label in labels]
    for i in range(B):
        bbox_outside_weights[i][labels[i] == 1, :] = positive_weights[i]
        bbox_outside_weights[i][labels[i] == 0, :] = negative_weights[i]

    # map up to original set of anchors
    # NOTE: there labels has been transformed to a list whose length is batch size.
    #      So as bbox_inside_weights & bbox_outside_weights
    labels = [_unmap(labels[i], total_anchors, inds_insides[i], fill=-1)
              for i in range(B)]
    bbox_targets = [_unmap(bbox_target, total_anchors, inds_inside, fill=0)
                    for inds_inside, bbox_target in zip(inds_insides, bbox_targets)]
    bbox_inside_weights = [_unmap(bbox_inside_weight, total_anchors, inds_inside, fill=0)
                           for inds_inside, bbox_inside_weight in zip(inds_insides, bbox_inside_weights)]
    bbox_outside_weights = [_unmap(bbox_outside_weight, total_anchors, inds_inside, fill=0)
                            for inds_inside, bbox_outside_weight in zip(inds_insides, bbox_outside_weights)]

    # Processing labels
    labels = np.array(labels).reshape((-1, height, width, A))  # (B, H, W, A)
    labels = labels.transpose(0, 3, 1, 2)  # (B, A, H, W)
    rpn_labels = labels.reshape(
        (-1, 1, A * height, width)).transpose(0, 2, 3, 1)  # (B, A * H, W, 1)

    # bbox_targets
    bbox_targets = np.array(bbox_targets).reshape(
        (-1, height, width, A * 4)).transpose(0, 3, 1, 2)  # (B, A * 4, H, W)

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = np.array(bbox_inside_weights).reshape(
        (-1, height, width, A * 4)).transpose(0, 3, 1, 2)  # (B, A * 4, H, W)
    # assert bbox_inside_weights.shape[2] == height
    # assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = np.array(bbox_outside_weights).reshape(
        (-1, height, width, A * 4)).transpose(0, 3, 1, 2)  # (B, A * 4, H, W)
    # assert bbox_outside_weights.shape[2] == height
    # assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
