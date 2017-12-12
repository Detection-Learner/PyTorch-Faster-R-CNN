#--------------------------------------
# PyTorch Faster R-CNN
# proposal_target_layer.py
# Written by Hongyu Pan
#--------------------------------------
import numpy as np
from utils.config import cfg
from utils.transform_bbox import bbox_transform
from utils.bbox import bbox_overlaps

def proposal_target_layer(rpn_rois, gt_boxes, num_classes, gt_ishard, dontcare_areas):
    """
    Assign object detection proposals (rpn_rois) to ground-truth targets (gt_boxes). Produces proposal classification labels and bounding-box regression targets.
    ----------
    Parameters
    ----------
    rpn_rois: (B, N, 5) [img_id, x1, y1, x2, y2] list, ndarray in list
    gt_boxes: (B, G, 5) [x1, y1, x2, y2, class] list, ndarray in list
    gt_ishard: (B, G, 1) {0|1} 1 indicates hard
    dontcare_areas: (B, D, 4) [x1, y1, x2, y2]
    num_classes: int
    ----------
    Returns
    ----------
    rois_list: (B, fg_inds_i + bg_inds_i, 5) [img_id, x1, y1, x2, y2] list, ndarray in list
    labels_list: (B, fg_inds_i + bg_inds_i) {0, 1, num_classes - 1} list, ndarray in list
    bbox_targets_list: (B, fg_inds_i + bg_inds_i, num_classes x 4) [dx1, dy1, dx2, dy2] list, ndarray in list
    bbox_inside_weights_list: (B, fg_inds_i + bg_inds_i, num_classes x 4) 0, 1 masks for computing loss
    bbox_outside_weights_list: (B, fg_inds_i + bg_inds_i, num_classes x 4) 0, 1 masks for computing loss
    ----------
    """

    rois_list = []
    labels_list = []
    bbox_targets_list = []
    bbox_inside_weights_list = []
    bbox_outside_weights_list = []

    all_rois = rpn_rois
    batch_size = len(all_rois)

    for i in range(batch_size):
        # assign the img id to gt_boxes
        img_id = np.empty((gt_boxes[i].shape[0], 1), dtype=gt_boxes[i].dtype)
        img_id.fill(i)

        # include ground-truth boxes in the set of candidate rois
        tmp_rois = all_rois[i]
        tmp_rois = np.vstack(
                (tmp_rois, np.hstack((img_id, gt_boxes[i][:,:-1])))
            )

        rois_per_image = cfg.TRAIN.BATCH_SIZE
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification lables and bounding box regression
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
                tmp_rois, gt_boxes[i], fg_rois_per_image, rois_per_image, num_classes)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

        labels_list.append(labels)
        rois_list.append(rois)
        bbox_targets_list.append(bbox_targets)
        bbox_inside_weights_list.append(bbox_inside_weights)
        bbox_outside_weights_list.append(bbox_outside_weights)

    return rois_list, labels_list, bbox_targets_list, bbox_inside_weights_list, bbox_outside_weights_list

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """
    bounding-box regression targets (bbox_target_data) are stored in a compact form N x (class, dx, dy, tw, th)

    This function expands those targets into the 4-of-4*K representation used by the networks (i.e. only one class has non-zero targets)

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    # background doesn't generate loss. So all of them are zeros.
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start: end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start: end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """
    Compute bounding-box regression targets for an image.
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    # [label, dx, dy, dw, dh]
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    Generate a random sample of ROIs comprising foreground and background examples.
    ------------
    Parameters:
    ------------
    all_rois: (N + G, 5) [img_id, x1, y1, x2, y2] int
    gt_boxes: (G, 5) [x1, y1, x2, y2, class] int
    fg_rois_per_image: int
    rois_per_image: int
    num_classes: int
    ------------
    Returns:
    ------------
    fg_inds and bg_inds vary from images to images.

    labels: (fg_inds + bg_inds), int
    rois: (fg_inds + bg_inds, 5) [img_id, x1, y1, x2, y2] int
    bbox_target: (fg_inds + bg_inds, 4 * num_classes) [dx, dy, dw, dh] int
    bbox_inside_weights: (fg_inds + bg_inds, 4 * num_classes) float32
    """

    # overlaps: (rois x gt_boxes), iou
    overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float32),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float32))
    # find the gt_boxes id, which has the max iou for each rois
    gt_assignment = overlaps.argmax(axis=1)
    # find the max iou for each rois
    max_overlaps = overlaps.max(axis=1)
    # get the max iou gt_boxes' label for each rois
    labels = gt_boxes[gt_assignment, -1]
    print overlaps

    # Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground ROIs
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    if fg_inds.size > 0:
        # randomly select fg_rois_per_image rois.
        # np.random.choice(), replace = False means there is no repeat objects in choiced.
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_image, replace=False)
    # Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI]
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Compute number of background ROIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # This indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background ROIs to 0
    labels[fg_rois_per_image: ] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
