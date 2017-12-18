from proposal_target_layer import *
import numpy as np
import random

batch_size = 3
H = 3
W = 3
A = 3
rpn_rois = np.random.rand(batch_size * H * W * A * 5)
rpn_rois = rpn_rois.reshape((batch_size, H*W*A, 5))
rois = []
for i in range(batch_size):
    rpn_rois[i,:,0] = i
    rpn_rois[i,:,3] = rpn_rois[i,:,1] + random.randint(0,100)
    rpn_rois[i,:,4] = rpn_rois[i,:,2] + random.randint(0, 100)
    rois.append(rpn_rois[i])
gt_boxes = []
for i in range(1, 5):
    gt_box = np.random.rand(i * 5)
    gt_box = gt_box.reshape(i, 5)
    gt_box[:, -1] = random.randint(0, 9)
    gt_box[:, 2] = gt_box[:, 0] + random.randint(0, 100)
    gt_box[:, 3] = gt_box[:, 1] + random.randint(0, 100)
    gt_boxes.append(gt_box)
rois_list, labels_list, bbox_targets_list, bbox_inside_weights_list, bbox_outside_weights_list = proposal_target_layer(rpn_rois, gt_boxes, 10, None, None)
print rois_list
print len(rois_list), rois_list.shape
print len(labels_list), labels_list.shape
print len(bbox_targets_list), bbox_targets_list.shape
print len(bbox_inside_weights_list), bbox_inside_weights_list.shape
print len(bbox_outside_weights_list), bbox_outside_weights_list.shape
