import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .rpn_layer import RPN

from ...utils import config as cfg


class FPN(nn.Module):
    def __init__(self, num_classes, in_channels=[512, ], feat_strides=[16, ], is_loss_ave=True):
        super(FPN, self).__init__()
        assert len(in_channels) == len(
            feat_strides), 'The length of features in_channels not equal feat_stride length !'

        self.feat_strides = feat_strides
        self.is_loss_ave = is_loss_ave

        self.RPN_Units = torch.nn.ModuleList([RPN(num_classes=num_classes, in_channel=in_channel,
                                                  feat_stride=feat_stride) for in_channel, feat_stride in zip(in_channels, feat_strides)])

        # loss
        self.losses = None

    def forward(self, features, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        assert len(features) == len(
            self.feat_strides), 'The numbers of features not equal feat_stride length !'

        roi_out = [None] * len(features)
        rpn_out = [None] * len(features)
        for i, rpn in enumerate(self.RPN_Units):
            roi_out[i], rpn_out[i] = rpn(features[i], im_info, gt_boxes=gt_boxes,
                                         gt_ishard=gt_ishard, dontcare_areas=dontcare_areas)
        # roi_out = torch.stack(roi_out, dim=0)
        roi_out = torch.cat(roi_out, dim=0)

        if self.training:
            rois_out = [None] * len(features)
            labels_blob = [None] * len(features)
            bbox_targets_blob = [None] * len(features)
            bbox_inside_weights_blob = [None] * len(features)
            bbox_outside_weights_blob = [None] * len(features)
            for i in range(len(rpn_out)):
                rois_out[i], labels_blob[i], bbox_targets_blob[i], bbox_inside_weights_blob[i], bbox_outside_weights_blob[i] = rpn_out[i]
            rois_out = np.vstack(rois_out)
            labels_blob = np.hstack(labels_blob)
            bbox_targets_blob = np.vstack(bbox_targets_blob)
            bbox_inside_weights_blob = np.vstack(bbox_inside_weights_blob)
            bbox_outside_weights_blob = np.vstack(bbox_outside_weights_blob)

            rpn_data_out = [rois_out, labels_blob, bbox_targets_blob,
                            bbox_inside_weights_blob, bbox_outside_weights_blob]
            self.losses = self.build_loss()
        else:
            rois_out = [None] * len(features)
            for i in range(len(rpn_out)):
                rois_out[i] = rpn_out[i][0]
            rois_out = np.vstack(rois_out)
            rpn_data_out = [rois_out]

        return roi_out, rpn_data_out

    def build_loss(self):
        for i, rpn in enumerate(self.RPN_Units):
            if i == 0:
                loss = rpn.loss
            else:
                loss += rpn.loss
        if self.is_loss_ave:
            loss = loss / len(self.feat_strides)
        return loss

    @property
    def loss(self):
        return self.losses
