import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .rpn_layer import RPN

from ...utils import config as cfg


class FPN(nn.Module):
    def __init__(self, num_classes, in_channels=[512, ], feat_strides=[16, ], out_channel=256, is_loss_ave=True):
        super(FPN, self).__init__()
        assert len(in_channels) == len(
            feat_strides), 'The length of features in_channels not equal feat_stride length !'

        self.feat_strides = feat_strides
        self.is_loss_ave = is_loss_ave

        self.feature_new = torch.nn.ModuleList([nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1) for in_channel in in_channels])
        # NOTE: To use deconv to get upsample.
        # self.feature_process = torch.nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(
        #     4, 4), stride=(2, 2), padding=(1, 1), output_padding=0, groups=out_channel, bias=False)] * (len(in_channels) - 1))
        self.RPN_Units = torch.nn.ModuleList([RPN(num_classes=num_classes, in_channel=out_channel,
                                                  feat_stride=feat_stride) for feat_stride in feat_strides])

        # loss
        self.losses = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                pass

    def forward(self, features, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        assert len(features) == len(
            self.feat_strides), 'The numbers of features not equal feat_stride length !'

        feature_L = len(features)

        newP = [None] * feature_L
        for i, f_new in enumerate(self.feature_new):
            newP[i] = f_new(features[i])
        # NOTE: To use deconv to get upsample. But maybe it is not equal. So it need a crop
        # for i, f_p in enumerate(self.feature_process):
            # newP[i + 1] = self.crop_sum(f_p(newP[i]), newP[i + 1])
        for i in range(feature_L - 1):
            _, _, h, w = newP[i + 1].size()
            newP[i + 1] = F.upsample(newP[i], (h, w),
                                     None, 'bilinear') + newP[i + 1]

        roi_out = [None] * feature_L
        rpn_out = [None] * feature_L
        for i, rpn in enumerate(self.RPN_Units):
            roi_out[i], rpn_out[i] = rpn(newP[i], im_info, gt_boxes=gt_boxes,
                                         gt_ishard=gt_ishard, dontcare_areas=dontcare_areas)
        # roi_out = torch.stack(roi_out, dim=0)
        roi_out = torch.cat(roi_out, dim=0)

        if self.training:
            rois_out = [None] * feature_L
            labels_blob = [None] * feature_L
            bbox_targets_blob = [None] * feature_L
            bbox_inside_weights_blob = [None] * feature_L
            bbox_outside_weights_blob = [None] * feature_L
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
            rois_out = [None] * feature_L
            for i in range(len(rpn_out)):
                rois_out[i] = rpn_out[i][0]
            rois_out = np.vstack(rois_out)
            rpn_data_out = [rois_out]

        return roi_out, rpn_data_out

    # NOTE: To use deconv to get upsample. But maybe it is not equal. So it need a crop
    # def crop_sum(self, A, B):
    #     _, _, ha, wa = A.size()
    #     _, _, hb, wb = B.size()
    #     assert (ha >= hb and wa >= wb), ValueError(
    #         'Size must be ha >= hb and wa >= wb, but not.')
    #     return A[:, :, :hb, :wb] + B

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
