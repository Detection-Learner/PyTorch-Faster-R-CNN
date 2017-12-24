import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from ...utils.config import cfg
from .proposal_layer import proposal_layer
from .anchor_target_layer import anchor_target_layer
from .proposal_target_layer import proposal_target_layer
from ..roi_pooling import RoIPool
from ..loss.warp_smooth_l1_loss.warp_smooth_l1_loss import WarpSmoothL1Loss


class RPN(nn.Module):
    def __init__(self, num_classes, in_channel, feat_stride=16):
        super(RPN, self).__init__()

        self.num_classes = num_classes
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = feat_stride
        self.anchor_scales_num = len(self.anchor_scales)
        self.anchor_ratios_num = len(self.anchor_ratios)

        self.rpn_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=512, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.score_conv = nn.Conv2d(
            in_channels=512, out_channels=self.anchor_scales_num * self.anchor_ratios_num * 2,
            kernel_size=1, stride=1, padding=0)
        self.bbox_conv = nn.Conv2d(
            in_channels=512, out_channels=self.anchor_scales_num * self.anchor_ratios_num * 4,
            kernel_size=1, stride=1, padding=0)

        self.proposal_layer = proposal_layer
        self.anchor_target_layer = anchor_target_layer
        self.proposal_target_layer = proposal_target_layer
        self.roi_pooling = RoIPool(7, 7, 1.0 / self.feat_stride)

        # loss
        self.warp_smooth_l1_loss = WarpSmoothL1Loss(
            sigma=3.0, size_average=False)
        self.cross_entropy = None
        self.loss_box = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, features, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        rpn_conv_feature = self.rpn_conv(features)
        rpn_conv_feature = self.relu(rpn_conv_feature)

        # rpn class score (neg/pos): default paras size is B * 18 * H * W
        rpn_cls_score = self.score_conv(rpn_conv_feature)
        rpn_cls_score_shape = rpn_cls_score.size()
        rpn_cls_score_reshape = rpn_cls_score.view(
            rpn_cls_score_shape[0], 2, -1, rpn_cls_score_shape[-1])
        # Fix to fit torch 0.3.0
        if float(torch.__version__[:3]) < 0.3:
            rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        else:
            rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = rpn_cls_prob.view(rpn_cls_score_shape)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv_feature)

        # get rois. NOTE: rois is a list witch length is batch size
        rois = self.proposal_layer(rpn_cls_prob.data.cpu().numpy(),
                                   rpn_bbox_pred.data.cpu().numpy(),
                                   im_info,
                                   'TRAIN' if self.training else 'TEST',
                                   self.feat_stride, self.anchor_scales, self.anchor_ratios)

        if self.training:
            assert gt_boxes is not None, 'Ground truth boxes is empty. Please check it.'
            rpn_data = self.anchor_target_layer(rpn_cls_score.data.cpu().numpy(), gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, self.feat_stride, self.anchor_scales, self.anchor_ratios)
            self.cross_entropy, self.loss_box = self.build_loss(
                rpn_cls_score, rpn_bbox_pred, rpn_data)

            rois_blob, labels_blob, bbox_targets_blob, bbox_inside_weights_blob, bbox_outside_weights_blob = self.proposal_target_layer(
                rois, gt_boxes, self.num_classes, gt_ishard, dontcare_areas)

            rois_out = rois_blob
            rpn_data_out = [rois_out, labels_blob, bbox_targets_blob,
                            bbox_inside_weights_blob, bbox_outside_weights_blob]
        else:
            rois_out = np.vstack(rois)
            rpn_data_out = [rois_out]

        if rpn_cls_score.is_cuda:
            if isinstance(rpn_bbox_pred.data, torch.cuda.FloatTensor):
                rois_out = torch.autograd.Variable(
                    torch.FloatTensor(rois_out)).cuda()
            elif isinstance(rpn_bbox_pred.data, torch.cuda.DoubleTensor):
                rois_out = torch.autograd.Variable(
                    torch.DoubleTensor(rois_out)).cuda()
        else:
            if isinstance(rpn_bbox_pred.data, torch.FloatTensor):
                rois_out = torch.autograd.Variable(
                    torch.FloatTensor(rois_out))
            elif isinstance(rpn_bbox_pred.data, torch.DoubleTensor):
                rois_out = torch.autograd.Variable(
                    torch.DoubleTensor(rois_out))

        pooled_features = self.roi_pooling(features, rois_out)

        return pooled_features, rpn_data_out

    def build_loss(self, rpn_cls_score, rpn_bbox_pred, rpn_data):
        rpn_label, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data
        # classification loss
        rpn_cls_score = rpn_cls_score.view(rpn_cls_score.size(
            0), 2, -1, rpn_cls_score.size(3)).permute(0, 2, 3, 1).contiguous().view(-1, 2)
        #  rpn_label process
        if rpn_cls_score.is_cuda:
            rpn_label = torch.autograd.Variable(
                torch.from_numpy(rpn_label.reshape(-1)).type(new_type=torch.LongTensor)).cuda()
        else:
            rpn_label = torch.autograd.Variable(
                torch.from_numpy(rpn_label.reshape(-1)).type(new_type=torch.LongTensor))
        # build loss
        rpn_cross_entropy = F.cross_entropy(
            rpn_cls_score, rpn_label, ignore_index=-1)

        # box loss
        if rpn_bbox_pred.is_cuda:
            if isinstance(rpn_bbox_pred.data, torch.cuda.FloatTensor):
                rpn_bbox_targets = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_targets)).cuda()
                rpn_bbox_inside_weights = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_inside_weights)).cuda()
                rpn_bbox_outside_weights = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_outside_weights)).cuda()
            elif isinstance(rpn_bbox_pred.data, torch.cuda.DoubleTensor):
                rpn_bbox_targets = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_targets)).cuda()
                rpn_bbox_inside_weights = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_inside_weights)).cuda()
                rpn_bbox_outside_weights = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_outside_weights)).cuda()
        else:
            if isinstance(rpn_bbox_pred.data, torch.FloatTensor):
                rpn_bbox_targets = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_targets))
                rpn_bbox_inside_weights = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_inside_weights))
                rpn_bbox_outside_weights = torch.autograd.Variable(
                    torch.FloatTensor(rpn_bbox_outside_weights))
            elif isinstance(rpn_bbox_pred.data, torch.DoubleTensor):
                rpn_bbox_targets = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_targets))
                rpn_bbox_inside_weights = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_inside_weights))
                rpn_bbox_outside_weights = torch.autograd.Variable(
                    torch.DoubleTensor(rpn_bbox_outside_weights))
        # build loss
        rpn_loss_box = self.warp_smooth_l1_loss(
            rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        return rpn_cross_entropy, rpn_loss_box

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box
