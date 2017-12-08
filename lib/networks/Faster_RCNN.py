#--------------------------------------------
# Faster R-CNN
# Written by Hongyu Pan
#--------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.Variable as Variable

from layers.rpn import RPN
from layers.rpn import proposal_target_layer
from utils import util

import basic_network
import RCNN

class FasterRCNN(nn.Module):

    def __init__(self):
        super(FasterRCNN, self).__init__()

        #TODO
        # initilize parameters

        # network
        self.basic_network = basic_network()
        self.rpn = RPN()
        self.rcnn = RCNN()
        self.cls_fc = nn.Linear(self.rcnn.out_dim, self.param.num_classes)
        self.bbox_fc = nn.Linear(self.rcnn.out_dim, self.param.num_classes * 4)

        # loss
        self.cross_entropy = None
        self.bbox_loss = None

    @property
    def loss(self):
        return self.cross_entropy + self.bbox_loss

    def forward(self, data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):

        features = self.basic_network(data)

        rois = self.rpn(im_info, gt_boxes, features, gt_ishard, dontcare_areas)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes)
            rois = roi_data[0]

        output = self.rcnn(features, rois)

        cls_pred = self.cls_fc(output)
        cls_score = F.softmax(cls_pred)
        bbox_pred = self.bbox_fc(output)

        if self.training:
            self.cross_entropy, self.bbox_loss = self.build_loss(cls_score, bbox_pred, roi_data[1], roi_data[2], roi_data[3], roi_data[4])

    def build_loss(self, cls_score, bbox_pred, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):

        pass

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas):

        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas)

        rois = util.np_to_variable(rois)
        labels = util.np_to_variable(labels, dtype=torch.LongTensor)
        bbox_targets = util.np_to_variable(bbox_targets)
        bbox_inside_weights = util.np_to_variable(bbox_inside_weights)
        bbox_outside_weights = util.np_to_variable(bbox_outside_weights)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
