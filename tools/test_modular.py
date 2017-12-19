import os
import sys
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_PATH, '..'))
from lib.layers.rpn.proposal_layer import proposal_layer
from lib.layers.rpn.anchor_target_layer import anchor_target_layer

from lib.layers.rpn.rpn_layer import RPN
from lib.layers.rpn.fpn_layer import FPN
from lib.layers.loss.wrap_smooth_l1_loss.wrap_smooth_l1_loss import WrapSmoothL1Loss
from lib.layers.loss.wrap_smooth_l1_loss_py import WrapSmoothL1Loss as WrapSmoothL1Loss_py

from lib.layers.roi_data_layer.image_loader import ImageLoader, detection_collate
import torch
from torch.autograd import gradcheck
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def test_image_loader():

    loader = ImageLoader('/home/stick/Dataset/Detection_demo/img_list.txt',
                         '/home/stick/Dataset/Detection_demo/ann_list.txt')

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=True,
        num_workers=3, collate_fn=detection_collate, pin_memory=True)

    for i, (imgs, im_infos, gt_boxes) in enumerate(train_loader):
        print imgs.size()
        # print len(im_infos), [len(im_info) for im_info in im_infos]
        # print len(gt_boxes), [len(gt_box) for gt_box in gt_boxes]
        # add
        len_batch = len(gt_boxes)
        for i in range(len_batch):
            img = imgs.numpy()[i].transpose(1, 2, 0)
            gt_box = gt_boxes[i]
            print type(gt_boxes), gt_box.shape
            print type(img)
            img *= 128
            img += 128
            img /= 255
            plt.imshow(img)
            for box in gt_box:
                plt.plot([box[0], box[2], box[2], box[0], box[0]], [
                         box[1], box[1], box[3], box[3], box[1]], 'r-')
            plt.show()
            a = raw_input('Continue? ')
            if a == 'Q' or a == 'q':
                sys.exit(1)


def test_proposal_layer():

    rpn_cls_prob = np.random.rand(15, 9 * 2, 64, 86)
    rpn_bbox_pred = np.random.rand(15, 9 * 4, 64, 86)
    im_info = np.array([480, 1200, 0.5] * 15).reshape(-1, 3)
    # print im_info[0]
    blobs = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, 'TRAIN')
    print len(blobs), blobs[0].shape
    print blobs[1]


def test_anchor_target_layer():

    rpn_cls_prob = np.random.rand(15, 9 * 2, 64, 86)
    gt_boxes = [np.random.randint(
        1, 255, (np.random.randint(1, 10), 5)) for i in range(15)]
    for i in range(15):
        gt_boxes[i].sort()
    gt_ishard = None
    dontcare_areas = None
    im_info = np.array([480, 1200, 0.5] * 15).reshape(-1, 3)
    # print im_info[0]
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
        rpn_cls_prob, gt_boxes, gt_ishard, dontcare_areas, im_info)
    print rpn_labels.shape, rpn_bbox_targets.shape, rpn_bbox_inside_weights.shape, rpn_bbox_outside_weights.shape
    print sum(rpn_labels[1].reshape(-1, 1) == 1)


def test_rpn():

    rpn = RPN(10, in_channel=5)
    optimizer = torch.optim.SGD(rpn.parameters(), 0.01)
    rpn.cuda()
    rpn.train()
    # rpn.eval()

    feature = torch.autograd.Variable(torch.Tensor(15, 5, 64, 86)).cuda()
    im_info = np.array([480, 1200, 0.5] * 15).reshape(-1, 3)
    gt_boxes = [np.random.randint(
        1, 255, (np.random.randint(1, 10), 5)) for i in range(15)]
    for i in range(15):
        gt_boxes[i].sort()
        gt_boxes[i][:, 4] = gt_boxes[i][:, 4].clip(max=9)
    pooled_features, rpn_data_out = rpn(feature, im_info, gt_boxes)
    print pooled_features.shape, rpn_data_out[0].shape, rpn.loss

    optimizer.zero_grad()
    rpn.loss.backward()
    optimizer.step()


def test_fpn():
    fpn = FPN(num_classes=10, in_channels=[10, 5], feat_strides=[16, 8])
    optimizer = torch.optim.SGD(fpn.parameters(), 0.01)
    fpn.cuda()
    fpn.train()
    # fpn.eval()
    feature1 = torch.autograd.Variable(torch.Tensor(15, 10, 64, 86)).cuda()
    feature2 = torch.autograd.Variable(torch.Tensor(15, 5, 32, 43)).cuda()
    im_info = np.array([480, 1200, 0.5] * 15).reshape(-1, 3)
    gt_boxes = [np.random.randint(
        1, 255, (np.random.randint(1, 10), 5)) for i in range(15)]
    for i in range(15):
        gt_boxes[i].sort()
        gt_boxes[i][:, 4] = gt_boxes[i][:, 4].clip(max=9)
    pooled_features, fpn_data_out = fpn(
        [feature1, feature2], im_info, gt_boxes)
    print pooled_features.shape, fpn_data_out[0].shape
    print type(pooled_features)
    print fpn.loss

    optimizer.zero_grad()
    fpn.loss.backward()
    optimizer.step()


def test_warp_smooth_l1_loss():
    warp_smooth_l1_loss = WrapSmoothL1Loss(
        sigma=1.0, size_average=True).cuda()
    warp_smooth_l1_loss_py = WrapSmoothL1Loss_py(sigma=1.0, size_average=True)
    smooth_l1_loss = torch.nn.SmoothL1Loss(size_average=True).cuda()
    feature1 = torch.autograd.Variable(
        torch.ones(10, 2)).cuda()  # torch.Tensor(15, 10)
    feature2 = torch.autograd.Variable(torch.ones(10, 2) * 2).cuda()
    feature1[9, 1] = 2
    l1 = warp_smooth_l1_loss(feature1, feature2)
    l1_py = warp_smooth_l1_loss_py(feature1, feature2)
    l2 = smooth_l1_loss(feature1, feature2)
    print l1, l1_py, l2


def test_warp_smooth_l1_loss_backward():
    warp_smooth_l1_loss = WrapSmoothL1Loss(
        sigma=1.0, size_average=True).cuda()
    warp_smooth_l1_loss_py = WrapSmoothL1Loss_py(sigma=1.0, size_average=True)
    smooth_l1_loss = torch.nn.SmoothL1Loss(size_average=True).cuda()
    feature1 = torch.autograd.Variable(
        torch.ones(10, 2))  # .cuda()  # torch.Tensor(15, 10)
    feature2 = torch.autograd.Variable(torch.ones(10, 2) * 2)  # .cuda()
    feature1[9, 1] = 2
    feature1[5, 1] = 10
    test = gradcheck(warp_smooth_l1_loss,
                     (feature1, feature2), eps=1e-6, atol=1e-4)
    test_py = gradcheck(F.smooth_l1_loss,
                        (feature1, feature2, True), eps=1e-6, atol=1e-4)
    test1 = gradcheck(warp_smooth_l1_loss_py,
                      (feature1, feature2), eps=1e-6, atol=1e-4)
    print test, test_py, test1


if __name__ == '__main__':
    test_image_loader()
