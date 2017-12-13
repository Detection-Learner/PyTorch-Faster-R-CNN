import torch
from torch.autograd import Function
import torch.nn as nn
from ._ext import WrapSmoothL1Loss

import numpy as np

class WrapSmoothL1LossFunction(Function):

    def __init__(self, sigma=1.0, size_average=True):
        
        self.sigma = sigma
        self.size_average = size_average

    def forward(self, input, target, inside_weights=None, outside_weights=None):

        loss = torch.zeros(1).cuda()
        WrapSmoothL1Loss.smooth_l1_loss_forward_cuda(self.sigma, self.size_average, input, target, inside_weights, outside_weights)

        self.save_for_backward(input, target, inside_weights, outside_weights)

        return loss

    def backward(self, grad_output):

        v1, v2, w1, w2 = self.saved_tensors

        grad_input = v1.new().resize_as_(v1).fill_(0)
        WrapSmoothL1Loss.smooth_l1_loss_backward_cuda(self.sigma, self.size_average, v1, v2, w1, w2)

        return grad_input, None, None, None

class WrapSmoothL1Loss(nn.Module):
    def __init__(self, sigma=1.0, size_average=True):
        super(WrapSmoothL1Loss, self).__init__()

        self.sigma = sigma
        self.size_average = size_average

    def forward(self, input, target, inside_weights=None, outside_weights=None):
        return WrapSmoothL1LossFunction(self.sigma, self.size_average)(input, target, inside_weights, outside_weights)
