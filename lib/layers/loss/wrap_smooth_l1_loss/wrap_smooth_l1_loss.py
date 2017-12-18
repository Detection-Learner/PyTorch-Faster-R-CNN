import torch
from torch.autograd import Function
from torch._C import _infer_size
import torch.nn as nn
from ._ext import wrap_smooth_l1_loss

import numpy as np


class WrapSmoothL1LossFunction(Function):

    def __init__(self, sigma=1.0, size_average=True):

        self.sigma = sigma
        self.size_average = size_average

    def forward(self, input, target, inside_weights=None, outside_weights=None):

        if input.is_cuda:
            loss = torch.zeros(1).cuda()
            wrap_smooth_l1_loss.wrap_smooth_l1_loss_forward_cuda(
                self.sigma, self.size_average, input, target, inside_weights, outside_weights, loss)
        else:
            loss = torch.zeros(1)
            wrap_smooth_l1_loss.wrap_smooth_l1_loss_forward(
                self.sigma, self.size_average, input, target, inside_weights, outside_weights, loss)

        self.save_for_backward(input, target, inside_weights, outside_weights)

        return loss

    def backward(self, grad_output):

        v1, v2, w1, w2 = self.saved_tensors

        grad_input1 = torch.zeros(v1.size())
        grad_input2 = torch.zeros(v2.size())
        if v1.is_cuda:
            grad_input1 = grad_input1.cuda()
            grad_input2 = grad_input2.cuda()
            wrap_smooth_l1_loss.wrap_smooth_l1_loss_backward_cuda(
                self.sigma, self.size_average, v1, v2, w1, w2, grad_input1, grad_input2, grad_output)
        else:
            wrap_smooth_l1_loss.wrap_smooth_l1_loss_backward(
                self.sigma, self.size_average, v1, v2, w1, w2, grad_input1, grad_input2, grad_output)

        return grad_input1, grad_input2, None, None


class WrapSmoothL1Loss(nn.Module):
    def __init__(self, sigma=1.0, size_average=True):
        super(WrapSmoothL1Loss, self).__init__()

        self.sigma = sigma
        self.size_average = size_average

    def forward(self, input, target, inside_weights=None, outside_weights=None):
        if inside_weights is None:
            inside_weights = torch.autograd.Variable(
                torch.ones(input.size()))
        else:
            new_size = _infer_size(input.size(), inside_weights.size())
            inside_weights = inside_weights.expand(new_size)
            if torch.is_tensor(inside_weights):
                inside_weights = torch.autograd.Variable(
                    inside_weights)
        if input.is_cuda:
            inside_weights = inside_weights.cuda()
        if outside_weights is None:
            outside_weights = torch.autograd.Variable(
                torch.ones(input.size()))
        else:
            new_size = _infer_size(input.size(), outside_weights.size())
            outside_weights = outside_weights.expand(new_size)
            if torch.is_tensor(outside_weights):
                outside_weights = torch.autograd.Variable(
                    outside_weights)
        if input.is_cuda:
            outside_weights = outside_weights.cuda()
        return WrapSmoothL1LossFunction(self.sigma, self.size_average)(input, target, inside_weights, outside_weights)
