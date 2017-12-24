import torch
from torch.autograd import Function
from torch._C import _infer_size
import torch.nn as nn
from ._ext import warp_smooth_l1_loss

import numpy as np


class WarpSmoothL1LossFunction(Function):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
    Also known as the Huber loss::

                              { 0.5 * w_out * (w_in * (x_i - y_i) * sigma^2)^2, if |w_in * (x_i - y_i)| < 1 / sigma^2
        loss(x, y) = 1/n \sum {
                              { w_out * (|w_in * (x_i - y_i)| - 0.5 / sigma^2),   otherwise

    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be batch size if one sets the internal variable
    `size_average` to ``False``

    Args:
        sigma (float): By default, the losses are computed same as the `nn.SmoothL1Loss`
            with all default. Default: ``1.0``
        size_average (bool, optional): By default, the losses are averaged
           over all elements. However, if the field size_average is set to ``False``,
           the losses are averaged over batch size `N`. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - inside_weights: :math:`(N, *)`, same shape as the input
        - outside_weights: :math:`(N, *)`, same shape as the input
        - Output: scalar.

    """

    def __init__(self, sigma=1.0, size_average=True):
        self.sigma = sigma
        self.size_average = size_average

    def forward(self, input, target, inside_weights=None, outside_weights=None):
        if self.size_average:
            size_avg = input.numel()
        else:
            size_avg = input.size(0)
        if input.is_cuda:
            loss = torch.zeros(1).cuda()
            warp_smooth_l1_loss.warp_smooth_l1_loss_forward_cuda(
                self.sigma, size_avg, input, target, inside_weights, outside_weights, loss)
        else:
            loss = torch.zeros(1)
            warp_smooth_l1_loss.warp_smooth_l1_loss_forward(
                self.sigma, size_avg, input, target, inside_weights, outside_weights, loss)

        self.save_for_backward(input, target, inside_weights, outside_weights)

        return loss

    def backward(self, grad_output):

        v1, v2, w1, w2 = self.saved_tensors

        if self.size_average:
            size_avg = v1.numel()
        else:
            size_avg = v1.size(0)

        grad_input1 = torch.zeros(v1.size())
        grad_input2 = torch.zeros(v2.size())
        if v1.is_cuda:
            grad_input1 = grad_input1.cuda()
            grad_input2 = grad_input2.cuda()
            warp_smooth_l1_loss.warp_smooth_l1_loss_backward_cuda(
                self.sigma, size_avg, v1, v2, w1, w2, grad_input1, grad_input2, grad_output)
        else:
            warp_smooth_l1_loss.warp_smooth_l1_loss_backward(
                self.sigma, size_avg, v1, v2, w1, w2, grad_input1, grad_input2, grad_output)

        return grad_input1, grad_input2, None, None


class WarpSmoothL1Loss(nn.Module):
    def __init__(self, sigma=1.0, size_average=True):
        super(WarpSmoothL1Loss, self).__init__()

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
        return WarpSmoothL1LossFunction(self.sigma, self.size_average)(input, target, inside_weights, outside_weights)
