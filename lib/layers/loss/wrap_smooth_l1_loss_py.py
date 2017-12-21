import torch
import torch.nn as nn


class WrapSmoothL1Loss(nn.Module):
    def __init__(self, sigma=1.0, size_average=True):
        super(WrapSmoothL1Loss, self).__init__()

        self.sigma = sigma
        self.size_average = size_average

    def forward(self, inputs, targets, inside_weights=None, outside_weights=None):
        assert (inputs.size() == targets.size())  # == len(targets.size())
        val = inputs - targets
        if inside_weights is not None:
            val = val * inside_weights
        abs_val = torch.abs(val)
        x = abs_val < 1. / self.sigma**2
        y = abs_val >= 1. / self.sigma**2
        abs_val[x] = 0.5 * \
            torch.pow(abs_val[x], 2) / self.sigma**2
        abs_val[y] = abs_val[y] - 0.5 / self.sigma**2
        loss = abs_val
        if outside_weights is not None:
            loss = abs_val * outside_weights
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss).div_(inputs.size(0))
        return loss
