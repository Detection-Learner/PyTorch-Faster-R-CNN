import numpy as np
import torch
import torch.autograd.Variable as Variable

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):

    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()

    return v
