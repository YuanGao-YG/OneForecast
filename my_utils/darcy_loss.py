import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
from icecream import ic
from functools import partial
from torch.autograd import Function
from torch.nn import Module, ModuleList, Sequential


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def safe_div(num, den, eps = 1e-10):
    return num / den.clamp(min = eps)



# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # [B]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class channel_wise_LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, scale=False):
        super(channel_wise_LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.scale = scale
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        num_channels = x.size()[1]

        x = x.reshape(num_examples, num_channels, -1)
        y = y.reshape(num_examples, num_channels, -1)

        # [B, C]
        diff_norms = torch.norm(x.reshape(num_examples, num_channels, -1) - y.reshape(num_examples, num_channels, -1), self.p, 2)

        y_norms = torch.norm(y.reshape(num_examples, num_channels, -1), self.p, 2)

        if self.reduction:
            if self.size_average:
                if self.scale:
                    channel_wise_mean = torch.mean(diff_norms/y_norms, 0) # [C]: Li, i=1,2,3,...,C
                    channel_mean = torch.mean(diff_norms/y_norms) # scaler 

                    scale_w = channel_mean / channel_wise_mean # [C]: L1/Li, i=1,2,3,...,C
                    channel_scale = torch.mean(scale_w * channel_wise_mean) # \sum w_i*L_i
                    return channel_scale, channel_wise_mean * scale_w 
                else:
                    channel_mean = torch.mean(diff_norms/y_norms, 0)
                    return torch.mean(diff_norms/y_norms), channel_mean
            else:
                if self.scale:
                    channel_sum = torch.sum(diff_norms/y_norms, 0)
                    scale_w = channel_sum[0] / channel_sum
                    channel_sum_scale = torch.sum(scale_w * channel_sum)
                    return channel_sum_scale, channel_sum*scale_w
                else:
                    channel_sum = torch.sum(diff_norms/y_norms, 0)
                    return torch.sum(diff_norms/y_norms), channel_sum
        else:
            return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class LossScaleFunction(Function):
    """
    refer to MetNet-3
    """
    @staticmethod
    def forward(ctx, x, eps):
        ctx.eps = eps
        assert x.ndim == 4
        return x

    @staticmethod
    def backward(ctx, grads):
        num_channels = grads.shape[1]

        safe_div_ = partial(safe_div, eps = ctx.eps)

        weight = safe_div_(1., grads.norm(p = 2, keepdim = True, dim = (-1, -2)))
        l1_normed_weight = safe_div_(weight, weight.sum(keepdim = True, dim = 1))

        scaled_grads = num_channels * l1_normed_weight * grads

        return scaled_grads, None

class LossScaler(Module):
    """
    refer to MetNet-3
    """
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return LossScaleFunction.apply(x, self.eps)