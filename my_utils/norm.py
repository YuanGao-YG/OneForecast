import logging
import glob
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py

def reshape_fields(img, inp_or_tar, params, train, normalize=True, orog=None, add_noise=False):

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1]
    
    if inp_or_tar == 'inp':
        channels = params.in_channels
    else:
        channels = params.out_channels


    if normalize and params.normalization == 'zscore':
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        img -=means
        img /=stds

    img = np.squeeze(img)
    
    return torch.as_tensor(img)
