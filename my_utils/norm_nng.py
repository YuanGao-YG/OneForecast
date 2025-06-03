import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import os

def reshape_fields(img, inp_or_tar, params, train, normalize=True, orog=None, add_noise=False):
    # Takes in np array of size (n_history+1, c, h, w) 
    # returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)
   
    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] # this will either be N_in_channels or N_out_channels
    
    if inp_or_tar == 'inp':
        channels = params.in_channels
    elif inp_or_tar == 'inp2':
        channels = params.in_channels2
    else:
        channels = params.out_channels2


    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    mean_path = os.path.join(parent_dir, params.global_means_path)
    std_path = os.path.join(parent_dir, params.global_stds_path)
    
    if normalize and params.normalization == 'zscore':
        means = np.load(mean_path)[:, channels]
        stds = np.load(std_path)[:, channels]
        img -=means
        img /=stds

    img = np.squeeze(img)

    return torch.as_tensor(img)
    