import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
from my_utils.norm import reshape_fields

def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    dataloader = DataLoader(dataset,
                            batch_size  = 1,
                            num_workers = params.num_data_workers,
                            shuffle     = False,
                            sampler     = sampler if train else None,
                            drop_last   = True,
                            pin_memory  = True)

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.normalize = params.normalize
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.atmos_channels = np.array(params.atmos_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        self._get_files_stats()


    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))

            self.n_samples_per_year = _f['fields'].shape[0] - self.params.multi_steps_finetune 

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] - 1 # just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(
            1 * self.dt * self.n_history, 1 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.params.multi_steps_finetune == 1:
            if local_idx == 1463:
                local_idx = 1462
            if local_idx == 1464:
                local_idx = 1463
            
            inp = reshape_fields( 
                    np.nan_to_num(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels, :120, :240], nan=0), 
                    'inp', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            
            tar = reshape_fields(
                    np.nan_to_num(self.files[year_idx][local_idx+step, self.out_channels, :120, :240], nan=0), 
                    'tar', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )
      
        return inp, tar  