import torch
import numpy as np
import os
from data.util import *

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, i_start, i_end, data_path, threshold=None):
        self.threshold = threshold
        self.data_path = os.path.expandvars(data_path)
        i_range = range(i_start, i_end+1)
        self.files = sorted([f for f in os.listdir(self.data_path) if file_info(f)[1] in i_range])
        self.n_samples = len(self.files)

    def __getitem__(self, index):
        f = self.files[index]
        batch_info = (index, f[:-4])
        with np.load(self.data_path + "/" + f) as xy_f:
            if self.threshold is None:
                return xy_f["X"], xy_f["Y"].astype(np.float32), batch_info 
            return xy_f["X"], np.where(xy_f["Y"] > self.threshold, 1.0, 0.0).astype(np.float64), batch_info
            
    def __len__(self):
        return self.n_samples
    
def get_datasets(train_i=None, inference_i=None, **kwargs):
    """Construct and return random number datasets"""
    if train_i is not None:
        train_dataset = SparseDataset(*train_i, **kwargs)
    else:
        train_dataset = None
    if inference_i is not None:
        inference_dataset = SparseDataset(*inference_i, **kwargs)
    else:
        inference_dataset = None
    return train_dataset, inference_dataset, {}
