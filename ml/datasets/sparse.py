import torch
import numpy as np
import os

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, i_start, i_end, data_path, threshold=None):
        self.i_start = i_start
        self.n_samples = i_end - i_start
        self.threshold = threshold
        self.data_path = os.path.expandvars(data_path)
        self.files = sorted(os.listdir(self.data_path))

    def __getitem__(self, index):
        f = self.files[index+self.i_start]
        with np.load(self.data_path + "/" + f) as xy_f:
            if self.threshold is None:
                return xy_f["X"], xy_f["Y"].astype(np.float32), index
            return xy_f["X"], np.where(xy_f["Y"] > self.threshold, 1.0, 0.0).astype(np.float64), index
            
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
