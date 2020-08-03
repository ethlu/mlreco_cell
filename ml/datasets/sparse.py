import torch
import numpy as np
import os

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, i_start, n_samples, data_path):
        self.i_start = i_start
        self.n_samples = n_samples
        self.data_path = os.path.expandvars(data_path)
        self.files = os.listdir(self.data_path)

    def __getitem__(self, index):
        f = [f for f in self.files if "batch%d"%(index+self.i_start) in f]
        assert len(f) == 1
        with np.load(self.data_path + "/" + f[0]) as xy_f:
            return xy_f["X"], xy_f["Y"].astype(np.float32)
            
    def __len__(self):
        return self.n_samples
    
def get_datasets(n_train, n_valid, **kwargs):
    """Construct and return random number datasets"""
    train_dataset = SparseDataset(0, n_train, **kwargs)
    if n_valid > 0:
        valid_dataset = SparseDataset(n_train, n_valid, **kwargs)
    else:
        valid_dataset = None
    return train_dataset, valid_dataset, {}
