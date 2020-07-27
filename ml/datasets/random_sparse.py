"""
PyTorch dataset description for a sparse random identity dataset.
"""

# Compatibility
from __future__ import print_function

# Externals
import torch
import numpy as np

class RandomDataset(torch.utils.data.Dataset):
    """Random number synthetic dataset.

    For now, generating all requested samples up front.
    """
    def __init__(self, n_samples, spatial_size, n_points):
        self.spatial = spatial_size
        self.n_samples = n_samples
        self.n_points = n_points

    def __getitem__(self, index):
        N = round(np.random.normal(self.n_points))
        coords = torch.randint(self.spatial, [N, 3]).float()
        features = torch.randn([N, 3])
        batch_idx = torch.full([N, 1], index)
        return torch.cat((coords, batch_idx, features), dim=1), features[:, 0:1]

    def __len__(self):
        return self.n_samples

def collate(batch):
    inputs = torch.cat([sample[0] for sample in batch])
    targets = torch.cat([sample[1] for sample in batch])
    return inputs, targets
    
def get_datasets(n_train, n_valid, **kwargs):
    """Construct and return random number datasets"""
    train_dataset = RandomDataset(n_train, **kwargs)
    if n_valid > 0:
        valid_dataset = RandomDataset(n_valid, **kwargs)
    else:
        valid_dataset = None
    return train_dataset, valid_dataset, {"collate_fn": collate}
