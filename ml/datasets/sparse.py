import torch
import numpy as np
import os
from data.util import *

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, i_range, data_path, tpcs=None, transforms={}, threshold=None, full_y=False):
        self.threshold = threshold
        self.transforms = transforms
        self.full_y = full_y
        if not isinstance(data_path, list):
            data_path = [data_path]
            i_range = [i_range]
            tpcs = [tpcs]
        elif tpcs is None:
            tpcs = [None for _ in range(len(data_path))]
        self.files = []
        path_i = 0
        for p, i, t in zip(data_path, i_range, tpcs):
            if i is None: continue
            p = os.path.expandvars(p)
            i = range(i[0], i[1]+1) 
            files = [p+'/'+f for f in os.listdir(p) if file_info(f)[1] in i]
            if t is not None:
                files = [f for f in files if tpc_info(f)[0] in t]
            files = [(f, path_i) for f in files]
            self.files.extend(files)
            path_i += 1
        self.files.sort()
        self.n_samples = len(self.files)

    def __getitem__(self, index):
        f, path_i = self.files[index]
        batch_info = (index, f[f.rfind('/')+1:-4])
        with np.load(f, allow_pickle=True) as xy_f:
            X = xy_f["X"].astype(np.float32)
            Y = xy_f["Y"].astype(np.float32) if "Y" in xy_f else None
        if self.transforms.get("coord_lim"):
            c = self.transforms["coord_lim"]
            for i, lim in enumerate(c):
                if lim is None: continue
                mask = np.logical_and(X[:, i] >= lim[0], X[:, i] < lim[1])
                X = X[mask]
                if Y is not None: Y = Y[mask]
        if self.transforms.get("z_segments"):
            z = self.transforms["z_segments"]
            encoding = []
            null_mask = np.ones(len(X))
            for upperbound in z:
                mask = X[:, 2] < upperbound
                encoding.append(np.logical_and(mask, null_mask))
                null_mask = np.logical_not(mask)
            encoding=np.array(encoding).T
            X = np.concatenate((X, encoding), axis=1)
        if self.transforms.get("noise"):
            noise = self.transforms["noise"]
            X[:, 4:] += np.random.normal(noise["mean"], noise["sd"], (len(X), 3))
        if self.transforms.get("round"):
            rounding = self.transforms["round"]
            X[:, 4:] = np.around(X[:, 4:], rounding)
        if self.transforms.get("scale"):
            scale = self.transforms["scale"]
            if type(scale) == list:
                scale = scale[path_i]
            if Y is not None:
                Y *= scale

        if Y is not None:
            if self.threshold is not None:
                Y = np.where(Y > self.threshold, 1.0, 0.0).astype(np.float32)
            if self.full_y:
                Y = (Y, f)
            return X, Y, batch_info 
        return X, -1, batch_info
            
    def __len__(self):
        return self.n_samples
    
def get_datasets(train_i=None, inference_i=None, **kwargs):
    """Construct and return random number datasets"""
    if train_i is not None:
        train_dataset = SparseDataset(train_i, **kwargs)
    else:
        train_dataset = None
    if inference_i is not None:
        try:
            kwargs["transforms"]["coord_lim"] = None
        except Exception:
            pass
        inference_dataset = SparseDataset(inference_i, **kwargs)
    else:
        inference_dataset = None
    return train_dataset, inference_dataset, {}
