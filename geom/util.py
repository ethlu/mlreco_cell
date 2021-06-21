import numpy as np

def voxelize(pts, voxel_size, to_numpy = True, n_sum = 1, n_avg = 0):
    voxels = dict()
    dim = len(voxel_size)
    for pt in pts:
        vox_i = tuple(pt[:dim]//voxel_size)
        val = pt[dim:]
        try:
            voxels[vox_i].append(val)
        except KeyError:
            voxels[vox_i] = [val]
    for k, vox in voxels.items():
        vox = np.array(vox)
        sums = np.sum(vox[:, :n_sum], 0)
        if n_avg:
            avgs = np.average(vox[:, n_sum: n_sum+n_avg], 0)
        else:
            avgs = []
        modes = [mode1d(x) for x in vox[:, n_sum+n_avg:].T]
        voxels[k] = np.concatenate((sums, avgs, modes))
    if to_numpy:
        return np.array([np.concatenate((k, v)) for k, v in voxels.items()])
    return voxels

def mode1d(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]
