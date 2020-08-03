import numpy as np

def voxelize(pts, voxel_size, to_numpy = True):
    voxels = dict()
    dim = len(voxel_size)
    for pt in pts:
        vox_i = tuple(pt[:dim]//voxel_size)
        val = pt[dim:]
        try:
            voxels[vox_i] += val
        except KeyError:
            voxels[vox_i] = val
    if to_numpy:
        return np.array([np.concatenate((k, v)) for k, v in voxels.items()])
    return voxels
