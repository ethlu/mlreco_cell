import numpy as np
import os, sys
from data.util import *
from geom.pdsp import get_TPC_box, TPC_LENGTH, APA_HEIGHT, APA_WIDTH, VOXEL_SIZE, DOWNSAMPLE, get_TPC_inverse_coords_transform, get_APA_geom
import itertools
import time

VOXEL_SIZE = np.array(VOXEL_SIZE) * np.array(DOWNSAMPLE)
R0 = get_TPC_box(1)[0]
TPC_LIMS = np.array((TPC_LENGTH, APA_HEIGHT, APA_WIDTH))//VOXEL_SIZE

def parse_y(xy_dir, yinf_file):
    yinf = np.load(yinf_file)[0]
    _, f = inf_file_info(yinf_file)
    xy_file = xy_dir + '/' + f.replace("yinf", "xy") + '.npz'
    with np.load(xy_file) as f:
        coords = f["X"][:,:3] 
        event_starts = f["starts"]
    Y = []
    for event in range(len(event_starts)):
        event_starti = event_starts[event]
        if event == len(event_starts)-1:
            event_coords = coords[event_starti:]
            event_yinf = yinf[event_starti:]
        else:
            event_endi = event_starts[event+1]
            event_coords = coords[event_starti:event_endi]
            event_yinf = yinf[event_starti:event_endi]
        Y.append(np.concatenate((event_coords, event_yinf), 1))
    return Y

def merge_tpcs(xy_dir, yinf_files, links=((1,5), (5,9))):
    """Link tuples should be ordered in increasing r"""
    from scipy.interpolate import LinearNDInterpolator
    INTERPOLATE_THRES = 0.01
    y_tpc, coords_transform_tpc, vox_offset_tpc = {}, {}, {}
    for yinf_file in yinf_files:
        tpc, = tpc_info(yinf_file)
        y_tpc[tpc] = parse_y(xy_dir, yinf_file)
        n_events = len(y_tpc[tpc])
        coords_transform, vox_offset = get_TPC_inverse_coords_transform(tpc, VOXEL_SIZE, R0) 
        print(vox_offset)
        coords_transform_tpc[tpc] = coords_transform
        vox_offset_tpc[tpc] = vox_offset
    Y = [[] for _ in range(n_events)]
    for tpc1, tpc2 in links:
        r_diff = get_TPC_box(tpc2)[0] - get_TPC_box(tpc1)[1]
        link_i = np.argwhere(r_diff > 0)
        assert len(link_i) == 1
        link_i = link_i[0][0]
        link_dist = round(r_diff[link_i]/VOXEL_SIZE[link_i])
        if link_dist == 0: continue
        print(link_dist)
        tpc_lim = TPC_LIMS[link_i]
        for event in range(n_events):
            y1, y2 = y_tpc[tpc1][event], y_tpc[tpc2][event]
            y1_slice = y1[y1[:, link_i] == tpc_lim]
            y2_slice = y2[y2[:, link_i] == 0]
            y2_slice[:, link_i] = tpc_lim + 1 + link_dist
            coords = np.concatenate((y1_slice[:, :3], y2_slice[:, :3]))
            vals = np.concatenate((y1_slice[:, 3], y2_slice[:, 3]))
            interpolator = LinearNDInterpolator(coords, vals)
            grid_coords = []
            for i in range(3):
                if i == link_i:
                    grid_coords.append(range(tpc_lim+1, tpc_lim+1+link_dist))
                else:
                    grid_coords.append(range(TPC_LIMS[i]))
            grid_coords = itertools.product(*grid_coords)
            grid_vals = interpolator(grid_coords)
            grid_mask = grid_vals > INTERPOLATE_THRES
            grid_coords = grid_coords[grid_mask]
            grid_vals = grid_vals[grid_mask]
            grid_y = np.concatenate((grid_coords, grid_vals))
            Y[event].append(coords_transform_tpc[tpc1](grid_y))
    for tpc, y in y_tpc.items():
        for event in range(n_events):
            Y[event].append(coords_transform_tpc[tpc](y[event]))
    return [np.concatenate(*event_y) for event_y in Y]
#merge_tpcs(sys.argv[1], [sys.argv[2]])

def charge_solve(xy_dir, parsed_dir, yinf_file, inf_thres=0.35, X_type="wire"):
    from tiling.pixel import Solver
    Y = parse_y(xy_dir, yinf_file)
    tpc, = tpc_info(yinf_file)
    _, index, _ = file_info(yinf_file)
    for f in os.listdir(parsed_dir):
        info = file_info(f)
        if info is None: continue
        if info[1] == index and info[2] == X_type and tpc_info(f)[0] == tpc:
            chan_file = f
            break
    print(chan_file)
    with np.load(parsed_dir+'/'+chan_file) as f:
        chan_vals = [f["arr_%d"%i] for i in range(len(Y))]
    geom = get_APA_geom(tpc)
    print("done loading")
    t0 = time.time()
    solver = Solver(geom, DOWNSAMPLE[1:], "Lasso")
    t1 = time.time()
    print("done loading solver: ", t1-t0)
    ret = []
    for yinf, chan_val in zip(Y, chan_vals):
        yinf = yinf[yinf[:, 3] >= inf_thres][:, :3].astype(np.int)
        yinf[:, 1], yinf[:, 2] = yinf[:, 2], yinf[:, 1].copy()
        slic_xs, slic_ptrs = np.unique(yinf[:, 0], return_index=True)
        charges = []
        for slic_i in range(len(slic_ptrs)):
            if slic_i == len(slic_ptrs)-1:
                pixels = yinf[slic_ptrs[slic_i]:, 1:3]
            else:
                pixels = yinf[slic_ptrs[slic_i]:slic_ptrs[slic_i+1], 1:3]
            slic_x = slic_xs[slic_i]*DOWNSAMPLE[0]
            chan_vals_slic = np.sum(chan_val[slic_x: slic_x+DOWNSAMPLE[0]], axis=0)
            #chan_vals_slic = chan_val[slic_x: slic_x+DOWNSAMPLE[0]].T
            charges.extend(solver(chan_vals_slic, pixels))
            #charges.extend(np.sum(solver(chan_vals_slic, pixels), axis=0))
        yinf[:, 1], yinf[:, 2] = yinf[:, 2], yinf[:, 1].copy()
        ret.append(np.concatenate((yinf, np.array(charges).reshape(-1, 1)), 1))
    print("done solving", time.time()-t1)
    return ret

#charge_solve(sys.argv[1], sys.argv[2], sys.argv[3])
def process_multi(xy_dir, parsed_dir, yinf_dir, out_dir, evt_type, epoch=-1, tpc=None):
    yinf_fd = filter_fd(files_info(os.listdir(yinf_dir), (0, 1)), lambda k,v: k[0]==evt_type and k[1]>=301 and k[1]<=301)
    yinf_files = flatten_fd(yinf_fd)
    if tpc is not None:
        yinf_files = filter(lambda f: tpc_info(f)[0]==tpc, yinf_files)
    yinf_epoch = files_info(yinf_files, [0], inf_file_info)
    epoch = sorted(yinf_epoch.keys())[epoch]
    yinf_files = yinf_epoch[epoch]
    print(yinf_files)
    for f in yinf_files:
        out_f = f[:-4]
        f_charge = charge_solve(xy_dir, parsed_dir, yinf_dir+'/'+f)
        np.savez_compressed(out_dir+'/'+out_f, *f_charge)

process_multi(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], "Mu", -1, 1)
