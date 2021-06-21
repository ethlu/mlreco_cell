import numpy as np
import os, sys
from geom.pdsp import get_APA_geom
from data.util import *
from geom.util import *
from tiling.pixel import Pixelator
import time

def charge_solve(parsed_dir, chan_file, downsample=(1, 1, 1), Y_thres=0, Y_type="depo"):
    from tiling.pixel import Solver
    tpc, = tpc_info(chan_file)
    _, index, _ = file_info(chan_file)
    for f in os.listdir(parsed_dir):
        info = file_info(f)
        if info is None: continue
        if info[1] == index and info[2] == Y_type and tpc_info(f)[0] == tpc:
            truth_file = f
    print(chan_file, truth_file)
    with np.load(parsed_dir+'/'+truth_file) as f:
        i = 0
        Y = []
        while "arr_%d"%i in f:
            Y.append(voxelize(f["arr_%d"%i], downsample))
            i+=1
    with np.load(chan_file) as f:
        chan_vals = [f["arr_%d"%i] for i in range(len(Y))]
    geom = get_APA_geom(tpc)
    print("done loading")
    t0 = time.time()
    solver = Solver(geom, downsample[1:], regression = "Lasso")
    t1 = time.time()
    print("done loading solver: ", t1-t0)
    ret = []
    for yinf, chan_val in zip(Y, chan_vals):
        yinf = yinf[yinf[:, 3] >= Y_thres][:, :3].astype(np.int)
        slic_xs = np.unique(yinf[:, 0])
        charges = []
        for slic_x in slic_xs:
            pixels = yinf[yinf[:, 0] == slic_x]
            chan_vals_slic = np.sum(chan_val[slic_x*downsample[0]: (slic_x+1)*downsample[0]], axis=0)
            charges.extend(np.concatenate((pixels, solver(chan_vals_slic, pixels[:, 1:3][:, ::-1]).reshape(-1, 1)), 1))
        ret.append(charges)
    print("done solving", time.time()-t1)
    return ret

def process_multi(parsed_dir, out_dir, chan_type="wire", evt_type=None, tpc=None):
    chan_fd = filter_fd(files_info(os.listdir(parsed_dir), (0, 1, 2)), \
            lambda k,v: (evt_type is None or k[0]==evt_type) and k[1]>=0 and k[1]<=1000 and k[2] == chan_type)
    chan_files = flatten_fd(chan_fd)
    if tpc is not None:
        chan_files = list(filter(lambda f: tpc_info(f)[0]==tpc, chan_files))
    print(chan_files)
    for f in chan_files:
        out_f = f[:-4].replace(chan_type, "charge")
        print(out_f)
        f_charge = charge_solve(parsed_dir, parsed_dir+'/'+f)
        np.savez_compressed(out_dir+'/'+out_f, *f_charge)


process_multi(sys.argv[1], sys.argv[2], tpc = 1)

"""
def process(pixelator, event_i = None, in_file = sys.argv[1], root_dir = sys.argv[3], out_dir = sys.argv[2]):
    out_file = out_dir + os.path.relpath(in_file, root_dir)
    print(out_file)
    with np.load(in_file) as batch_chan_vals:
        if event_i is not None:
            event_pixels = pixelator(batch_chan_vals["arr_%d"%event_i])
            if out_dir is None: 
                return event_pixels
            np.savez_compressed(out_file.replace("wire", "pixel").replace(".npz", "-Evt%d"%event_i), event_pixels)
            return
        batch_channel_vals=[]
        i = 0
        while "arr_%d"%i in batch_chan_vals:
            batch_channel_vals.append(batch_chan_vals["arr_%d"%i])
            i += 1
    t0 = time.time()
    batch_pixels, event_starts = pixelator(batch_channel_vals)
    t1 = time.time()
    print("Time to pixelate", t1 - t0)
    if out_dir is None: 
        return batch_pixels
    np.savez_compressed(out_file.replace("wire", "pixel"), batch_pixels, event_starts)

def process_multi(pixelator, root_dir = sys.argv[1]):
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if "wire" in f:
                process(pixelator, None, root+"/"+f, root_dir)


#geom = get_APA_geom()
#pix = Pixelator(geom, sparse_output = True)
def pix_batch(batch):
    event_pixels = []
    for i, slic in enumerate(batch):
        print(i)
        event_pixels.append(pix(slic))
    return event_pixels

#pix = pix.to_numba()
#process(pix)
"""
