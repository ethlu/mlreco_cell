import numpy as np
import os, sys, time
from geom.pdsp import get_APA_geom, VOXEL_SIZE, DOWNSAMPLE, get_TPC_inverse_coords_transform, get_TPC_box
from geom.util import voxelize
from data.util import *

#DOWNSAMPLE = (4, 2, 2)
VOXEL_SIZE = np.array(VOXEL_SIZE) * np.array(DOWNSAMPLE)
NUM_BATCHES = 1 #number of events per parsed file // batch_size; just to make sure we are done with the file
#X_TYPE = "hit"
X_TYPE = "wire"
Y_TYPE = ("energy", "depo", "depoElectron")
#Y_TYPE = None

def process_xy(downsample = DOWNSAMPLE, save_pix=False, batch_size = None, tpc_num = int(sys.argv[3]), parsed_dir = sys.argv[1], out_dir = sys.argv[2]):
    from tiling.pixel import Pixelator, Geom
    def info(f):
        try:
            return [file_info(f)[1], tpc_info(f)[0], file_info(f)[0]]
        except:
            return None
    in_fd = filter_fd(files_info(os.listdir(parsed_dir), (0, 1, 2), info), lambda k,v: k[1]==tpc_num)
    #in_fd = filter_fd(files_info(os.listdir(parsed_dir), (0, 1, 2), info), lambda k,v: k[0]>=300 and k[0]<=305 and k[1]==tpc_num)
    done_fd = filter_fd(files_info(os.listdir(out_dir), (0, 1, 2), info), lambda k,v: len(v)==NUM_BATCHES)
    fd = diff_fd(in_fd, done_fd)
    fd_keys = slurm_split(sorted(fd.keys()))

    print("file indexes: ", fd_keys)
    print("num files: ", len(fd_keys))

    geom = get_APA_geom(tpc_num)
    pix = Pixelator(geom)
    pix = pix.to_numba()
    for index in fd_keys:
        type_fd = files_info(fd[index], [2])
        cf = type_fd[X_TYPE,][0] 
        with np.load(parsed_dir+'/'+cf) as cf_vals:
            cf_event_vals=[]
            i = 0
            while "arr_%d"%i in cf_vals:
                cf_event_vals.append(cf_vals["arr_%d"%i])
                i += 1
        if Y_TYPE is not None:
            if type(Y_TYPE) is tuple:
                for y_type in Y_TYPE:
                    if (y_type,) in type_fd:
                        ef = type_fd[y_type,][0]
                        break
            else:
                ef = type_fd[Y_TYPE,][0]
            with np.load(parsed_dir+'/'+ef) as ef_vals:
                ef_event_vals=[]
                i = 0
                while "arr_%d"%i in ef_vals:
                    ef_event_vals.append(ef_vals["arr_%d"%i])
                    i += 1
            assert len(ef_event_vals) == len(cf_event_vals), "energy/channel files mismatch"
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = len(cf_event_vals)//batch_size
        cf_batches = np.split(np.array(cf_event_vals), num_batches)
        if Y_TYPE is not None:
            ef_batches = np.split(np.array(ef_event_vals), num_batches)
        for batch in range(num_batches):
            out_f = "batch%d-"%batch + cf.replace(X_TYPE, "xy")
            cf_batch = cf_batches[batch]
            pix_batch = pix(cf_batch)
            if save_pix:
                out_pixel_f = "batch%d-"%batch + cf.replace(X_TYPE, "pixel")
                pix_original, starts_original = Pixelator.numba_to_numpy(pix_batch)
                np.savez_compressed(parsed_dir+"/"+out_pixel_f, 
                    pix=pix_original.astype(np.float32), starts=starts_original)
            pix_batch = Pixelator.downsamples(pix_batch, (downsample[0], downsample[2], downsample[1]))

            if Y_TYPE is not None:
                ef_batch = ef_batches[batch]
                truth_dict_batch = [voxelize(truth_event, downsample, False) for truth_event in ef_batch]
                Y_truth = np.array([np.array([np.concatenate((k, v)) for k, v in voxels.items()]) for voxels in truth_dict_batch])
                X, Y, event_starts= generate_xy(pix_batch, truth_dict_batch)
                np.savez_compressed(out_dir+"/"+out_f, X=X.astype(np.float32), Y=Y.astype(np.float32), starts=event_starts, Y_truth=Y_truth)
            else:
                X, event_starts = Pixelator.numba_to_numpy(pix_batch)
                X[:, 1], X[:, 2] = X[:, 2], X[:, 1].copy()
                np.savez_compressed(out_dir+"/"+out_f, X=X.astype(np.float32), starts=event_starts)

def generate_xy(pix_batch, truth_dict_batch, num_truths=1):
    num_events = len(pix_batch)
    num_slices = len(pix_batch[0])
    non_active_truth = np.zeros(num_truths)
    X, Y = [], []
    event_starts = []
    for event_i in range(num_events):
        event_starts.append(len(Y))
        truth_dict = truth_dict_batch[event_i]
        for slic_i in range(num_slices):
            for pt in pix_batch[event_i][slic_i]:
                coord = (slic_i, int(pt[1]), int(pt[0])) #swap XY
                vals = np.array(pt[2:])
                if coord in truth_dict:
                    truth_vals = truth_dict[coord][:num_truths]
                else:
                    truth_vals = non_active_truth
                X.append(np.concatenate([coord, (event_i,), vals]))
                Y.append(truth_vals)
    return np.array(X), np.array(Y), np.array(event_starts)

def process_xy_from_pix(downsample = DOWNSAMPLE, parsed_dir = sys.argv[1], out_dir = sys.argv[2]):
    import re
    in_fd = filter_fd(files_info(os.listdir(parsed_dir)), 
        lambda k,v: k[0]>0 and k[0]<=1000 and ("pixel",) in files_info(v, [2]))
    done_fd = filter_fd(files_info(os.listdir(out_dir)), lambda k,v: len(v)==NUM_BATCHES)
    fd = diff_fd(in_fd, done_fd)
    fd_keys = slurm_split(sorted(fd.keys()))
    for index in fd_keys:
        type_fd = files_info(fd[index], [2])
        ef = type_fd[Y_TYPE,][0]
        pfs = type_fd["pixel",]
        with np.load(parsed_dir+'/'+ef) as ef_vals:
            ef_event_vals=[]
            i = 0
            while "arr_%d"%i in ef_vals:
                ef_event_vals.append(ef_vals["arr_%d"%i])
                i += 1
        num_truths = len(ef_event_vals[0][0])-3
        non_active_truth = np.zeros(num_truths)
        for pf in pfs:
            out_f = pf.replace("pixel", "xy")
            X, Y, Y_truth= [], [], []
            event_starts_new = []
            batch = int(re.findall("batch(\d+).+np.", pf)[0])
            with np.load(parsed_dir+'/'+pf) as pix_f:
                pix_batch = pix_f["pix"]
                event_starts = pix_f["starts"]
                batch_size = len(event_starts)
            for event in range(batch_size):
                event_starts_new.append(len(Y))
                event_starti = int(event_starts[event])
                if event == len(event_starts)-1:
                    event_endi = -1
                    event_pixels = pix_batch[event_starti:]
                else:
                    event_endi = int(event_starts[event+1])
                    event_pixels = pix_batch[event_starti:event_endi]
                truth_event = ef_event_vals[batch*batch_size + event]

                event_pixels = voxelize(event_pixels, downsample+(1,))
                truth_dict = voxelize(truth_event, downsample, False)
                
                for pt in event_pixels:
                    coord = (int(pt[0]), int(pt[2]), int(pt[1])) #swap XY
                    vals = np.array(pt[3:])
                    if coord in truth_dict:
                        truth_vals = truth_dict[coord]
                    else:
                        truth_vals = non_active_truth
                    X.append(np.concatenate([coord, vals]))
                    Y.append(truth_vals)
                Y_truth.append([np.concatenate((k, v)) for k, v in truth_dict.items()])
            np.savez_compressed(out_dir+"/"+out_f, 
                    X=np.array(X).astype(np.float32), 
                    Y=np.array(Y).astype(np.float32), 
                    starts=event_starts_new, 
                    Y_truth=Y_truth)

def join_tpcs(tpcs=[1, 5, 9], xy_dir = sys.argv[1]):
    R0 = get_TPC_box(tpcs[0])[0]
    tpcs_key = int("".join(map(str, tpcs)))
    def info(f):
        try:
            return [file_info(f)[1], batch_info(f)[0]]
        except:
            return None
    fd = filter_fd(files_info(os.listdir(xy_dir), (0,1), info), lambda k,v: k[0]>0 and k[0]<=1000)
    fd_keys = slurm_split(sorted(fd.keys()))
    for index in fd_keys:
        tpc_fd = files_info(fd[index], (0,), tpc_info)
        if (tpcs_key,) in tpc_fd: continue
        Xs, Ys, Starts, Y_truths = [], [], [], []
        for tpc in tpcs:
            tpc = (tpc, )
            if tpc not in tpc_fd: 
                break
            with np.load(xy_dir+'/'+tpc_fd[tpc][0], allow_pickle=True) as xy_f:
                Xs.append(xy_f["X"])
                Ys.append(xy_f["Y"])
                Starts.append(xy_f["starts"])
                Y_truths.append(xy_f["Y_truth"])
        if len(Xs) != len(tpcs): continue
        out_f = tpc_fd[(tpcs[0],)][0].replace("TPC%d"%tpcs[0], "TPC%d"%tpcs_key)
        Transforms = [get_TPC_inverse_coords_transform(tpc, VOXEL_SIZE, R0)[0] for tpc in tpcs]
        num_events = len(Starts[0])
        X, Y, starts, Y_truth = [], [], [], [[] for _ in range(num_events)]
        for event_i in range(num_events):
            starts.append(len(X))
            for tpc in range(len(tpcs)):
                event_starti = Starts[tpc][event_i]
                if event_i == num_events-1:
                    X.extend(Transforms[tpc](Xs[tpc][event_starti:]))
                    Y.extend(Ys[tpc][event_starti:])
                else:
                    event_endi = Starts[tpc][event_i+1]
                    X.extend(Transforms[tpc](Xs[tpc][event_starti:event_endi]))
                    Y.extend(Ys[tpc][event_starti:event_endi])
                Y_truth[event_i].extend(Transforms[tpc](Y_truths[tpc][event_i]))
        np.savez_compressed(xy_dir+"/"+out_f, 
            X=np.array(X).astype(np.float32), Y=np.array(Y).astype(np.float32), starts=starts, Y_truth=Y_truth)

def downsample_y(downsample, xy_dir = sys.argv[1]):
    files = slurm_split(flatten_fd(filter_fd(files_info(os.listdir(xy_dir)), lambda k,v: k[0]>700 and k[0]<=701)))
    print(files)
    for f in files:
        out_f = "test-"+f
        val_downsample, map_downsample = [], []
        with np.load(sys.argv[1]+'/'+f, allow_pickle=True) as fi:
            event_starts = fi["starts"]
            pix_batch = fi["X"]
            Y_truth = fi["Y_truth"]
            for event in range(len(Y_truth)):
                event_starti = event_starts[event]
                if event == len(event_starts)-1:
                    event_coords = pix_batch[event_starti:, :3]
                else:
                    event_coords = pix_batch[event_starti:event_starts[event+1], :3]
                truth_dict = voxelize(Y_truth[event], downsample, False)
                truth_dict_index = {}
                for coord in event_coords:
                    coord = tuple(coord//downsample)
                    if coord in truth_dict and coord not in truth_dict_index:
                        truth_dict_index[coord] = len(val_downsample)
                        val_downsample.append(truth_dict[coord])
                for coord in event_coords:
                    coord = tuple(coord//downsample)
                    if coord in truth_dict_index:
                        map_downsample.append(truth_dict_index[coord])
                    else:
                        map_downsample.append(-1)
            np.savez_compressed(xy_dir+"/"+out_f, 
                X=pix_batch, Y=fi["Y"], starts=event_starts, Y_truth=Y_truth,
                Y_downsample=(np.array(val_downsample).astype(np.float32), np.array(map_downsample)))
                
if __name__ == "__main__":
    if len(sys.argv) >= 5:
        batch_size = int(sys.argv[4])
    else:
        batch_size = None
    process_xy(batch_size=batch_size)
    #process_xy_from_pix()
    #join_tpcs()
    #downsample_y((5, 5, 5))
