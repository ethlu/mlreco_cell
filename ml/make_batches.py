import numpy as np
import os, sys, time
from geom.pdsp import get_APA_geom
from geom.util import voxelize

DOWNSAMPLE = (8, 4, 4)

def process_xy(downsample = DOWNSAMPLE, tpc_num = 1, batch_size = int(sys.argv[3]), parsed_dir = sys.argv[1], out_dir = sys.argv[2]):
    from tiling.pixel import Pixelator, Geom
    parsed_files = [f for f in os.listdir(parsed_dir) if "TPC%d"%tpc_num in f]
    energy_files = [f for f in parsed_files if "energy" in f]
    geom = get_APA_geom(tpc_num)
    pix = Pixelator(geom)
    pix = pix.to_numba()
    batch_num = 0
    for ef in energy_files:
        cf = ef.replace("energy", "wire")
        if cf not in parsed_files:
            print(ef + " has no associated channel file")
            continue
        print("processing " + ef)
        with np.load(parsed_dir+'/'+ef) as ef_vals:
            ef_event_vals=[]
            i = 0
            while "arr_%d"%i in ef_vals:
                ef_event_vals.append(ef_vals["arr_%d"%i])
                i += 1
        with np.load(parsed_dir+'/'+cf) as cf_vals:
            cf_event_vals=[]
            i = 0
            while "arr_%d"%i in cf_vals:
                cf_event_vals.append(cf_vals["arr_%d"%i])
                i += 1
        assert len(ef_event_vals) == len(cf_event_vals), "energy/channel files mismatch"
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = len(ef_event_vals)//batch_size
        ef_batches = np.split(np.array(ef_event_vals), num_batches)
        cf_batches = np.split(np.array(cf_event_vals), num_batches)
        for batch in range(num_batches):
            out_f = "batch%d-"%batch_num + ef.replace("energy", "xy")
            ef_batch = ef_batches[batch]
            cf_batch = cf_batches[batch]
            pix_batch = pix(cf_batch)
            pix_batch = Pixelator.downsamples(pix_batch, (downsample[0], downsample[2], downsample[1]))
            truth_dict_batch = [voxelize(truth_event, downsample, False) for truth_event in ef_batch]
            Y_truth = np.array([[np.concatenate((k, v)) for k, v in voxels.items()] for voxels in truth_dict_batch])
            X, Y, event_starts= generate_xy(pix_batch, truth_dict_batch)
            np.savez_compressed(out_dir+"/"+out_f, X=X, Y=Y, starts=event_starts, Y_truth=Y_truth)
            batch_num += 1

def generate_xy(pix_batch, truth_dict_batch):
    num_events = len(pix_batch)
    num_slices = len(pix_batch[0])
    num_truths = len(list(truth_dict_batch[0].values())[0])
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
                    truth_vals = truth_dict[coord]
                else:
                    truth_vals = non_active_truth
                X.append(np.concatenate([coord, (event_i,), vals]))
                Y.append(truth_vals)
    return np.array(X), np.array(Y), np.array(event_starts)

if __name__ == "__main__":
    process_xy()
