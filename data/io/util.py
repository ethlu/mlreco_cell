import numpy as np
import os, sys, time
from geom.pdsp import get_APA_geom, VOXEL_SIZE, DOWNSAMPLE, get_TPC_inverse_coords_transform, get_TPC_box
from geom.util import voxelize
from data.util import *

def join_tpcs(product, tpcs=[1, 2], parsed_dir = sys.argv[1]):
    R0 = get_TPC_box(tpcs[0])[0]
    tpcs_key = int("".join(map(str, tpcs)))
    fd = filter_fd(files_info(os.listdir(parsed_dir), (1, 2)), lambda k,v: k[1]==product and k[0]>=301 and k[0]<=301)
    fd_keys = slurm_split(sorted(fd.keys()))
    print(fd)
    for index in fd_keys:
        tpc_fd = files_info(fd[index], (0,), tpc_info)
        if (tpcs_key,) in tpc_fd: continue
        Pts = []
        for tpc in tpcs:
            tpc = (tpc, )
            if tpc not in tpc_fd: break
            Pts.append([])
            with np.load(parsed_dir+'/'+tpc_fd[tpc][0], allow_pickle=True) as parsed_f:
                i = 0
                while "arr_%d"%i in parsed_f:
                    Pts[-1].append(parsed_f["arr_%d"%i])
                    i += 1
        if len(Pts) != len(tpcs): continue
        out_f = tpc_fd[(tpcs[0],)][0].replace("TPC%d"%tpcs[0], "TPC%d"%tpcs_key)
        #Transforms = [get_TPC_inverse_coords_transform(tpc, VOXEL_SIZE, R0, -250)[0] for tpc in tpcs]
        Transforms = [get_TPC_inverse_coords_transform(tpc, VOXEL_SIZE, R0, -6)[0] for tpc in tpcs]  #for SpacePointSolver
        Out = []
        for event_i in range(len(Pts[0])):
            data = []
            for tpc in range(len(tpcs)):
                data.extend(Transforms[tpc](Pts[tpc][event_i]))
            Out.append(np.array(data).astype(np.float32))
        np.savez_compressed(parsed_dir+"/"+out_f, *Out)

if __name__ == "__main__":
    #join_tpcs("depoElectron", (1,2,5,6))
    join_tpcs("depoElectron", (1, 2))
    
