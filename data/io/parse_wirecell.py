import numpy as np
import json
from geom.util import voxelize
from data.util import *
import sys, os, glob
from geom.pdsp import get_TPC_coord_transform, VOXEL_SIZE

LARLENGTH = 10 #larsoft uses cm instead of mm
LARTIME = 0.001 #larsoft uses ns instead of us


def to_bee(cluster_file, bee_file):
    cmd = 'wirecell-img bee-blobs -s uniform -o %s %s' % (
        bee_file, cluster_file, )
    print(cmd)
    os.system(cmd)

def process(file_dir=sys.argv[1], parsed_dir=sys.argv[2], tpc=1, apa=0, parsed_prefix=None):
    print(file_dir)
    if parsed_prefix is None:
        parsed_prefix = file_dir[file_dir.rfind('/', 0, file_dir.rfind('/'))+1:file_dir.rfind('/')]
    index = file_dir[file_dir.rfind('/')+1:]
    out_file = parsed_dir+'/'+parsed_prefix+"_%s_wirecell-TPC%d"%(index, tpc)
    print(out_file)
    data = []
    transform = get_TPC_coord_transform(tpc, False)
    for event in range(len(os.listdir(file_dir))):
        bee_file = file_dir+'/%d/bee-apa%d.json'%(event, apa)
        to_bee(file_dir+'/%d/clusters-apa%d-0000.json'%(event, apa), bee_file)
        with open(bee_file, 'r') as f:
            event_data = json.load(f)
        pts = []
        for x, y, z, q in zip(event_data['x'], event_data['y'], event_data['z'], event_data['q']):
            coord = np.array((x, y, z))*LARLENGTH
            coord = transform(coord, 3)
            pts.append(np.concatenate((coord, (q,))))
        pts = np.array(pts)
        data.append(voxelize(pts, VOXEL_SIZE))
    np.savez_compressed(out_file, *data)

def process_multi(files_dir=sys.argv[1], parsed_dir=sys.argv[2], **kwargs):
    for file_dir in os.listdir(files_dir):
        #if "0009" not in file_dir: continue
        try:
            process(files_dir+'/'+file_dir, parsed_dir, **kwargs)
        except Exception:
            print(files_dir+'/'+file_dir, "failed")

if __name__ == "__main__":
    #process()
    process_multi(tpc=5, apa=2)
