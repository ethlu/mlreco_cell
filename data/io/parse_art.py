import os, sys
import ROOT
import numpy as np
from geom.util import voxelize

LARLENGTH = 10 #larsoft uses cm instead of mm
LARTIME = 0.001 #larsoft uses ns instead of us
data_dict = {
    "energy" : {
        "type" : "std::vector<sim::SimEnergyDeposit>",
        "type_p": ROOT.vector(ROOT.sim.SimEnergyDeposit),
        "tag" : ROOT.art.InputTag("IonAndScint"),
        "filename" : "g4.root",
        "proc": lambda _ : _
        },
    "wire": {
        "type" : "std::vector<recob::Wire>",
        "type_p": ROOT.vector(ROOT.recob.Wire),
        "tag": ROOT.art.InputTag("wclsdatanfsp", "gauss"),
        "filename" : "reco.root",
        "proc": lambda _ : _
        },
    "simchannel": {
        "type" : "std::vector<sim::SimChannel>",
        "type_p": ROOT.vector(ROOT.sim.SimChannel),
        "tag": ROOT.art.InputTag("tpcrawdecoder", "simpleSC"),
        "filename" : "reco.root",
        "proc": lambda _ : _
        }
    }

# Some functions that I find useful to reduce error-prone typing.
def read_header(h):
    """Make the ROOT C++ jit compiler read the specified header."""
    ROOT.gROOT.ProcessLine('#include "%s"' % h)

def provide_get_valid_handle(klass):
    """Make the ROOT C++ jit compiler instantiate the
       Event::getValidHandle member template for template
       parameter klass."""
    ROOT.gROOT.ProcessLine('template gallery::ValidHandle<%(name)s> gallery::Event::getValidHandle<%(name)s>(art::InputTag const&) const;' % {'name' : klass})

# Now for the script...
def setup(data_dict = data_dict):
    print ("Reading headers...")
    read_header('gallery/ValidHandle.h')

    print ("Instantiating member templates...")
    for v in data_dict.values():
        provide_get_valid_handle(v["type"])

def process_reco(in_dir = sys.argv[1], data_dict = data_dict):
    reco_files = [f for f in os.listdir(in_dir) if "reco" in f]
    for f in reco_files:
        filename = ROOT.vector(ROOT.string)(1, in_dir+'/'+f)
        ev = ROOT.gallery.Event(filename)
        for v in data_dict.values():
            v["handle"] = ev.getValidHandle(v["type_p"])
        while (not ev.atEnd()) :
            for v in data_dict.values():
                v["proc"](v["handle"](v["tag"]))
            ev.next()
        for v in data_dict.values():
            v["proc"](f.replace(".root", ""))

def process(dir_name = sys.argv[1], root_dir = sys.argv[2], data_dict = data_dict):
    for v in data_dict.values():
        filename = ROOT.vector(ROOT.string)(1, dir_name+v["filename"])
        ev = ROOT.gallery.Event(filename)
        handle = ev.getValidHandle(v["type_p"])
        while (not ev.atEnd()) :
            v["proc"](handle(v["tag"]))
            ev.next()
        v["proc"](os.path.relpath(dir_name, root_dir).replace("/", "_"))

def process_multi(root_dir = sys.argv[1]):
    for root, dirs, files in os.walk(root_dir):
        if not "reco.root" in files: continue
        process(root+"/", root_dir)

def proc_energy_factory(coord_transform=None, voxel_size=None, output_suffix="energy", output_dir=sys.argv[2]):
    data = []
    if voxel_size is not None:
        voxel_size = np.array(voxel_size)
    def proc_energy(inp):
        nonlocal data
        if isinstance(inp, str):
            inp = "" if inp == "." else inp
            np.savez_compressed(output_dir+inp+"_"+output_suffix, *data) 
            data.clear()
        elif not inp.empty():
            if voxel_size is None:
                event_data = []
            else:
                voxels = {}
            for depo in inp.product():
                midpoint = np.array((depo.MidPointX(), depo.MidPointY(), depo.MidPointZ()))*LARLENGTH
                if coord_transform is not None:
                    midpoint = coord_transform(midpoint, depo.Time()*LARTIME)
                    if midpoint is None: continue
                val = np.array([depo.Energy()])
                #val = np.array([depo.NumElectrons()])
                if voxel_size is None:
                    event_data.append(list(np.concatenate((midpoint, val))))
                else:
                    vox_i = tuple(midpoint//voxel_size)
                    try:
                        voxels[vox_i] += val
                    except KeyError:
                        voxels[vox_i] = val
            if voxel_size is not None:
                event_data = [list(np.concatenate((k, v))) for k, v in voxels.items()]
            data.append(np.array(event_data))
    return proc_energy

def proc_wire_factory(channels=None, output_suffix="wire", output_dir=sys.argv[2]):
    """ASSUMES the vector is in order of channel number"""
    data = []
    def proc_wire(inp):
        nonlocal data
        if isinstance(inp, str):
            inp = "" if inp == "." else inp
            np.savez_compressed(output_dir+inp+"_"+output_suffix, *data) 
            data.clear()
        elif not inp.empty():
            if channels is None:
                data.append(np.array([np.array(wire.Signal()) for wire in inp.product()]).T)
            else:
                ch_vals = inp.product()
                data.append(np.array([np.array(ch_vals[ch].Signal()) for ch in channels]).T)
    return proc_wire

def proc_simchan_factory(channels, coord_transform, voxel_size=None, output_suffix="simchan", output_dir=sys.argv[2]):
    data = []
    def proc_simchan(inp):
        nonlocal data
        if isinstance(inp, str):
            inp = "" if inp == "." else inp
            np.savez_compressed(output_dir+inp+"_"+output_suffix, *data) 
            data.clear()
        elif not inp.empty():
            pts = set()
            for i in channels:
                chan = inp.product()[i]
                for tick, edep in chan.TDCIDEMap():
                    for e in edep:
                        point = np.array([e.x, e.y, e.z])*LARLENGTH
                        coord = coord_transform(point, 0)
                        if coord is None:
                            continue
                        pts.add(tuple(coord)+(e.energy,))
            pts = np.array(list(pts))
            if voxel_size is None:
                data.append(pts)
            else:
                data.append(voxelize(pts, voxel_size))
    return proc_simchan

def proc_simchan_driftvtick_factory(channels, coord_transform):
    def proc_simchan(inp):
        X = []
        for i in channels:
            chan = inp.product()[i]
            for tick, edep in chan.TDCIDEMap():
                for e in edep:
                    point = np.array([e.x, e.y, e.z])*LARLENGTH
                    coord = coord_transform(point, 0)
                    if coord is None:
                        continue
                    X.append([tick, coord[0]])

        X.sort(key=lambda x: x[0])
        print("X", X[:10])
        print("X last", X[-10:])
        X=np.array(X).T
        z = np.polyfit(*X, 2, full=True)
        print(z)
    return proc_simchan

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from geom.pdsp import get_TPC_coord_transform, get_APA_chans, VOXEL_SIZE
    TPC_NUM = 1

    #proc_energy = proc_energy_factory(get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "energy-E-TPC%d"%TPC_NUM)
    #data_dict["energy"]["proc"]=proc_energy
    data_dict.pop("energy")
    proc_wire = proc_wire_factory(get_APA_chans(TPC_NUM), "wire-TPC%d"%TPC_NUM)
    data_dict["wire"]["proc"]=proc_wire
    proc_simchan = proc_simchan_factory(get_APA_chans(TPC_NUM), get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "energy-TPC%d"%TPC_NUM)
    data_dict["simchannel"]["proc"]=proc_simchan
    setup()
    process_reco()
    
