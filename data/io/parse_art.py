import os, sys
import ROOT
import numpy as np
from geom.util import voxelize

LARLENGTH = 10 #larsoft uses cm instead of mm
LARTIME = 0.001 #larsoft uses ns instead of us
product_dict = {
    "depo" : {
        "type" : "std::vector<sim::SimEnergyDeposit>",
        "type_p": ROOT.vector(ROOT.sim.SimEnergyDeposit),
        "tag" : ROOT.art.InputTag("IonAndScint"),
        "artkey" : "g4",
        "proc": lambda _ : False
        },
    "wire": {
        "type" : "std::vector<recob::Wire>",
        "type_p": ROOT.vector(ROOT.recob.Wire),
        "tag": ROOT.art.InputTag("wclsdatanfsp", "gauss"),
        "artkey" : "reco",
        "proc": lambda _ : False 
        },
    "simchan": {
        "type" : "std::vector<sim::SimChannel>",
        "type_p": ROOT.vector(ROOT.sim.SimChannel),
        "tag": ROOT.art.InputTag("tpcrawdecoder", "simpleSC"),
        "artkey" : "reco",
        "proc": lambda _ : False
        },
    "hit": {
        "type" : "std::vector<recob::Hit>",
        "type_p": ROOT.vector(ROOT.recob.Hit),
        "tag": ROOT.art.InputTag("gaushit"),
        "artkey" : "reco",
        "proc": lambda _ : False
        },
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
def setup(product_dict = product_dict):
    print ("Reading headers...")
    read_header('gallery/ValidHandle.h')

    print ("Instantiating member templates...")
    for v in product_dict.values():
        provide_get_valid_handle(v["type"])

def process(art_file, art_dir, product_dict):
    parsed_prefix = os.path.relpath(art_file, art_dir).replace("/", "_")
    prod_need_proc = []
    for v in product_dict.values():
        if v["artkey"] in parsed_prefix and v["proc"](parsed_prefix):
            prod_need_proc.append(v)
    if not prod_need_proc: return
    filename = ROOT.vector(ROOT.string)(1, art_file) 
    ev = ROOT.gallery.Event(filename)
    for v in prod_need_proc:
        v["handle"] = ev.getValidHandle(v["type_p"])
    while (not ev.atEnd()):
        for v in prod_need_proc:
            v["proc"](v["handle"](v["tag"]))
        ev.next()
    for v in prod_need_proc:
        v["proc"]("END")

def process_multi(product_dict = product_dict, art_dir = sys.argv[1]):
    from data.util import KEYS
    for root, dirs, files in os.walk(art_dir):
        for f in files:
            if KEYS["MU"](f) > 200: continue
            process(root+'/'+f, art_dir, product_dict)

def proc_factory(proc_event, output_suffix, output_dir=sys.argv[2]):
    done_files = set(os.listdir(output_dir))
    data = []
    output_file = None
    def proc(inp):
        nonlocal data, output_file
        if isinstance(inp, str):
            if inp == "END":
                np.savez_compressed(output_dir+output_file, *data) 
                data.clear()
            else:
                output_file = inp.replace(".root", "")+"_"+output_suffix+".npz"
                return False if output_file in done_files else True
        elif not inp.empty():
            data.append(proc_event(inp.product()))
    return proc

def proc_wire_factory(channels, *outputargs):
    """ASSUMES the vector is in order of channel number"""
    def proc_wire(event):
        if channels is None:
            return np.array([np.array(wire.Signal()) for wire in event]).T
        else:
            return np.array([np.array(event[ch].Signal()) for ch in channels]).T
    return proc_factory(proc_wire, *outputargs)

def proc_hit_factory(channels, *outputargs, Nticks=6000):
    def gauss(start, end, mean, sd, amp):
        pdf = lambda x: 1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean)/sd)**2)
        x = np.arange(start, end)
        return pdf(x)*amp

    def proc_hit(event):
        chan_vals = np.zeros((len(channels), Nticks))
        for hit in event:
            channel = hit.Channel()
            if channel not in channels: continue
            channel -= channels[0]
            start, end = hit.StartTick(), hit.EndTick()
            chan_vals[channel][start: end] = \
                gauss(start, end, hit.PeakTime(), hit.RMS(), hit.PeakAmplitude())
        return chan_vals.T
    return proc_factory(proc_hit, *outputargs)

def proc_simchan_factory(channels, coord_transform, voxel_size=None, *outputargs):
    def proc_simchan(event):
        pts = []
        for i in channels:
            chan = event[i]
            for tick, edep in chan.TDCIDEMap():
                for e in edep:
                    point = np.array([e.x, e.y, e.z])*LARLENGTH
                    coord = coord_transform(point, 0)
                    if coord is None:
                        continue
                    vals = np.array([e.energy])
                    pts.append(np.concatenate((coord, vals)))
        pts = np.array(pts)
        if voxel_size is None:
            return pts
        return voxelize(pts, voxel_size)
    return proc_factory(proc_simchan, *outputargs)

def proc_depo_factory(coord_transform, voxel_size=None, *outputargs):
    if voxel_size is not None:
        voxel_size = np.array(voxel_size)
    def proc_depo(event):
        pts = []
        for depo in event:
            midpoint = np.array((depo.MidPointX(), depo.MidPointY(), depo.MidPointZ()))*LARLENGTH
            midpoint = coord_transform(midpoint, depo.Time()*LARTIME)
            if midpoint is None: continue
            vals = np.array([depo.Energy()])
            #val = np.array([depo.NumElectrons()])
            pts.append(np.concatenate((midpoint, vals)))
        if voxel_size is None:
            return pts
        return voxelize(pts, voxel_size)
    return proc_factory(proc_depo, *outputargs)

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
    from geom.pdsp import get_TPC_coord_transform, get_APA_chans, VOXEL_SIZE
    TPC_NUM = 1

    #proc_energy = proc_depo_factory(get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "depo-TPC%d"%TPC_NUM)
    #product_dict["depo"]["proc"]=proc_energy
    #product_dict.pop("energy")
    proc_wire = proc_wire_factory(get_APA_chans(TPC_NUM), "wire-TPC%d"%TPC_NUM)
    product_dict["wire"]["proc"]=proc_wire
    proc_hit = proc_hit_factory(get_APA_chans(TPC_NUM), "hit-TPC%d"%TPC_NUM)
    product_dict["hit"]["proc"]=proc_hit
    proc_simchan = proc_simchan_factory(get_APA_chans(TPC_NUM), get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "energy-TPC%d"%TPC_NUM)
    product_dict["simchan"]["proc"]=proc_simchan
    setup()
    process_multi()
    
