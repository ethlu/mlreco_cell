import os, sys
import ROOT
import numpy as np
from geom.util import voxelize
from data.util import *

LARLENGTH = 10 #larsoft uses cm instead of mm
LARTIME = 0.001 #larsoft uses ns instead of us

evt_procs = []
product_dict = {
    "depo" : {
        "type" : "std::vector<sim::SimEnergyDeposit>",
        "type_p": ROOT.vector(ROOT.sim.SimEnergyDeposit),
        "tag" : ROOT.art.InputTag("IonAndScint"),
        "artkey" : "reco",
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
    "spacepoint": {
        "type" : "std::vector<recob::SpacePoint>",
        "type_p": ROOT.vector(ROOT.recob.SpacePoint),
        "tag": ROOT.art.InputTag("reco3d"),
        "artkey" : "reco",
        "proc": lambda _ : False
        },
    "mcparticle": {
        "type" : "std::vector<simb::MCParticle>",
        "type_p": ROOT.vector(ROOT.simb.MCParticle),
        "tag": ROOT.art.InputTag("largeant"),
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

def process(art_file, art_dir, product_dict, evt_procs):
    parsed_prefix = os.path.relpath(art_file, art_dir).replace("/", "_")
    prod_need_proc, evt_procs_need_proc = [], []
    for v in product_dict.values():
        if v["artkey"] in parsed_prefix and v["proc"](parsed_prefix):
            prod_need_proc.append(v)
    for proc in evt_procs:
        if proc(parsed_prefix):
            evt_procs_need_proc.append(proc)
    if not prod_need_proc and not evt_procs_need_proc: return
    filename = ROOT.vector(ROOT.string)(1, art_file) 
    ev = ROOT.gallery.Event(filename)
    for v in prod_need_proc:
        v["handle"] = ev.getValidHandle(v["type_p"])
    while (not ev.atEnd()):
        for v in prod_need_proc:
            v["proc"](v["handle"](v["tag"]))
        for proc in evt_procs_need_proc:
            proc(ev)
        ev.next()
    for v in prod_need_proc:
        v["proc"]("END")
    for proc in evt_procs_need_proc:
        proc("END")

def process_multi(product_dict = product_dict, evt_procs = evt_procs, art_dir = sys.argv[1]):
    done_fd = {}
    #done_fd = filter_fd(files_info(os.listdir(sys.argv[2])), lambda k,v: len(v)>=20) 
    for root, dirs, files in os.walk(art_dir):
        art_fd = files_info(files)
        #art_fd = filter_fd(files_info(files), lambda k,v: k[0]>=301 and k[0]<=302)
        f_filtered = flatten_fd(diff_fd(art_fd, done_fd))
        files = slurm_split(f_filtered)
        print(files)
        for f in files:
            process(root+'/'+f, art_dir, product_dict, evt_procs)

def proc_factory(proc_event, output_suffix, overwrite=False, evt_proc=False, output_dir=sys.argv[2]):
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
                return False if (not overwrite and output_file in done_files) else True
        elif evt_proc:
            data.append(proc_event(inp))
        elif not inp.empty():
            data.append(proc_event(inp.product()))
    return proc

def proc_wire_factory(channels, chan_transform = None, *outputargs):
    """ASSUMES the vector is in order of channel number"""
    def proc_wire(event):
        if channels is None:
            return np.array([np.array(wire.Signal()) for wire in event]).T
        chan_vals = []
        for i, ch in enumerate(channels):
            vals = np.array(event[ch].Signal())
            if chan_transform is not None:
                vals = chan_transform(i, vals)
            chan_vals.append(vals)
        return np.array(chan_vals).T
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
            #vals = np.array([depo.Energy(), depo.TrackID(), depo.PdgCode()]) #for depo
            vals = np.array([depo.NumElectrons()]) #for depoElectron
            pts.append(np.concatenate((midpoint, vals)))
        if voxel_size is None:
            return pts
        return voxelize(pts, voxel_size)
    return proc_factory(proc_depo, *outputargs)

def proc_spacepoint_factory(tpc_min=1, tpc_max=2, *outputargs):
    from geom.pdsp import get_TPC_box, DRIFT_SPEED, SIM_T0
    def proc_spacepoint(ev):
        spacepoints = ev.getValidHandle(ROOT.vector(ROOT.recob.SpacePoint))(ROOT.art.InputTag("reco3d")).product()
        charges = ev.getValidHandle(ROOT.vector(ROOT.recob.PointCharge))(ROOT.art.InputTag("reco3d")).product()
        pts = []
        for point, charge in zip(spacepoints, charges):
            coord = point.position()
            coord = np.array((coord.x(), coord.y(), coord.z()))*LARLENGTH
            r_min = get_TPC_box(tpc_min)[0]
            r_min_full = list(r_min)
            r_min_full[0] -= SIM_T0*DRIFT_SPEED
            r_max = get_TPC_box(tpc_max)[1]
            if any(coord > r_max) or any(coord < r_min_full): 
                continue
            coord -= r_min
            #vals = np.array([point.Chisq()])
            #vals = np.empty(0)
            vals = np.array(charge.charge()).reshape(1)
            if not charge.hasCharge():
                print("no charge: ", vals)
            pts.append(np.concatenate((coord, vals)))
        return voxelize(pts, VOXEL_SIZE)
    return proc_factory(proc_spacepoint, *outputargs, evt_proc=True)

def proc_calorimetry_factory(coord_transform, tpcs=None, *outputargs):
    import re
    def proc_calorimetry(ev):
        calo_dict = {
            "Track": ev.getValidHandle(ROOT.vector(ROOT.anab.Calorimetry))(ROOT.art.InputTag("pandoracalonosce")).product(),
            "Shower": ev.getValidHandle(ROOT.vector(ROOT.anab.Calorimetry))(ROOT.art.InputTag("pandoraShowercalonosce")).product() 
            }
        """
        calo_dict = {
            "Track": ev.getValidHandle(ROOT.vector(ROOT.anab.Calorimetry))(ROOT.art.InputTag("pandoracalo")).product(),
            "Shower": ev.getValidHandle(ROOT.vector(ROOT.anab.Calorimetry))(ROOT.art.InputTag("pandoraShowercalo")).product() 
            }
        """
        ev_pts = []
        for name, calos in calo_dict.items():
            for calo in calos:
                planeID = re.findall("C:(\d+) T:(\d+) P:(\d+)", calo.PlaneID().toString())[0]
                if tpcs is not None and int(planeID[1]) not in tpcs: continue
                pts = []
                for coord, dQdx, dEdx, residual in zip(calo.XYZ(), calo.dQdx(), calo.dEdx(), calo.ResidualRange()):
                    coord = coord_transform(np.array([coord.x(), coord.y(), coord.z()])*LARLENGTH, 0)
                    if coord is None:
                        continue
                    pts.append(np.concatenate((coord, [dQdx, dEdx, residual])))
                ev_pts.append(np.array([name, int(planeID[2]), calo.KineticEnergy()]+voxelize(pts, VOXEL_SIZE, True, 0, 3).tolist(), 
                    dtype=object))
        return ev_pts
    return proc_factory(proc_calorimetry, *outputargs, evt_proc=True)

def proc_mcparticle_factory(coord_transform, *outputargs):
    def proc_mcparticle(event):
        ptcls = []
        for ptcl in event:
            start = ptcl.Position()
            start_coord = np.array((start.X(), start.Y(), start.Z()))*LARLENGTH
            start_coord = coord_transform(start_coord, ptcl.T()*LARTIME)//VOXEL_SIZE
            end = ptcl.Position()
            end_coord = np.array((end.X(), end.Y(), end.Z()))*LARLENGTH
            end_coord = coord_transform(end_coord, ptcl.T()*LARTIME)//VOXEL_SIZE
            ptcls.append(np.concatenate(([ptcl.TrackId(), ptcl.NumberTrajectoryPoints()], start_coord, end_coord)))
        return np.array(ptcls)
    return proc_factory(proc_mcparticle, *outputargs)

def proc_mcbeamcosmic_factory(*outputargs):
    """ Determines if particle is from cosmic or beam, only used for BeamCosmic data"""
    def proc_mc(ev):
        tag_beam = ROOT.art.InputTag("generator")
        tag_cosmic = ROOT.art.InputTag("cosmicgenerator")
        tag_assns = ROOT.art.InputTag("largeant")
        findMaker = ROOT.gallery.FindMaker()

        beam_trackids = []
        handle = ROOT.gallery.Handle(ROOT.vector(ROOT.simb.MCTruth))()
        ev.getByLabel(tag_beam, handle)
        findMany = findMaker.makeFindMany(ROOT.simb.MCParticle, ROOT.sim.GeneratedParticleInfo, ROOT.gallery.Handle(ROOT.vector(ROOT.simb.MCTruth)))(handle, ev, tag_assns)
        for ptcl in findMany.at(0):
            beam_trackids.append(ptcl.TrackId())

        cosmic_trackids = []
        handle = ROOT.gallery.Handle(ROOT.vector(ROOT.simb.MCTruth))()
        ev.getByLabel(tag_cosmic, handle)
        findMany = findMaker.makeFindMany(ROOT.simb.MCParticle, ROOT.sim.GeneratedParticleInfo, ROOT.gallery.Handle(ROOT.vector(ROOT.simb.MCTruth)))(handle, ev, tag_assns)
        for ptcl in findMany.at(0):
            cosmic_trackids.append(ptcl.TrackId())
        return np.array((beam_trackids, cosmic_trackids))
    return proc_factory(proc_mc, *outputargs, evt_proc=True)

if __name__ == "__main__":
    from geom.pdsp import get_TPC_coord_transform, get_TPC_chans, get_APA_chan_transform, VOXEL_SIZE
    TPC_NUM = 1

    proc_depo = proc_depo_factory(get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "depoElectron-TPC%d"%TPC_NUM)
    #proc_depo = proc_depo_factory(get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "depo-TPC%d"%TPC_NUM) #"depo" contains energy values and some additional info. Need to modify the code above as indicated
    product_dict["depo"]["proc"]=proc_depo

    proc_wire = proc_wire_factory(get_TPC_chans(TPC_NUM), get_APA_chan_transform(TPC_NUM), "wire-TPC%d"%TPC_NUM)
    product_dict["wire"]["proc"]=proc_wire

    #proc_hit = proc_hit_factory(get_TPC_chans(TPC_NUM), "hit-TPC%d"%TPC_NUM)
    #product_dict["hit"]["proc"]=proc_hit

    #proc_simchan = proc_simchan_factory(get_TPC_chans(TPC_NUM), get_TPC_coord_transform(TPC_NUM), VOXEL_SIZE, "energy-TPC%d"%TPC_NUM)
    #product_dict["simchan"]["proc"]=proc_simchan

    #proc_mcparticle = proc_mcparticle_factory(get_TPC_coord_transform(1, False), "mcparticle")
    #product_dict["mcparticle"]["proc"] = proc_mcparticle

    proc_spacepoint = proc_spacepoint_factory(1, 2, "spacepoint-TPC12")
    #proc_spacepoint = proc_spacepoint_factory(1, 10, "spacepoint-TPC1*", False) 
    evt_procs.append(proc_spacepoint)

    proc_calo = proc_calorimetry_factory(get_TPC_coord_transform(TPC_NUM), [1, 5, 9], "calonosce-TPC%d"%TPC_NUM)
    #proc_calo = proc_calorimetry_factory(get_TPC_coord_transform(TPC_NUM, coord_lims=False), [1, 5, 9], "calonosce-TPC%d*"%TPC_NUM)
    evt_procs.append(proc_calo)

    #evt_procs.append(proc_mcbeamcosmic_factory("mcbeamcosmic")) 
    setup()
    process_multi()
    
