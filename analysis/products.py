import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *

def parse_xy(event = 1, E_scale = 1, xy_file = sys.argv[1]):
    with np.load(xy_file, allow_pickle=True) as xy_f:
        pix_batch=xy_f["X"]
        event_starts=xy_f["starts"]
        if "Y_truth" in xy_f:
            voxel_truth = {tuple(pt[:3]):pt[3]*E_scale for pt in xy_f["Y_truth"][event]}
        else:
            voxel_truth = {}
    event_starti = event_starts[event]
    if event == len(event_starts)-1:
        event_endi = -1
        event_pixels = pix_batch[event_starti:]
    else:
        event_endi = event_starts[event+1]
        event_pixels = pix_batch[event_starti:event_endi]
    coords_active = event_pixels[:,:3]
    voxel_active = {tuple(pt[:3]):sum(pt[4:]) for pt in event_pixels}
    #voxel_active = {tuple(pt[:3]):pt[5] for pt in event_pixels}
    return voxel_truth, voxel_active, (event, event_starti, event_endi, coords_active)

def parse_xy_mc(event = 1, mcbeamcosmic_dir = None, xy_file = sys.argv[1]):
    with np.load(xy_file, allow_pickle=True) as xy_f:
        event_truth = xy_f["Y_truth"][event]
    mcbeamcosmic_dict = {}
    if mcbeamcosmic_dir is not None:
        index = file_info(xy_file)[1]
        mcbeamcosmic_file = files_info(os.listdir(mcbeamcosmic_dir), [1, 2])[(index, "mcbeamcosmic")][0]
        with np.load(mcbeamcosmic_dir+'/'+mcbeamcosmic_file, allow_pickle=True) as mcbeamcosmic_f:
            event_mcbeamcosmic = mcbeamcosmic_f["arr_%d"%event]
            for beam_trackid in event_mcbeamcosmic[0]:
                mcbeamcosmic_dict[beam_trackid] = "beam"
            for cosmic_trackid in event_mcbeamcosmic[1]:
                mcbeamcosmic_dict[cosmic_trackid] = "cosmic"
    clusters, clusters_val = {}, {}
    for pt in event_truth:
        mc = tuple(pt[4:])
        coord = pt[:3]
        val = pt[3]
        if mc in clusters:
            clusters[mc].append(coord)
            clusters_val[mc] += val
        else:
            clusters[mc] = [coord]
            clusters_val[mc] = val
    for mc, cluster in clusters.items():
        clusters[mc] = []
        size = len(cluster)
        clusters[mc].append(size)
        clusters[mc].append(track_stats(cluster))
        if mc[0] in mcbeamcosmic_dict:
            clusters[mc].append(mcbeamcosmic_dict[mc[0]])
        else:
            clusters[mc].append(None)
        clusters[mc].append(clusters_val[mc])
    voxel_mc = {}
    for pt in event_truth:
        mc = list(pt[4:])
        mc.extend(clusters[tuple(mc)])
        voxel_mc[tuple(pt[:3])] = tuple(mc)
    return voxel_mc

def get_clusters_mc(stats_file = None, event = 1, mcbeamcosmic_dir = None, xy_file = sys.argv[1]):
    if stats_file is not None:
        with np.load(stats_file, allow_pickle=True) as f:
            return f["%d_%d"%(file_info(xy_file)[1], event)].item()
    from scipy.spatial import KDTree
    _, voxel_active, _ = parse_xy(event, 1, xy_file)
    coords = list(voxel_active.keys())
    voxel_mc = parse_xy_mc(event, mcbeamcosmic_dir, xy_file)
    coords_mc = list(voxel_mc.keys())
    vals_mc = list(voxel_mc.values())
    kd_tree = KDTree(coords_mc)
    _, closest_i = kd_tree.query(coords, 1, 0)
    clusters = {val_mc : [[], []] for val_mc in vals_mc}
    for coord, closest in zip(coords, closest_i):
        #mc = mode1d([vals_mc[i] for i in closest])
        mc = vals_mc[closest]
        clusters[mc][0].append(coord)
    for coord in set(voxel_mc.keys()).difference(coords):
        clusters[voxel_mc[coord]][1].append(coord)
    return clusters

def mc_stats(mcbeamcosmic_dir = None, tpc=159, stats_dir = "mc", xy_dir = sys.argv[1]):
    MIN_CLUSTER = 20
    n_events = 5
    n_files = 100
    xy_files = [f for f in os.listdir(xy_dir) if tpc_info(f)[0]==tpc and file_info(f)[1] <= n_files]
    mc_file = stats_dir+'/'+xy_dir[xy_dir.rfind('/')+1:] + "-tpc%d-mc"%tpc
    print(mc_file)
    mc_dict = {}
    for f in xy_files:
        f = xy_dir+'/'+f
        for event in range(n_events):
            print("NEXT")
            key = "%d_%d"%(file_info(f)[1], event)
            clusters = get_clusters_mc(None, event, mcbeamcosmic_dir, f)
            mc_dict[key] = {k:v for k, v in clusters.items() if k[2] > MIN_CLUSTER}
    np.savez_compressed(mc_file, **mc_dict)

def compare_true_active(fig, fig_histo, fig_inf, fig_histo_inf, event = 1, downsample=(1,1,1), true_thres = 0, plot_slice=False, plot_lims=None, view_angle=None, xy_file = sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1/3, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)

    thres, plot_lims, evt_purity, evt_sensitivity, evt_voxels_comp, purity, sensitivity, voxels_comp_T, voxels_comp_inf, voxel_truth, voxel_active = \
            inference_analysis(voxel_truth, voxel_active, 0, plot_lims, downsample, plot_slice)

    evt_voxel_FN, evt_voxel_TP, _, evt_voxel_FP = evt_voxels_comp
    voxel_FN, voxel_TP, _, _ = voxels_comp_T
    x_lim, y_lim, z_lim = plot_lims

    if fig is not None:
        scatter_voxels_comp(fig, voxels_comp_T, "True [MeV]", "Active [MeV]", plot_slice, view_angle)
        fig.text(0, 0.95, "True Threshold: %.2f"%true_thres)
        fig.text(0., 0.85, "EVENT STATS: ")
        fig.text(0., 0.75, "E TP: %.2f, E FN: %.2f"%(sum(evt_voxel_TP.values()), sum(evt_voxel_FN.values())))
        fig.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig.text(0., 0.55, "E TP: %.2f, E FN: %.2f"%(sum(voxel_TP.values()), sum(voxel_FN.values())))
        fig.suptitle("True vs. Active [Energy]\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig.show()

    if fig_histo is not None:
        histo_voxels_comp(fig_histo, fig_histo.add_subplot(), voxels_comp_T, "True [MeV]", "Active [MeV]", 20)
        fig_histo.text(0.4, 0.8, "True Threshold: %.2f"%true_thres)
        fig_histo.suptitle("Energy Histo True vs. Active\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_histo.show()

    if fig_inf is not None:
        scatter_voxels_comp(fig_inf, voxels_comp_inf, "True [Channel Val]", "Active [Channel Val]", plot_slice, view_angle)
        fig_inf.text(0, 0.95, "True Threshold: %.2f"%true_thres)
        fig_inf.text(0., 0.85, "EVENT STATS: ")
        fig_inf.text(0., 0.75, "N TP: %d, N FN: %d, N FP: %d"%(len(evt_voxel_TP), len(evt_voxel_FN), len(evt_voxel_FP)))
        fig_inf.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig_inf.suptitle("True vs. Active [Channel Val]\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_inf.show()

    if fig_histo_inf is not None:
        ax_comp = fig_histo_inf.add_subplot()
        histo_voxels_comp(fig_histo_inf, ax_comp, voxels_comp_inf, "True [Channel Val]", "Active [Channel Val]", 20, log_yscale=True)
        fig_histo_inf.text(0.4, 0.8, "True Threshold: %.2f"%true_thres)
        fig_histo_inf.suptitle("Channel Val Histo \nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_histo_inf.show()

def plot_pixel_stats(event = 1, true_thres=0, xy_file=sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1/3, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)

    voxel_FN, voxel_T_active, _, voxel_FP = comp_voxels(voxel_truth, voxel_active)

    n_T = len(voxel_truth)
    n_TP_naive = len(voxel_T_active) #naive: treat all active as positive
    n_FP_naive = len(voxel_FP)

    xs, ys, thresholds = SP_curve(voxel_T_active, voxel_active, thresholds=np.arange(100)*0.1)
    plt.plot(xs, ys)
    plt.show()

def plot_mc_stats(fig, stats_file):
    ax = fig.add_subplot()
    TPRs, Thetas, Phis = [], [], []
    with np.load(stats_file, allow_pickle=True) as f:
        for event_mc in f.values():
            for mc, coords in event_mc.item().items():
                track_stats = mc[3]
                if track_stats is None: continue
                TPRs.append((mc[2]-len(coords[1]))/mc[2])
                Thetas.append(track_stats[0])
                Phis.append(track_stats[1])
    cmap=plt.get_cmap('viridis')
    fig.colorbar(ax.scatter(Thetas, Phis, s=5, c=TPRs, cmap=cmap))
    ax.text(0.1, 0.1, "N particles: %d"%len(TPRs))
    ax.set_xlabel("Theta (+z in reverse drift direction)")
    ax.set_ylabel("Phi (+y upwards, +x in beam direction)")
    ax.set_title("Tiling Sensitivity vs. Polar Angle \n stats file: %s"%stats_file[3:])
    plt.show()
                 
def plot_mc(fig, event = 1, mc_filter = lambda _:True, plot_all = True, mc_beamcosmic=None, stats_file=None, xy_file = sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1, xy_file)
    coords_active = set(voxel_active.keys())
    coords_truth = set(voxel_truth.keys())
    clusters_mc = get_clusters_mc(stats_file, event, mc_beamcosmic, xy_file)
    names, voxels_truth_mc, voxels_active_mc, voxel_missing, coords_mc = [], [], [], {}, set()
    i = 0
    for mc, coords in clusters_mc.items():
        coords = coords[0]
        if not coords or not mc_filter(mc): continue
        voxel_truth_mc, voxel_active_mc = filter_voxels_coords(coords, voxel_truth, voxel_active)
        names.append("%d: PDG=%d; N(T)=%d"%(i, mc[1], mc[2]))
        if mc[3] is not None:
            names[-1]+=", θ=%d, ϕ=%d"%(mc[3][0], mc[3][1])
        if mc_beamcosmic is not None:
            names[-1]+="; %s"%mc[4]
        voxels_truth_mc.append(voxel_truth_mc)
        voxels_active_mc.append(voxel_active_mc)
        voxel_missing.update(filter_voxels_coords(clusters_mc[mc][1], voxel_truth)[0])
        coords_mc.update(coords)
        i += 1
    colors = [None for _ in names]
    if plot_all:
        colors.append("gray")
        voxels_active_mc.append({c:None for c in coords_active.difference(coords_mc)})
        voxels_truth_mc.append({c:None for c in coords_truth.intersection(coords_active).difference(coords_mc)})
        names.append("Other Voxels; N (true)=%d"%len(voxels_truth_mc[-1]))
    ax_active = fig.add_subplot(211, projection="3d")
    scatter_voxels(fig, ax_active, voxels_active_mc, names, colors, "Active Voxels", size=1)
    ax_active.get_legend().remove()
    colors.append("black")
    voxels_truth_mc.append({c:None for c in coords_truth.difference(coords_active)} if plot_all else voxel_missing)
    names.append("Missing Voxels (True but not Active), N=%d"%len(voxels_truth_mc[-1]))
    scatter_voxels(fig, fig.add_subplot(212, projection="3d"), voxels_truth_mc, names, colors, "True & Active Voxels", size=1)
    fig.suptitle("MC Event Display \n xy file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
    plt.show()

def plot_xy(fig, event=0, E_scale=1, true_thres=0, active_thres=0, xy_file=sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, E_scale, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)
    voxel_active = filter_voxel_val(voxel_active, active_thres)
    coord_lims = get_voxels_lims(voxel_truth, voxel_active)
    scatter_voxel(fig, fig.add_subplot(211, projection="3d"), voxel_truth, "True Voxels", plot_lims=coord_lims)
    scatter_voxel(fig, fig.add_subplot(212, projection="3d"), voxel_active, "Active Voxels", plot_lims=coord_lims)
    fig.text(0.1, 0.75, "True Thres: %.2f [MeV]"%true_thres)
    fig.text(0.1, 0.5, "Purity: %.3f \n Sensitivity: %.3f"%SP_stats(voxel_truth, voxel_active)[:2])
    fig.text(0.1, 0.25, "Active Thres: %.2f [Channel Val]"%active_thres)
    plt.show()

def plot_spacepoint(fig, fig_comp=None, event=0, downsample=(1,1,1), sp_thres=0, E_scale=1, true_thres=0, plot_lims=None, spacepoint_file=sys.argv[1], truth_file=sys.argv[2]):
    with np.load(spacepoint_file) as f:
        voxel_spacepoint = {tuple(pt[:3]): pt[3] for pt in f["arr_%d"%event]}
    voxel_truth = {}
    if truth_file is not None:
        with np.load(truth_file) as f:
            voxel_truth = {tuple(pt[:3]): pt[3]*E_scale for pt in f["arr_%d"%event]}
    voxel_truth, voxel_spacepoint =  filter_voxels_coord(plot_lims, *downsample_voxels(downsample, voxel_truth, voxel_spacepoint, reduce=True))
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)
    coord_lims = get_voxels_lims(voxel_truth, voxel_spacepoint)
    voxel_spacepoint = filter_voxel_val(voxel_spacepoint, sp_thres)
    fig.text(0.1, 0.25, "SpacePoint Threshold: %.2f"%sp_thres)
    if truth_file is not None:
        scatter_voxel(fig, fig.add_subplot(211, projection="3d"), voxel_truth, "True Voxels", plot_lims=coord_lims)
        scatter_voxel(fig, fig.add_subplot(212, projection="3d"), voxel_spacepoint, "SpacePoint Voxels", plot_lims=coord_lims)
        fig.text(0.1, 0.75, "True Threshold: %.2f"%true_thres)
        fig.text(0.1, 0.5, "Purity: %.3f \n Sensitivity: %.3f"%SP_stats(voxel_truth, voxel_spacepoint)[:2])
    else:
        scatter_voxel(fig, fig.add_subplot(projection="3d"), voxel_spacepoint, "SpacePoint Voxels", plot_lims=coord_lims)
    fig.suptitle("SpacePoint \nfile: %s, Event: %d"%(spacepoint_file[spacepoint_file.rfind('/')+1:], event))
    if fig_comp is not None:
        scatter_voxels_comp(fig_comp, comp_voxels(voxel_truth, voxel_spacepoint), "True [MeV]", "SpacePoint")

def plot_channel(fig, event=0, x_lim=None, channel_file=sys.argv[1]):
    from geom.pdsp import get_APA_wireplane_maps
    with np.load(channel_file) as f:
        chan_vals = f["arr_%d"%event]
    tpc, = tpc_info(channel_file)
    maps = get_APA_wireplane_maps(tpc)
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(chan_vals[:, maps[i]], aspect='auto', origin='lower', vmax=50)
        ax.set_ylim(x_lim)
    
if __name__=="__main__":
    #for i in range(5):
        #compare_energys(i)
        #compare_channels(i)
        #compare_true_active(plt.figure(), i)
        #plot_pixel_stats(i)
    #plot_mc(plt.figure(), plot_all=True, mc_filter = lambda mc: mc[2]>30)
    #plot_mc(plt.figure(), plot_all=False, mc_filter = lambda mc: mc[2]>30)
    #mc_stats("/global/cscratch1/sd/ethanlu/larsim/reco_1GeV_BeamCosmic_parsed", 1)
    plot_spacepoint(plt.figure(), plt.figure(), 0, (8, 4, 4), true_thres=0.5/3)
    plt.show()

