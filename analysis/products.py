import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *

def parse_xy(event = 1, E_scale = 1, xy_file = sys.argv[1]):
    with np.load(xy_file, allow_pickle=True) as xy_f:
        pix_batch=xy_f["X"]
        energys_truth = xy_f["Y_truth"]
        event_starts=xy_f["starts"]
    event_starti = event_starts[event]
    if event == len(event_starts)-1:
        event_endi = -1
        event_pixels = pix_batch[event_starti:]
    else:
        event_endi = event_starts[event+1]
        event_pixels = pix_batch[event_starti:event_endi]
    coords_active = event_pixels[:,:3]
    voxel_active = {tuple(pt[:3]):sum(pt[4:]) for pt in event_pixels}
    event_truth = energys_truth[event]
    voxel_truth = {tuple(pt[:3]):pt[3]*E_scale for pt in event_truth}
    return voxel_truth, voxel_active, (event_starti, event_endi, coords_active)


def compare_energys(event = 3, simchan_file = sys.argv[1], energy_file=sys.argv[2]):
    with np.load(simchan_file) as batch_simchans:
        event_simchans = batch_simchans["arr_%d"%event]
    with np.load(energy_file) as batch_energys:
        event_energys = batch_energys["arr_%d"%event]
    sub = 1
    active_energys = {(pt[0]//sub, pt[1]//sub, pt[2]//sub):pt[3]  for pt in event_energys}
    total_E = sum(active_energys.values())
    print(sorted(active_energys.values())[100:120])
    print(sorted(active_energys.values())[-1000:-980])
    print("total E", total_E)
    active_E_coords = set(active_energys.keys())
    print("active energys: ",len(active_E_coords))
    active_simchans = {(pt[0]//sub, pt[1]//sub, pt[2]//sub):pt[3] for pt in event_simchans}
    """
    for i, slic in enumerate(event_pixels):
        for pt in slic:
            active_pixels.append((i//sub, int(pt[1])//sub, int(pt[0])//sub))
            """
    active_simchans_coords = set(active_simchans.keys())
    total_E_simchans = sum(active_simchans.values())
    print("total simchans E ", total_E_simchans)
    print("active simchans: ",len(active_simchans_coords))
    intersect = active_E_coords.intersection(active_simchans_coords)
    for i in intersect:
        pass
        #print(active_simchans[i]/active_energys[i])
    print("intersect: ", len(intersect))
    missing = active_E_coords.difference(intersect)
    missing_E = sum([active_energys[coord] for coord in missing])
    print("missing E ", missing_E)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    intersect = np.array([list(pt) for pt in intersect])
    missing = np.array([list(pt) for pt in missing])

    ax = plt.subplot(211, projection="3d")
    if len(intersect) > 0:
        ax.scatter3D(*intersect.T)
    if len(missing) > 0:
        ax.scatter3D(*missing.T)
    ax2 = plt.subplot(212, projection="3d")
    ax2.scatter3D(*np.array(list(active_simchans_coords)).T)
    plt.show()

def compare_channels(event = 10, wire_file = sys.argv[1], hit_file=sys.argv[2]):
    with np.load(wire_file) as batch_wires:
        event_wires = batch_wires["arr_%d"%event].T
    with np.load(hit_file) as batch_hits:
        event_hits = batch_hits["arr_%d"%event].T

    for i, (w, h) in enumerate(zip (event_wires, event_hits)):
        if i%20 or sum(w) < 10: 
            print("pass")
            continue
        print(i)
        ax = plt.subplot(211)
        ax.plot(w)
        ax2 = plt.subplot(212)
        ax2.plot(h)
        ax2.set_xlim(4000, 5000)
        plt.show()

def compare_active_pixels(event = 10, pixel_file = sys.argv[1], energy_file=sys.argv[2]):
    with np.load(pixel_file) as batch_pixels_f:
        batch_pixels=batch_pixels_f["arr_0"]
        event_starts=batch_pixels_f["arr_1"]
        print(event_starts)
        event_pixels = np.array(batch_pixels[event_starts[event]:event_starts[event+1]])
    with np.load(energy_file) as batch_energys:
        event_energys = batch_energys["arr_%d"%event]
    sub = 1
    active_energys = {(pt[0]//sub, pt[1]//sub, pt[2]//sub):pt[3]  for pt in event_energys}
    total_E = sum(active_energys.values())
    print("total E", total_E)
    active_E_coords = set(active_energys.keys())
    print("active energys: ",len(active_E_coords))
    active_pixels = [(pt[0]//sub, pt[2]//sub, pt[1]//sub) for pt in event_pixels[:, :3]]
    """
    for i, slic in enumerate(event_pixels):
        for pt in slic:
            active_pixels.append((i//sub, int(pt[1])//sub, int(pt[0])//sub))
            """
    active_pixels = set(active_pixels)
    print("active pixels: ",len(active_pixels))
    intersect = active_E_coords.intersection(active_pixels)
    print("intersect: ", len(intersect))
    missing = active_E_coords.difference(intersect)
    missing_E = sum([active_energys[coord] for coord in missing])
    print("missing E ", missing_E)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    intersect = np.array([list(pt) for pt in intersect])
    missing = np.array([list(pt) for pt in missing])
    active_pixels = np.array([list(pt) for pt in active_pixels])
    ax = plt.subplot(211, projection="3d")
    #ax = plt.axes(projection="3d")
    ax.scatter3D(*intersect.T)
    ax.scatter3D(*missing.T)
    ax2 = plt.subplot(212, projection="3d")
    ax2.scatter3D(*active_pixels.T)
    plt.show()

def compare_true_active(fig, fig_histo, fig_inf, fig_histo_inf, event = 1, downsample=(1,1,1), true_thres = 0, plot_slice=False, plot_lims=None, view_angle=None, plot_histo=True, xy_file = sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1/3, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres, False)

    thres, plot_lims, evt_purity, evt_sensitivity, evt_voxels_comp, purity, sensitivity, voxels_comp_T, voxels_comp_inf = \
            inference_analysis(voxel_truth, voxel_active, 0, plot_lims, plot_slice)

    evt_voxel_FN, evt_voxel_TP, _, evt_voxel_FP = evt_voxels_comp
    voxel_FN, voxel_TP, _, voxel_FP = voxels_comp_T
    x_lim, y_lim, z_lim = plot_lims

    if fig is not None:
        scatter_voxels_comp(fig, voxels_comp_T, "True [MeV]", "Active [MeV]", plot_slice, view_angle)
        fig.text(0, 0.95, "True Threshold: %f"%true_thres)
        fig.text(0., 0.85, "EVENT STATS: ")
        fig.text(0., 0.75, "E TP: %.2f, E FP: %.2f, E FN: %.2f"%(voxel_sum(evt_voxel_TP), voxel_sum(evt_voxel_FP), voxel_sum(evt_voxel_FN)))
        fig.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig.text(0., 0.55, "E TP: %.2f, E FP: %.2f, E FN: %.2f"%(voxel_sum(voxel_TP), voxel_sum(voxel_FP), voxel_sum(voxel_FN)))
        fig.suptitle("True vs. Active [Energy]\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig.show()

    if fig_histo is not None:
        histo_voxels_comp(fig_histo, fig_histo.add_subplot(), voxels_comp_T, "True [MeV]", "Active [MeV]")
        fig_histo.text(0.4, 0.8, "True Threshold: %f"%true_thres)
        fig_histo.suptitle("Energy Histo True vs. Active\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_histo.show()

    if fig_inf is not None:
        scatter_voxels_comp(fig_inf, voxels_comp_inf, "True [Channel Val]", "Active [Channel Val]", plot_slice, view_angle)
        fig_inf.text(0, 0.95, "True Threshold: %f"%true_thres)
        fig_inf.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig_inf.suptitle("True vs. Active [Channel Val]\nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_inf.show()

    if fig_histo_inf is not None:
        ax_comp = fig_histo_inf.add_subplot()
        histo_voxels_comp(fig_histo_inf, ax_comp, voxels_comp_inf, "True [Channel Val]", "Active [Channel Val]", log_yscale=True)
        fig_histo_inf.text(0.4, 0.8, "True Threshold: %f"%true_thres)
        fig_histo_inf.suptitle("Channel Val Histo \nXY file: %s, Event: %d"%(xy_file[xy_file.rfind('/')+1:], event))
        fig_histo_inf.show()

def plot_pixel_stats(event = 1, true_thres=0, xy_file=sys.argv[1]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1/3, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres, False)

    voxel_FN, voxel_T_active, _, voxel_FP = comp_voxels(voxel_truth, voxel_active)

    n_T = len(voxel_truth)
    n_TP_naive = len(voxel_T_active) #naive: treat all active as positive
    n_FP_naive = len(voxel_FP)

    xs, ys, thresholds = SP_curve(voxel_T_active, voxel_active, np.arange(100)*0.1)
    plt.plot(xs, ys)
    plt.show()


if __name__=="__main__":
    for i in range(5):
        print(i)
        #compare_energys(i)
        #compare_channels(i)
        #compare_true_active(plt.figure(), i)
        #plot_pixel_stats(i)
