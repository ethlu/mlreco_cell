import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *

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

def compare_true_active(fig, event = 1, true_thres = 0, plot_lims=None, view_angle=None, plot_histo=True, xy_file = sys.argv[1]):
    dict_truth, coords_active, _ = parse_xy(event, true_thres, xy_file=xy_file)
    coords_active = set(coords_active)
    coords_T = set(dict_truth.keys())
    coords_T_active = coords_T.intersection(coords_active)
    E_T = np.array(list(dict_truth.values()))
    E_T_active = [dict_truth[pt] for pt in coords_T_active]
    if plot_histo:
        fig_hist, ax = plt.subplots()
        _, bins, _ = ax.hist(E_T, bins=20, label="True")
        ax.hist(E_T_active, bins=bins, label="Active")
        E_T = sum(E_T)
        E_T_active = sum(E_T_active)
        fig_hist.text(0.5, 0.7, "True Threshold: %f"%true_thres)
        fig_hist.text(0.5, 0.6, "Total True Energy: %.2f"%E_T)
        fig_hist.text(0.5, 0.5, "Total True & Active Energy: %.2f"%E_T_active)
        ax.set_xticks(bins)
        #ax.set_xlim(0, 3)
        ax.legend()
    else:
        E_T = sum(E_T)
        E_T_active = sum(E_T_active)

    compare_voxels(fig, coords_T, coords_active, "True", "Active", plot_lims, view_angle)
    fig.text(0.05, 0.7, "True Threshold: %f"%true_thres)
    fig.text(0.05, 0.6, "Total True Energy: %.2f"%E_T)
    fig.text(0.05, 0.5, "Total True & Active Energy: %.2f"%E_T_active)
    fig.suptitle("Truth vs. Active")
    plt.show()

if __name__=="__main__":
    for i in range(5):
        print(i)
        #compare_energys(i)
        #compare_channels(i)
        compare_true_active(plt.figure(), i)
