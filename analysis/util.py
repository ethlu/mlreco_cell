import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from geom.util import voxelize

def parse_xy(event = 1, true_thres=0, downsample = (1, 1, 1), xy_file = sys.argv[1]):
    with np.load(xy_file, allow_pickle=True) as xy_f:
        pix_batch=xy_f["X"]
        energys_truth = xy_f["Y_truth"]
        event_starts=xy_f["starts"]
    n_events = len(event_starts)
    if event == n_events-1:
        event_pixels = pix_batch[event_starts[event]:]
    else:
        event_pixels = pix_batch[event_starts[event]:event_starts[event+1]]
    event_truth = energys_truth[event]
    if downsample != (1, 1, 1):
        downsample = np.array(downsample)
        event_truth = voxelize(event_truth, downsample)
        event_pixels[:, :3] = event_pixels[:, :3]//downsample

    dict_truth = {tuple(pt[:3]):pt[3] for pt in event_truth if pt[3]>=true_thres}
    coords_active = [tuple(pt) for pt in event_pixels[:, :3]]
    return dict_truth, coords_active, event_starts

def set_thres(Y, thres):
    return {pt: E for pt, E in Y.items() if E > thres}

def P_stats(P, coords_T, N): 
    n_T = len(coords_T)
    coords_P = set(P.keys())
    n_P = len(coords_P)

    coords_TP = coords_T.intersection(coords_P)
    coords_FP = coords_P.difference(coords_TP) 
    coords_FN = coords_T.difference(coords_TP)
    n_TP = len(coords_TP)
    n_FP = len(coords_FP)
    n_FN = len(coords_FN)
    n_TN = N-n_TP-n_FP-n_FN

    sensitivity = n_TP/n_T
    #specificity = n_TN/(n_FP+n_TN)
    purity = 1 if n_P==0 else n_TP/n_P 
    """
    print("TP N: ", n_TP)
    print("FP N: ", n_FP)
    print("FN N: ", n_FN)
    print("TN N: ", n_TN) 
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    return sensitivity, 1-specificity
    """
    return purity, sensitivity

def compare_voxels(fig, coords_A, coords_B, name_A=None, name_B=None, plot_lims=None, view_angle=None, plot_all=False):
    coords_A = set(coords_A)
    coords_B = set(coords_B)
    intersect = coords_A.intersection(coords_B)
    missing_from_A = coords_A.difference(intersect)
    missing_from_B = coords_B.difference(intersect)
    coords_A = np.array([list(pt) for pt in coords_A])
    coords_B = np.array([list(pt) for pt in coords_B])
    intersect = np.array([list(pt) for pt in intersect])
    missing_from_A = np.array([list(pt) for pt in missing_from_A])
    missing_from_B = np.array([list(pt) for pt in missing_from_B])

    if plot_lims is not None:
        x_lim, y_lim, z_lim = plot_lims
    SIZE = 2
    if plot_all:
        ax = fig.add_subplot(311, projection="3d")
        if len(intersect) > 0:
            ax.scatter3D(*intersect.T, s=SIZE, label="%s & %s"%(name_A, name_B))
        if len(missing_from_A) > 0:
            ax.scatter3D(*missing_from_A.T, s=SIZE, label="%s but not %s"%(name_A, name_B))
        ax2 = fig.add_subplot(312, projection="3d")
        if len(intersect) > 0:
            ax2.scatter3D(*intersect.T, s=SIZE, label="%s & %s"%(name_A, name_B))
        if len(missing_from_B) > 0:
            ax2.scatter3D(*missing_from_B.T, s=SIZE, label="%s but not %s"%(name_B, name_A), color="green")

        ax3 = fig.add_subplot(313, projection="3d")
        if len(intersect) > 0:
            ax3.scatter3D(*intersect.T, s=SIZE, label="%s & %s"%(name_A, name_B))
        if len(missing_from_A) > 0:
            ax3.scatter3D(*missing_from_A.T, s=SIZE, label="%s but not %s"%(name_A, name_B))
        if len(missing_from_B) > 0:
            ax3.scatter3D(*missing_from_B.T, s=SIZE, label="%s but not %s"%(name_B, name_A))
        axes = [ax, ax2, ax3]
    else:
        ax = fig.add_subplot(411, projection="3d")
        if len(missing_from_A) > 0:
            ax.scatter3D(*missing_from_A.T, s=SIZE, label="%s but not %s"%(name_A, name_B), c="r")
        ax2 = fig.add_subplot(412, projection="3d")
        if len(missing_from_B) > 0:
            ax2.scatter3D(*missing_from_B.T, s=SIZE, label="%s but not %s"%(name_B, name_A), color="y")
        ax3 = fig.add_subplot(413, projection="3d")
        if len(intersect) > 0:
            ax3.scatter3D(*intersect.T, s=SIZE, label="%s & %s"%(name_A, name_B))
        if len(missing_from_A) > 0:
            ax3.scatter3D(*missing_from_A.T, s=SIZE, label="%s but not %s"%(name_A, name_B), c="r")
        if len(missing_from_B) > 0:
            ax3.scatter3D(*missing_from_B.T, s=SIZE, label="%s but not %s"%(name_B, name_A), c="y")
        ax4 = fig.add_subplot(414, projection="3d")
        if len(intersect) > 0:
            ax4.scatter3D(*intersect.T, s=SIZE, label="%s & %s"%(name_A, name_B))
        axes = [ax, ax2, ax3, ax4]

        if plot_lims is None:
            x_lim = ax4.get_xlim()
            y_lim = ax4.get_ylim()
            z_lim = ax4.get_zlim()

    for axi in axes:
        axi.set_xlim(*x_lim)
        axi.set_ylim(*y_lim)
        axi.set_zlim(*z_lim)
        if view_angle is not None:
            axi.view_init(*view_angle)

    if name_A is not None and name_B is not None:
        if plot_all:
            ax.set_title("All %s Voxels"%name_A)
            ax2.set_title("All %s Voxels"%name_B)
        else:
            ax.set_title("%s but not %s"%(name_A, name_B))
            ax2.set_title("%s but not %s"%(name_B, name_A))
            ax4.set_title("%s & %s"%(name_A, name_B))

        ax3.set_title("All %s or %s Voxels"%(name_A, name_B))
        for axi in axes:
            axi.legend()
            axi.set_xlabel('X')
            axi.set_ylabel('Y')
            axi.set_zlabel('Z')
        fig.text(0.05, 0.4, "N %s Voxels: %d"%(name_A, len(coords_A)))
        fig.text(0.05, 0.3, "N %s Voxels: %d"%(name_B, len(coords_B)))
        fig.text(0.05, 0.2, "N %s & %s Voxels: %d"%(name_A, name_B, len(intersect)))
    return x_lim, y_lim, z_lim
