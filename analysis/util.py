import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *

def filter_voxel_val(voxel, thres, none_v=True):
    return {pt: None if none_v else v for pt, v in voxel.items() if v > thres}

def filter_voxels_coord(coord_lims, *voxels):
    if coord_lims is None: return voxels
    ret = []
    for voxel in voxels:
        for i, lim in enumerate(coord_lims):
            if lim is None: continue
            voxel = {c : voxel[c] for c in voxel if c[i] >= lim[0] and c[i] < lim[1]}
        ret.append(voxel)
    return ret 

def downsample_voxels(downsample, *voxels):
    ret = []
    for voxel in voxels:
        new_voxel = {}
        for coord, val in voxel.items():
            new_coord = []
            for i, factor in enumerate(downsample):
                if factor == -1: continue
                new_coord.append(coord[i]//factor)
            new_coord = tuple(new_coord)
            try:
                if new_voxel[new_coord] is None:
                    new_voxel[new_coord] = val
                elif val is not None:
                    new_voxel[new_coord] += val
            except KeyError:
                new_voxel[new_coord] = val
        ret.append(new_voxel)
    return ret

def comp_voxels(voxel_A, voxel_B, coords_only=False):
    coords_A = set(voxel_A.keys())
    coords_B = set(voxel_B.keys())
    intersect = coords_A.intersection(coords_B)
    coords_A_B = coords_A.difference(intersect)
    coords_B_A = coords_B.difference(intersect)
    if coords_only:
        return coords_A_B, intersect, coords_B_A
    
    voxel_A_B = {c: voxel_A[c] for c in coords_A_B}
    voxel_int_A = {c: voxel_A[c] for c in intersect}
    voxel_int_B = {c: voxel_B[c] for c in intersect}
    voxel_B_A = {c: voxel_B[c] for c in coords_B_A}
    return voxel_A_B, voxel_int_A, voxel_int_B, voxel_B_A

def set_voxel_vals(voxel_coords, voxel_vals):
    return {coord : voxel_vals[coord] if coord in voxel_vals else None \
            for coord in voxel_coords}

def voxel_sum(voxel):
    if voxel_is_none(voxel):
        return 0
    else:
        return sum(voxel.values())

def voxel_to_numpy(voxel, filter_none=True):
    coords = np.array([list(coord) for coord in voxel.keys()])
    vals = np.array(list(voxel.values()))
    return coords, vals

def voxel_is_none(voxel):
    return not voxel or next(iter(voxel.values())) is None

def get_voxels_lims(*voxels, default=np.zeros((3, 2))):
    coords = []
    for voxel in voxels:
        coords.extend(voxel.keys())
    if not coords: return default
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)+1
    return np.array([coord_min, coord_max]).T

def SP_stats(voxel_T, voxel_P, coords_only=False): 
    comp = comp_voxels(voxel_T, voxel_P, coords_only)
    n_T = len(voxel_T)
    n_P = len(voxel_P)
    n_TP = len(comp[1]) 
    sensitivity = 1 if n_T==0 else n_TP/n_T
    purity = 1 if n_P==0 else n_TP/n_P 
    return purity, sensitivity, comp

def SP_curve(voxel_T, voxel_inf, thresholds = np.arange(50)*2E-2):
    xs, ys = [], []
    for thres in thresholds:
        x, y, _, = SP_stats(voxel_T, filter_voxel_val(voxel_inf, thres), True)
        xs.append(x)
        ys.append(y)
    return xs, ys, thresholds

def optim_threshold(SP_curve):
    xs, ys, thresholds = SP_curve
    sums = np.array(xs) + np.array(ys)
    max_i = np.argmax(sums)
    return thresholds[max_i], xs[max_i], ys[max_i]

def inference_analysis(voxel_T, voxel_inf, thres=None, coord_lims=None, to_slice=False):
    if thres is None:
        thres, _, _ = optim_threshold(SP_curve(voxel_T, voxel_inf))
    voxel_P = filter_voxel_val(voxel_inf, thres, False)
    evt_purity, evt_sensitivity, evt_voxels_comp = SP_stats(voxel_T, voxel_P)

    voxel_T, voxel_P = filter_voxels_coord(coord_lims, voxel_T, voxel_P)
    if coord_lims is None:
        coord_lims = get_voxels_lims(voxel_T, voxel_P)
    x_lim, y_lim, z_lim = coord_lims
    if y_lim is None:
        coord_lims = get_voxels_lims(voxel_T, voxel_P)
        coord_lims[0] = x_lim
    coord_lims = tuple(map(tuple, coord_lims))

    voxel_T_inf = set_voxel_vals(voxel_T, voxel_inf)
    voxel_P_T = set_voxel_vals(voxel_P, voxel_T)
    if to_slice:
        voxel_T, voxel_T_inf, voxel_P, voxel_P_T = downsample_voxels((-1, 1, 1), voxel_T, voxel_T_inf, voxel_P, voxel_P_T)
    purity, sensitivity, voxels_comp_T = SP_stats(voxel_T, voxel_P_T)
    voxels_comp_inf = comp_voxels(voxel_T_inf, voxel_P)
    return thres, coord_lims, evt_purity, evt_sensitivity, evt_voxels_comp, purity, sensitivity, voxels_comp_T, voxels_comp_inf

def scatter_voxel(fig, ax, voxel, name="", is2d=False, view_angle=None, plot_lims=None, size=2, cmap=plt.get_cmap('viridis')):
    ax.set_title(name)
    if not voxel: return
    coords, vals = voxel_to_numpy(voxel)
    label = name+"\nN: %d"%len(coords)
    if voxel_is_none(voxel):
        if is2d:
            coords = coords[:, ::-1] if view_angle else coords
            ax.scatter(*coords.T, s=size, label=label, c='black')
        else:
            ax.scatter3D(*coords.T, s=size, label=label, c='black')
    else:
        label=label+"\nTotal Val: %.2f"%sum(vals)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1, pos.y0, 0.02, pos.y1-pos.y0])
        if is2d:
            coords = coords[:, ::-1] if view_angle else coords
            fig.colorbar(ax.scatter(*coords.T, s=size, c=vals, cmap=cmap, label=label), cax = cax)
        else:
            fig.colorbar(ax.scatter3D(*coords.T, s=size, c=vals, cmap=cmap, label=label), cax = cax)
    ax.legend()
    if is2d:
        if view_angle:
            ax.set_xlabel('Z')
            ax.set_ylabel('Y')
        else:
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        fig.subplots_adjust(left=0.25, hspace=0.7)
        x_lim, y_lim = plot_lims
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if view_angle is not None:
            ax.view_init(*view_angle)
        x_lim, y_lim, z_lim = plot_lims
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_ylim(*z_lim)

def scatter_voxels(fig, ax, voxels, names, colors, title="", is2d=False, view_angle=None, size=2, alpha=0.4):
    for i, voxel in enumerate(voxels):
        if not voxel: continue
        coords, vals = voxel_to_numpy(voxel)
        if is2d:
            coords = coords[:, ::-1] if view_angle else coords
            ax.scatter(*coords.T, s=size, label=names[i], c=colors[i], alpha=alpha)
        else:
            ax.scatter3D(*coords.T, s=size, label=names[i], c=colors[i])
    if not is2d and view_angle is not None:
        ax.view_init(*view_angle)
    ax.set_title(title)
    ax.legend()
    if is2d:
        if view_angle:
            ax.set_xlabel('Z')
            ax.set_ylabel('Y')
        else:
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        fig.subplots_adjust(left=0.25, hspace=0.7)
        return ax.get_xlim(), ax.get_ylim()
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if view_angle is not None:
            ax.view_init(*view_angle)
        return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

def scatter_voxels_comp(fig, voxels_comp, name_A="A", name_B="B", is2d=False, view_angle=None):
    voxel_missing_from_A, voxel_intersect, _, voxel_missing_from_B = voxels_comp
    
    projection = "rectilinear" if is2d else "3d"
    plot_lims = scatter_voxels(fig, fig.add_subplot(414, projection=projection),
                    [voxel_intersect, voxel_missing_from_A, voxel_missing_from_B],
                    ["%s & %s"%(name_A, name_B), "%s but not %s"%(name_A, name_B), "%s but not %s"%(name_B, name_A)],
                    ["b", "r", "y"],
                    "All %s or %s Voxels"%(name_A, name_B),
                    is2d, view_angle)
    scatter_voxel(fig, fig.add_subplot(411, projection=projection), voxel_missing_from_A, "%s but not %s"%(name_A, name_B), is2d, view_angle, plot_lims)
    scatter_voxel(fig, fig.add_subplot(412, projection=projection), voxel_missing_from_B, "%s but not %s"%(name_B, name_A), is2d, view_angle, plot_lims)
    scatter_voxel(fig, fig.add_subplot(413, projection=projection), voxel_intersect, "%s & %s"%(name_A, name_B), is2d, view_angle, plot_lims)
    fig.text(0., 0.4, "N %s Voxels: %d"%(name_A, len(voxel_missing_from_A)+len(voxel_intersect)))
    fig.text(0., 0.3, "N %s Voxels: %d"%(name_B, len(voxel_missing_from_B)+len(voxel_intersect)))

def histo_voxels(fig, ax, voxels, labels, colors, title="", bins=None, log_yscale=False):
    data = [voxel_to_numpy(voxel)[1] for voxel in voxels]
    labels = [label+"\nN/Sum: %d/%.2f"%(len(vals), sum(vals)) for label, vals in zip(labels, data)]
    _, bins, _ = ax.hist(data, histtype='step', label=labels, color=colors, bins=bins)
    ax.set_xticks(bins)
    ax.legend()
    ax.set_title(title)
    if log_yscale:
        ax.set_yscale('log', basey=10)

def histo_voxels_comp(fig, ax, voxels_comp, name_A="A", name_B="B", bins=None, log_yscale=False):
    voxel_missing_from_A, voxel_intersect, _, voxel_missing_from_B = voxels_comp
    voxels_comp = (voxel_missing_from_A, voxel_intersect, voxel_missing_from_B)
    labels = ["%s but not %s"%(name_A, name_B), 
            "%s & %s"%(name_A, name_B),
            "%s but not %s"%(name_B, name_A)]
    colors = ["r", "b", "y"]
    data = np.array([[voxel, label, color] \
            for voxel, label, color in zip(voxels_comp, labels, colors) if not voxel_is_none(voxel)])
    
    title = "%s vs. %s"%(name_A, name_B)
    if len(data)>0:
        histo_voxels(fig, ax, data[:,0], data[:,1], data[:,2], title, bins, log_yscale)
    else:
        ax.set_title(title)

    fig.text(0.4, 0.7, "N %s Voxels: %d"%(name_A, len(voxel_missing_from_A)+len(voxel_intersect)))
    fig.text(0.4, 0.65, "N %s Voxels: %d"%(name_B, len(voxel_missing_from_B)+len(voxel_intersect)))

