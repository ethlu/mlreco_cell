import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *

def filter_voxel_val(voxel, thres):
    return {pt: v for pt, v in voxel.items() if v >= thres}

def filter_voxels_coord(coord_lims, *voxels):
    if coord_lims is None: return voxels
    ret = []
    for voxel in voxels:
        for i, lim in enumerate(coord_lims):
            if lim is None: continue
            voxel = {c : voxel[c] for c in voxel if c[i] >= lim[0] and c[i] < lim[1]}
        ret.append(voxel)
    return ret 

def filter_voxels_coords(coords, *voxels):
    ret = []
    coords = set(coords)
    for voxel in voxels:
        ret.append({c:voxel[c] for c in voxel if c in coords})
    return ret 

def downsample_voxels(downsample, *voxels, reduce=False, avg=False):
    ret = []
    for voxel in voxels:
        new_voxel = {}
        for coord, val in voxel.items():
            new_coord = []
            for i, factor in enumerate(downsample):
                if factor == -1: continue
                new_coord.append(coord[i]//factor if reduce else coord[i]//factor*factor)
            new_coord = tuple(new_coord)
            try:
                if new_voxel[new_coord] is None:
                    new_voxel[new_coord] = [val, 1]
                elif val is not None:
                    new_voxel[new_coord][0] += val
                    new_voxel[new_coord][1] += 1
            except KeyError:
                new_voxel[new_coord] = [val, 1]
        if avg:
            new_voxel = {k: v[0]/v[1] for k, v in new_voxel.items()}
        else:
            new_voxel = {k: v[0] for k, v in new_voxel.items()}
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

def voxel_to_numpy(voxel):
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

def mode1d(X):
    counts = {}
    for x in X:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return max(counts.items(), key=lambda x: x[1])[0]
    
def track_stats(coords, fig=None):
    from sklearn.decomposition import PCA
    from numpy import arctan, arccos, degrees
    if len(coords) < 20: return None
    pca = PCA()
    coords = np.array(coords)
    pca.fit(coords)
    if fig is not None:
        ax=fig.add_subplot(projection="3d") 
        scatter_voxel(fig, ax, {tuple(c):None for c in coords})
        ax.quiver(*np.mean(coords, axis=0), *(20*pca.components_[0].T))
        plt.show()
    axis = np.flip(pca.components_[0])
    phi = degrees(arctan(axis[1]/axis[0]))
    phi = phi+180 if phi<0 else phi
    theta = degrees(arccos(axis[2]))
    theta = theta+180 if theta<0 else theta
    #print("explained",pca.explained_variance_)
    #print("singuar",pca.singular_values_)
    return theta, phi, pca.explained_variance_ratio_[0]

def track_fit(voxel_track, coord_start=None, voxel_size=(1, 1, 1), step=1, max_deviation=20, pca_thres=0.98, pca_nrange=(5, 100), fig=None):
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    pca = PCA()
    if type(step) == tuple:
        step_range = step
    else:
        step_range = (step, step+np.linalg.norm(voxel_size))
    coords, _ = voxel_to_numpy(voxel_track)
    cluster_labels = DBSCAN(eps=max_deviation, min_samples=pca_nrange[0]*5).fit_predict(coords*voxel_size) 
    if np.count_nonzero(cluster_labels==-1) > len(voxel_track)/2:
        cluster_labels = DBSCAN(eps=max_deviation, min_samples=pca_nrange[0]).fit_predict(coords*voxel_size) 
    voxel = filter_voxels_coords(map(tuple, coords[cluster_labels!=-1]), voxel_track)[0]
    """
    print("DBSCAN filtered out %d samples, found %d clusters"%(len(voxel_track)-len(voxel), len(np.unique(cluster_labels))-1))
    if fig is not None:
        labels = np.unique(cluster_labels)
        fig_ = plt.figure()
        scatter_voxels(fig_, fig_.add_subplot(projection="3d"), 
                       [{tuple(coord): None for coord in coords[cluster_labels==label]} for label in labels], list(map(str, labels)))
    """
    if coord_start is None:
        coords, _ = voxel_to_numpy(voxel)
        coord_start = coords[np.argmax(coords[:, 1])]
    coord_start_copy = coord_start = np.array(coord_start)
    track_ranges, track_vals = [0], []
    voxel_range, pca_vectors = {}, []
    coords_this = []
    while len(voxel) >= pca_nrange[0]:
        coords = voxel_to_numpy(voxel)[0]
        coords_sorted = coords[np.argsort(np.linalg.norm((coords-coord_start)*voxel_size, axis=1))]
        coords_pca = coords_sorted[:pca_nrange[1]]
        reset = False
        while True:
            pca.fit(coords_pca * voxel_size)
            if pca.explained_variance_ratio_[0] < pca_thres:
                if len(coords_pca) > pca_nrange[0]: 
                    coords_pca = coords_sorted[: max(len(coords_pca)-pca_nrange[0], pca_nrange[0])]
                    continue
                if coords_this:
                    old_ratio = pca.explained_variance_ratio_[0]
                    pca.fit(np.concatenate((coords_pca, coords_this)) * voxel_size)
                    if pca.explained_variance_ratio_[0] < old_ratio:
                        pca.fit(coords_pca * voxel_size)
            elif len(voxel) - len(coords_pca) < pca_nrange[0]:
                coords_pca = coords
            coords_rel = (coords_pca - coord_start)*voxel_size
            dists_parallel = coords_rel@pca.components_[0]
            mask_pos, mask_neg = dists_parallel > step_range[0], dists_parallel < -step_range[0]
            count_pos, count_neg = np.count_nonzero(mask_pos), np.count_nonzero(mask_neg)
            if count_pos >= count_neg and count_neg:
                assert count_neg < pca_nrange[0], "PCA detected %d samples in wrong direction"%count_neg
                for coord in coords_pca[mask_neg]: voxel.pop(tuple(coord), None)
                reset = True
                break
            if count_pos < count_neg and count_pos:
                assert count_pos < pca_nrange[0], "PCA detected %d samples in wrong direction"%count_pos
                for coord in coords_pca[mask_pos]: voxel.pop(tuple(coord), None)
                reset = True
                break
            dists_transverse = np.linalg.norm(coords_rel - np.outer(dists_parallel, pca.components_[0]), axis=1)
            dists_parallel_abs = np.abs(dists_parallel) + track_ranges[-1]
            dists_sorted = np.argsort(dists_parallel_abs)
            dists_parallel_abs, dists_transverse, coords_pca = dists_parallel_abs[dists_sorted], dists_transverse[dists_sorted], coords_pca[dists_sorted]
            break_i = np.argmax(np.diff(dists_parallel_abs) > step_range[1])
            coord_end = coords_pca[break_i if break_i else -1]
            if fig is not None: pca.fit(coords_pca)
            pca_vectors.append((coord_start, pca.components_[0]*(1 if count_pos>count_neg else -1)))
            break
        if reset: continue
        coords_this, vals_this, is_last = [], [], False
        for dist_parallel_abs, dist_transverse, coord in zip(dists_parallel_abs, dists_transverse, map(tuple, coords_pca)):
            if dist_parallel_abs >= dists_parallel_abs[-1] - step_range[1]:
                is_last = True
            step_size = dist_parallel_abs - track_ranges[-1] 
            if step_size > step_range[0]:
                if step_size > step_range[1]: 
                    del track_ranges[-1]
                else:
                    track_vals.append(np.sum(vals_this)/step_size)
                    for c in coords_this:
                        voxel_range[c] = track_ranges[-1]
                track_ranges.append(dist_parallel_abs)
                if is_last:
                    coord_start = coord
                    break
                coords_this.clear()
                vals_this.clear()
            if dist_transverse <= max_deviation:
                coords_this.append(coord)
                vals_this.append(voxel[coord])
            del voxel[coord]
    del track_ranges[-1]
    if fig is not None:
        if fig is True: fig = plt.figure()
        ax=fig.add_subplot(211, projection="3d") 
        coords_filtered, vals_filtered = voxel_to_numpy(comp_voxels(voxel_track, voxel_range)[0])
        ax.scatter3D(*coords_filtered.T, s=3, 
                     label="filtered \nN: %d, Total Val: %.1f"%(len(coords_filtered), sum(vals_filtered)), c='r', marker='^')
        ax.scatter3D(*coord_start_copy, s=50, label="start", c='g', marker='^')
        ax.scatter3D(*coord_end, s=50, label="end", c='purple', marker='^')
        ax.scatter3D(*voxel_to_numpy(voxel_range)[0].T, s=2, c=voxel_to_numpy(voxel_range)[1],
                     label="Fitted Track \nN: %d, Total Val: %.1f"%(len(voxel_range), sum(comp_voxels(voxel_track, voxel_range)[1].values())))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        track_size = np.linalg.norm(get_voxels_lims(voxel_track)[:, 1] - get_voxels_lims(voxel_track)[:, 0])
        step_size = step_range[1] / np.linalg.norm(voxel_size)
        for pt, direction in pca_vectors:
            ax.quiver(pt[0], pt[1], pt[2]+track_size/10, *(step_size*track_size/50*direction.T))
        ax2=fig.add_subplot(212) 
        ax2.scatter(track_ranges, track_vals)
        ax2.set_xlabel("Range")
        ax2.set_ylabel("Val density")
        plt.show()
    return track_ranges, track_vals, coord_end, (voxel_range, pca_vectors)

def SP_stats(voxel_T, voxel_P, T_weighted=False, get_voxels_comp=True): 
    if get_voxels_comp:
        comp = comp_voxels(voxel_T, voxel_P)
        coords_TP = comp[1].keys()
    else:
        coords_TP = voxel_T.keys() & voxel_P.keys()
    n_TP = len(coords_TP) 
    n_P = len(voxel_P)
    if T_weighted:
        T_TP = sum([voxel_T[coord] for coord in coords_TP])
        T_T = sum(voxel_T.values())
        sensitivity = 1 if T_TP==0 else T_TP/T_T
    else:
        n_T = len(voxel_T)
        sensitivity = 1 if n_T==0 else n_TP/n_T
    purity = 1 if n_P==0 else n_TP/n_P 
    if get_voxels_comp:
        return purity, sensitivity, comp
    return purity, sensitivity

def SP_curve(voxel_T, voxel_inf, T_weighted=False, downsample=(1,1,1), thresholds = np.arange(50)*2E-2):
    xs, ys = [], []
    if not T_weighted and downsample == (1,1,1):
        coords = voxel_inf.keys()
        infs = np.array(list(voxel_inf.values()))
        truths = np.array([1 if coord in voxel_T else 0 for coord in coords])
        n_T = len(voxel_T)
        for thres in thresholds:
            preds = np.where(infs>thres, 1, 0)
            n_P = np.sum(preds)
            n_TP = np.sum(preds[preds == truths])
            xs.append(1 if n_P==0 else n_TP/n_P)
            ys.append(1 if n_T==0 else n_TP/n_T)
    else:
        for thres in thresholds:
            x, y = SP_stats(voxel_T, downsample_voxels(downsample, filter_voxel_val(voxel_inf, thres))[0], T_weighted, False)
            xs.append(x)
            ys.append(y)
    """
    xs, ys = [], []
    n_T = len(voxel_T)
    n_F = len(comp_voxels(voxel_T, voxel_inf)[3])
    for thres in thresholds:
        vox_P =  filter_voxel_val(voxel_inf, thres)
        n_TP = len(voxel_T.keys() & vox_P.keys())
        x = 1 - (len(vox_P) - n_TP)/n_F
        y = n_TP/n_T
        xs.append(x)
        ys.append(y)
    """
    return xs, ys, thresholds

def SP_score(SP_curve):
    purity = SP_curve[0]
    sensitivity = SP_curve[1]
    delta_sens = sensitivity - np.concatenate((sensitivity[1:], [0]))
    return delta_sens@purity
    
def optim_threshold(SP_curve):
    xs, ys, thresholds = SP_curve
    sums = np.array(xs) + np.array(ys)
    max_i = np.argmax(sums)
    return thresholds[max_i], xs[max_i], ys[max_i]

def inference_analysis(voxel_T, voxel_inf, thres=None, coord_lims=None, downsample=(1,1,1), to_slice=False):
    if thres is None:
        thres, _, _ = optim_threshold(SP_curve(voxel_T, voxel_inf))
    voxel_P = filter_voxel_val(voxel_inf, thres)
    evt_purity, evt_sensitivity, evt_voxels_comp = SP_stats(voxel_T, voxel_P)

    voxel_T, voxel_P, voxel_inf = filter_voxels_coord(coord_lims, voxel_T, voxel_P, voxel_inf)
    voxel_T, voxel_P, voxel_inf = downsample_voxels(downsample, voxel_T, voxel_P, voxel_inf)
    if coord_lims is None:
        coord_lims = get_voxels_lims(voxel_T, voxel_P)
    x_lim, y_lim, z_lim = coord_lims
    if y_lim is None:
        coord_lims = get_voxels_lims(voxel_T, voxel_P)
        coord_lims[0] = x_lim
    coord_lims = tuple(map(tuple, coord_lims))

    if to_slice:
        voxel_T, voxel_P, voxel_inf = downsample_voxels((-1, 1, 1), voxel_T, voxel_P, voxel_inf)
    voxel_T_inf = set_voxel_vals(voxel_T, voxel_inf)
    voxel_P_T = set_voxel_vals(voxel_P, voxel_T)
    purity, sensitivity, voxels_comp_T = SP_stats(voxel_T, voxel_P_T)
    voxels_comp_inf = comp_voxels(voxel_T_inf, voxel_P)
    return thres, coord_lims, evt_purity, evt_sensitivity, evt_voxels_comp, purity, sensitivity, \
        voxels_comp_T, voxels_comp_inf, voxel_T, voxel_inf

def scatter_voxel(fig, ax, voxel, name="", is2d=False, view_angle=None, plot_lims=None, size=2, cmap=plt.get_cmap('viridis'), cax=None):
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
        if cax is None:
            cax = fig.add_axes([pos.x1, pos.y0, 0.02, pos.y1-pos.y0])
        if is2d:
            coords = coords[:, ::-1] if view_angle else coords
            if cax: 
                fig.colorbar(ax.scatter(*coords.T, s=size, c=vals, cmap=cmap, label=label), cax = cax)
            else:
                #ax.scatter(*coords.T, s=size, c=vals, cmap=cmap, vmax=500, label=label)
                ax.scatter(*coords.T, s=size, c=vals, cmap=cmap, label=label)
        else:
            fig.colorbar(ax.scatter3D(*coords.T, s=size, c=vals, cmap=cmap, label=label), cax = cax)
    ax.legend(loc='lower left', bbox_to_anchor=(0.7, 0.7))
    if is2d:
        if view_angle:
            ax.set_xlabel('Z')
            ax.set_ylabel('Y')
        else:
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        fig.subplots_adjust(left=0.25, hspace=0.7)
        if plot_lims is not None:
            x_lim, y_lim = plot_lims
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if view_angle is not None:
            ax.view_init(*view_angle)
        if plot_lims is not None:
            x_lim, y_lim, z_lim = plot_lims
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_zlim(*z_lim)

def scatter_voxels(fig, ax, voxels, names, colors=None, title="", is2d=False, view_angle=None, size=2, alpha=0.4):
    for i, voxel in enumerate(voxels):
        if not voxel: continue
        coords, vals = voxel_to_numpy(voxel)
        if is2d:
            coords = coords[:, ::-1] if view_angle else coords
            ax.scatter(*coords.T, s=size, label=names[i], c=colors[i] if colors else None, alpha=alpha)
        else:
            ax.scatter3D(*coords.T, s=size, label=names[i], c=colors[i] if colors else None)
    if not is2d and view_angle is not None:
        ax.view_init(*view_angle)
    ax.set_title(title)
    ax.legend(loc='lower left', bbox_to_anchor=(0.7, 0.7))
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

    ax.text(0.4, 0.7, "N %s Voxels: %d"%(name_A, len(voxel_missing_from_A)+len(voxel_intersect)), transform=ax.transAxes)
    ax.text(0.4, 0.65, "N %s Voxels: %d"%(name_B, len(voxel_missing_from_B)+len(voxel_intersect)), transform=ax.transAxes)

def plot_voxel_projection(fig, voxel, size=0.1, tpc=1):
    from geom.pdsp import get_APA_wireplane_maps, get_APA_wireplane_projectors, MAX_TICK
    umap, vmap, xmap, = get_APA_wireplane_maps(tpc)
    uproj, vproj, xproj = get_APA_wireplane_projectors(tpc)
    voxel_u, voxel_v, voxel_x = {}, {}, {}
    for coord, val in voxel.items():
        coord_u = (int(coord[0]), uproj((coord[2], coord[1])))
        if coord_u in voxel_u:
            voxel_u[coord_u] += val
        else:
            voxel_u[coord_u] = val
        coord_v = (int(coord[0]), vproj((coord[2], coord[1])))
        if coord_v in voxel_v:
            voxel_v[coord_v] += val
        else:
            voxel_v[coord_v] = val
        coord_x = (int(coord[0]), xproj((coord[2], coord[1])))
        if coord_x in voxel_x:
            voxel_x[coord_x] += val
        else:
            voxel_x[coord_x] = val
    scatter_voxel(fig, fig.add_subplot(131), voxel_u, is2d=True, view_angle=True, cax=False, size=size)
    scatter_voxel(fig, fig.add_subplot(132), voxel_v, is2d=True, view_angle=True, cax=False, size=size)
    scatter_voxel(fig, fig.add_subplot(133), voxel_x, is2d=True, view_angle=True, cax=False, size=size)