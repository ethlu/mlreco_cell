import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *
from analysis.products import parse_xy, get_clusters_mc

N_VOXEL = (3585//(1.565/2)//8)*(5984//2//4)*(2300//2//4)
def parse_yinf(event_info, yinf_file=sys.argv[1]):
    if type(event_info) == int:
        event = event_info
    elif len(event_info) == 4:
        event, event_starti, event_endi, coords_active = event_info
    else:
        event, xy_dir = event_info
        _, _, event_info = parse_xy(event, 1, get_xy_file(xy_dir, yinf_file))
        event, event_starti, event_endi, coords_active = event_info
    if yinf_file[-3:] == "npz":
        with np.load(yinf_file) as f:
            event_yinf = f["arr_%d"%event]
        return {tuple(pt[:3]):pt[3] for pt in event_yinf}
    voxel_ghost = None
    if type(yinf_file) != str:
        yinf_file, (yinf_file_ghost, thres) = yinf_file
        voxel_ghost = parse_yinf(event_info, yinf_file_ghost)
    yinf_batch = np.load(yinf_file)[0]
    if event_endi == -1:
        event_yinf = yinf_batch[event_starti:]
    else:
        event_yinf = yinf_batch[event_starti:event_endi]
    assert len(event_yinf) == len(coords_active), "mismatch in yinf length?"
    voxel_yinf = {tuple(pt):val[0] for pt, val in zip(coords_active, event_yinf)}
    if voxel_ghost is not None:
        voxel_yinf = filter_voxels_coords(filter_voxel_val(voxel_ghost, thres), voxel_yinf)[0]
    return voxel_yinf

def get_xy_file(xy_dir, yinf_file, old=False):
    if old:
        return xy_dir + '/' + sorted(os.listdir(xy_dir))[int(yinf_file[yinf_file.find("batch")+5:-4])+200]
    _, f = inf_file_info(yinf_file if type(yinf_file) == str else yinf_file[0])
    xy_file = xy_dir + '/' + f.replace("yinf", "xy") + '.npz'
    return xy_file

def yinf_stats(event = 1, coord_lims=None, downsample=(1,1,1), true_thres=0, comp_true=False, T_weighted=False, mc_args=False, xy_file=sys.argv[1], yinf_file=sys.argv[2]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1, xy_file)
    voxel_yinf = parse_yinf(event_info, yinf_file)
    
    voxel_truth, voxel_active, voxel_yinf = filter_voxels_coord(coord_lims, voxel_truth, voxel_active, voxel_yinf)
    voxel_truth, voxel_active = downsample_voxels(downsample, voxel_truth, voxel_active)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)
    voxel_FN, voxel_T_active, _, voxel_FP = comp_voxels(voxel_truth, voxel_active)
    
    n_T = len(voxel_truth)
    n_TP_naive = len(voxel_T_active) #naive: treat all active as positive
    n_FP_naive = len(voxel_FP)
    
    if comp_true:
        voxel_baseline = voxel_truth
    else:
        voxel_baseline = voxel_T_active 
    xs, ys, thresholds = SP_curve(voxel_baseline, voxel_yinf, T_weighted, downsample)
    #xs, ys, thresholds = SP_curve(voxel_T_active, downsample_voxels(downsample, voxel_yinf)[0], T_weighted, thresholds = 8*np.arange(50)*2E-2)
    
    if mc_args:
        assert downsample == (1,1,1)
        index = file_info(yinf_file)[1]
        mc_beamcosmic, mc_stats_file = mc_args
        MIN_CLUSTER = 50
        MIN_CLUSTER_TRUTH = 20
        mc_stats = []
        clusters = get_clusters_mc(mc_stats_file, event, mc_beamcosmic, xy_file)
        for mc, coords in clusters.items():
            coords_P, coords_FN = coords
            coords = [*coords_P, *coords_FN]
            voxel_baseline_mc, voxel_yinf_mc = filter_voxels_coords(coords, voxel_baseline, voxel_yinf)
            cluster_size = len(voxel_yinf_mc)
            if cluster_size < MIN_CLUSTER or mc[2] < MIN_CLUSTER_TRUTH: continue
            n_T_mc = len(filter_voxels_coords(coords, voxel_truth)[0])
            n_TP_naive_mc = len(filter_voxels_coords(coords, voxel_T_active)[0])
            val_baseline_mc = sum(voxel_baseline_mc.values())
            xs_mc, ys_mc, _ = SP_curve(voxel_baseline_mc, voxel_yinf_mc, T_weighted, downsample, thresholds)
            mc_stats.append([mc, xs_mc, ys_mc, cluster_size, n_T_mc, n_TP_naive_mc, val_baseline_mc, index, event])
        return xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive, mc_stats
    return xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive

def yinf_avg_stats(n_yinf_files = 10, n_events=50, epoch = -1, tpc=None, evt_type=None, downsample=(1,1,1), true_thres = 0, comp_true=False, T_weighted=False, mc_args=False, segmented=False, event_stats=False, stats_dir = "stats", tag="v2", plot = False, xy_dir = sys.argv[1], yinf_dir=sys.argv[2]):
    yinf_files = os.listdir(yinf_dir)
    if tpc is not None:
        yinf_files = [f for f in yinf_files if int(tpc_info(f)[0])==tpc]
    if evt_type is not None:
        yinf_files = [f for f in yinf_files if file_info(f)[0]==evt_type]
    yinf_epoch = files_info(yinf_files, [0], inf_file_info)
    epoch = sorted(yinf_epoch.keys())[epoch]
    yinf_files = sorted(yinf_epoch[epoch])[:n_yinf_files]
    stats_file = stats_dir+"/epoch%d%s%s_stats-thres%.2f%s%s%s-N%d%s.npz"% \
        (epoch[0],  "-%s"%evt_type if evt_type is not None else "", "-tpc%d"%tpc if tpc is not None else "", true_thres, "-comptrue" if comp_true else "", "-weighted" if T_weighted else "", "-downsample%s"%str(downsample) if downsample!=(1,1,1) else "", n_events*n_yinf_files, "-%s"%tag if tag else "")
    assert os.path.exists(stats_dir)
    print(stats_file)
    if os.path.exists(stats_file):
        print(stats_file + " Exists")
        return
    print("Num Yinf Files: ", len(yinf_files))
    print("True Threshold: ", true_thres)
    Segmented_stats = {}
    if segmented:
        Y_INC = 374
        N_YSEG = 2
        Z_INC = 72
        N_ZSEG = 4
        Z0 = 0
        print("Segmented N_Z: %d, Z0: %d"%(N_ZSEG, Z0))
        for y_i in range(N_YSEG):
            for z_i in range(Z0, Z0+N_ZSEG):
                Segmented_stats["%d_%d"%(y_i, z_i)] = [[] for _ in range(5)]
    Xs, Ys, N_T, N_TP_naive, N_FP_naive, MC_stats, Event_stats = [], [], [], [], [], [], []
    for yinf_file in yinf_files:
        print(yinf_file)
        index = file_info(yinf_file)[1]
        xy_file = get_xy_file(xy_dir, yinf_file)
        for event in range(n_events):
            stats = yinf_stats(event, None, downsample, true_thres, comp_true, T_weighted, mc_args, xy_file, yinf_file=yinf_dir+'/'+yinf_file)
            if mc_args:
                xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive, mc_stats = stats
                MC_stats.extend(mc_stats)
            else:
                xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive = stats
            if event_stats:
                Event_stats.append([index, event, xs, ys, n_T, n_TP_naive, n_FP_naive])
            Xs.append(xs)
            Ys.append(ys)
            N_T.append(n_T)
            N_TP_naive.append(n_TP_naive)
            N_FP_naive.append(n_FP_naive)
            if segmented:
                for y_i in range(N_YSEG):
                    for z_i in range(Z0, Z0+N_ZSEG):
                        coord_lims = (None, (y_i*Y_INC, (y_i+1)*Y_INC), (z_i*Z_INC, (z_i+1)*Z_INC))
                        stats = list(yinf_stats(event, coord_lims, downsample, true_thres, comp_true, T_weighted, False, xy_file, yinf_file=yinf_dir+'/'+yinf_file))
                        stats.pop(2)
                        for i in range(5):
                            Segmented_stats["%d_%d"%(y_i, z_i)][i].append(stats[i])
                        
    xs = np.mean(Xs, 0)
    ys = np.mean(Ys, 0)
    n_T = np.mean(N_T)
    n_TP_naive = np.mean(N_TP_naive)
    n_FP_naive = np.mean(N_FP_naive)
    if segmented:
        for y_i in range(N_YSEG):
            for z_i in range(Z0, Z0+N_ZSEG):
                for i in range(5):
                    Segmented_stats["%d_%d"%(y_i, z_i)][i] = np.mean(Segmented_stats["%d_%d"%(y_i, z_i)][i], axis=0)

    if plot:
        ax = plt.subplot()
        ax.plot(xs, ys, marker='o')
        ax.text(0.2, 0.5, "N True Voxels: %.2f"% n_T, transform = ax.transAxes)
        ax.text(0.2, 0.4, "N Active & True: %.2f"% n_TP_naive, transform = ax.transAxes)
        ax.text(0.2, 0.3, "N Active but False : %.2f"% n_FP_naive, transform = ax.transAxes)
        ax.set_xlabel("Purity")
        ax.set_ylabel("Efficiency")
        ax.set_title("Efficiency vs. Purity (w.r.t. True and Active Voxels)")
        plt.show()
        #plt.savefig("plots_ghost/ratio_avg-N%d.png"%(n_events*n_yinf_files))
        plt.clf()

    Event_stats = {"event_stats": Event_stats} if event_stats else {}
    np.savez(stats_file, *MC_stats, **Segmented_stats, **Event_stats, xs=xs, ys=ys, thresholds=thresholds, n_T=n_T, n_TP_naive=n_TP_naive, n_FP_naive=n_FP_naive)
    return xs, ys, n_T, thresholds, n_TP_naive, n_FP_naive

def compare_yinf_stats(fig, stats_files = sys.argv[1:], labels=None):
    ax = fig.add_subplot()
    n_T, n_TP_naive, n_FP_naive, auc = [], [], [], []
    for i, f in enumerate(stats_files):
        if f == "a":
            continue
        with np.load(f) as stats_f:
            xs = stats_f["xs"]
            ys = stats_f["ys"]
            if "n_T" in stats_f:
                n_T.append("%d: %.1f"%(i, stats_f["n_T"]))
            else:
                n_T.append("%d: %.1f"%(i, stats_f["n_P"]))
            n_TP_naive.append("%d: %.1f"%(i, stats_f["n_TP_naive"]))
            n_FP_naive.append("%d: %.1f"%(i, stats_f["n_FP_naive"]))
            auc.append("%d: %.3f"%(i, SP_score((xs, ys))))
            if labels is None:
                label = f
            else:
                label = labels[i]
            ax.plot(xs, ys, marker='o', label="%d: %s"%(i, label), markersize=2)

    ax.text(0.1, 0.5, "N True Voxels: " + ", ".join(n_T), transform = ax.transAxes)
    ax.text(0.1, 0.45, "N Active & True: "+ ", ".join(n_TP_naive), transform = ax.transAxes)
    ax.text(0.1, 0.4, "N Active but False : "+ ", ".join(n_FP_naive), transform = ax.transAxes)
    ax.text(0.1, 0.35, "Area under curve: "+ ", ".join(auc), transform = ax.transAxes)
    ax.legend()
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs. Purity (w.r.t. True and Active Voxels)")
    plt.show()
    #plt.savefig("plots_comp_yinf/comp_yinf.png")
    #plt.clf()

def plot_mc_stats(fig_pdg, fig_track, fig_histo, filter_mc=lambda _:True, stats_file = sys.argv[1]):
    MC_stats = {}
    PDG_ids = set()
    ax = fig_pdg.add_subplot()
    with np.load(stats_file, allow_pickle=True) as stats_f:
        xs = stats_f["xs"]
        ys = stats_f["ys"]
        ax.plot(xs, ys, marker='o', label="Event curve", markersize=2)
        n_active_voxels = stats_f["n_TP_naive"] + stats_f["n_FP_naive"]
        n_true_voxels = stats_f["n_T"]
        i = 0
        while "arr_%d"%i in stats_f:
            mc_stats = stats_f["arr_%d"%i]
            i += 1
            mc = tuple(mc_stats[0])
            if not filter_mc(mc): continue
            PDG_ids.add(mc[1])
            MC_stats[mc] = mc_stats[1:]
    if not MC_stats:
        print("NOTHING")
        return
    AUCs, N_active_voxels, N_true_voxels, labels = ["Event: %.3f"%SP_score((xs, ys))], ["Event: %.0f"%n_active_voxels], ["Event: %.0f"%n_true_voxels], [] 
    Xs_all, Ys_all, AUCs_histo = [], [], []
    for i, pdg_id in enumerate(sorted(PDG_ids)):
        Xs = np.array([stats[0] for mc, stats in MC_stats.items() if mc[1] == pdg_id])
        Ys = np.array([stats[1] for mc, stats in MC_stats.items() if mc[1] == pdg_id])
        Xs_all.extend(Xs)
        Ys_all.extend(Ys)
        Sizes_active = np.array([stats[2] for mc, stats in MC_stats.items() if mc[1] == pdg_id])
        Sizes_true = np.array([mc[2] for mc, stats in MC_stats.items() if mc[1] == pdg_id])
        xs_avg = np.mean(Xs, axis=0)
        ys_avg = np.mean(Ys, axis=0)
        size_active_avg = np.mean(Sizes_active)
        size_true_avg = np.mean(Sizes_true)
        label = "%d: PDG id = %d, N Particles: %d"%(i, pdg_id, len(Xs))
        ax.plot(xs_avg, ys_avg, marker='o', label=label, markersize=2)
        AUCs.append("%d: %.3f"%(i, SP_score((xs_avg, ys_avg))))
        N_active_voxels.append("%d: %.0f"%(i, size_active_avg))
        N_true_voxels.append("%d: %.0f"%(i, size_true_avg))
        AUCs_histo.append([SP_score((xs, ys)) for xs, ys in zip(Xs, Ys)])
        labels.append(label)
    xs_all_avg = np.mean(Xs_all, axis=0)
    ys_all_avg = np.mean(Ys_all, axis=0)
    ax.plot(xs_all_avg, ys_all_avg, marker='o', label="All ptcls: N=%d"%(len(Xs_all)), markersize=2)
    AUCs.append("All ptcls: %.3f"%(SP_score((xs_all_avg, ys_all_avg))))
    
    ax.text(0.1, 0.5, "Average N True Voxels: "+", ".join(N_true_voxels), transform = ax.transAxes)
    ax.text(0.1, 0.4, "Average N Active Voxels: "+", ".join(N_active_voxels), transform = ax.transAxes)
    ax.text(0.1, 0.3, "Area under curve: "+", ".join(AUCs), transform = ax.transAxes)
    ax.legend()
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs. Purity (All) (w.r.t. True and Active Voxels)\n stats file: %s"%stats_file[6:])
    
    ax = fig_histo.add_subplot()
    AUCs_histo.append(np.concatenate(AUCs_histo))
    labels.append("All ptcls: N=%d"%len(AUCs_histo[-1]))
    labels = [label+"\nMean/SD: %.3f/%.3f"%(np.mean(auc), np.std(auc)) for label, auc in zip(labels, AUCs_histo)]
    _, bins, _ = ax.hist(AUCs_histo, histtype='step', label=labels, bins=np.linspace(0.5, 1, 11), weights=[np.ones(len(auc))/len(auc) for auc in AUCs_histo])
    ax.set_xticks(bins)
    ax.invert_xaxis()
    ax.legend()
    ax.set_title("AUC Histo\n stats file: %s"%stats_file[6:])
    
    ax = fig_track.add_subplot()
    AUCs, Thetas, Phis = [], [], []
    for mc, stats in MC_stats.items():
        track_stats = mc[3]
        if track_stats is None: continue
        AUCs.append(SP_score((stats[0], stats[1])))
        Thetas.append(track_stats[0])
        Phis.append(track_stats[1])
    cmap=plt.get_cmap('viridis')
    fig_track.colorbar(ax.scatter(Thetas, Phis, s=5, c=AUCs, cmap=cmap))
    ax.text(0.1, 0.1, "N particles: %d"%len(AUCs))
    ax.set_xlabel("Theta (+z in reverse drift direction)")
    ax.set_ylabel("Phi (+y upwards, +x in beam direction)")
    ax.set_title("AUC vs. Polar Angle (All) \n stats file: %s"%stats_file[6:])
    plt.show()
    
def plot_segmented_stats(fig, fig_bar, stats_file = sys.argv[1]):
    N_YSEG = 2
    N_ZSEG = 12
    Z0 = 0
    ax = fig.add_subplot()
    axs_bar = [fig_bar.add_subplot(N_YSEG, 1, N_YSEG-i) for i in range(N_YSEG)]
    x_bar = np.arange(N_ZSEG)
    def autolabel(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        size=8,
                        textcoords="offset points",
                        ha='center', va='bottom')
    N_T, N_TP_naive, N_FP_naive, AUC = [], [], [], []
    with np.load(stats_file, allow_pickle=True) as stats_f:
        xs = stats_f["xs"]
        ys = stats_f["ys"]
        ax.plot(xs, ys, marker='o', label="Event curve", markersize=2)
        N_T.append("Event: %.1f"%stats_f["n_T"])
        N_TP_naive.append("Event: %.1f"%stats_f["n_TP_naive"])
        N_FP_naive.append("Event: %.1f"%stats_f["n_FP_naive"])
        auc_event = SP_score((xs, ys))
        AUC.append("Event: %.3f"%auc_event)
        i = 0
        for y_i in range(N_YSEG):
            AUC_bar, N_TP_naive_bar = [], []
            ax_bar = axs_bar[y_i]
            for z_i in range(Z0, Z0+N_ZSEG):
                xs, ys, n_T, n_TP_naive, n_FP_naive = stats_f["%d_%d"%(y_i, z_i)]
                N_T.append("%d: %.1f"%(i, n_T))
                N_TP_naive.append("%d: %.1f"%(i, n_TP_naive))
                N_TP_naive_bar.append("%d\n %.1f"%(z_i, n_TP_naive))
                N_FP_naive.append("%d: %.1f"%(i, n_FP_naive))
                auc = SP_score((xs, ys))
                AUC_bar.append(auc)
                AUC.append("%d: %.3f"%(i, auc))
                label = "y_i: %d, z_i: %d" % (y_i, z_i)
                ax.plot(xs, ys, marker='o', label="%d: %s"%(i, label), markersize=2)
                i += 1
            autolabel(ax_bar, ax_bar.bar(x_bar, AUC_bar))
            ax_bar.set_ylim(0.5, 1)
            ax_bar.set_xticks(x_bar)
            ax_bar.set_xticklabels(N_TP_naive_bar)
            ax_bar.set_title("y_i = %d"%y_i)

    ax.text(0.1, 0.5, "N True Voxels: " + ", ".join(N_T), transform = ax.transAxes)
    ax.text(0.1, 0.45, "N Active & True: "+ ", ".join(N_TP_naive), transform = ax.transAxes)
    ax.text(0.1, 0.4, "N Active but False : "+ ", ".join(N_FP_naive), transform = ax.transAxes)
    ax.text(0.1, 0.35, "Area under curve: "+ ", ".join(AUC), transform = ax.transAxes)
    ax.legend()
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_title("Segmented Efficiency vs. Purity (w.r.t. True and Active Voxels)\n stats file: %s"%stats_file[6:])
    fig_bar.suptitle("AUC vs. Z (Event AUC: %.3f)\n stats file: %s"%(auc_event, stats_file[6:]))
    plt.show()
    
def plot_yinf(fig, fig_histo=None, fig_inf=None, fig_histo_inf=None, event=1, downsample=(1,1,1), thres=None, true_thres=0, plot_slice=False, plot_lims=None, view_angle=None, xy_dir = sys.argv[1], yinf_file=sys.argv[2]):
    xy_file = get_xy_file(xy_dir, yinf_file)
    voxel_truth, voxel_active, event_info = parse_xy(event, 1/3, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)
    _, voxel_T_active, _, _ = comp_voxels(voxel_truth, voxel_active)
    voxel_yinf = parse_yinf(event_info, yinf_file)

    thres, plot_lims, evt_purity, evt_sensitivity, evt_voxels_comp, purity, sensitivity, voxels_comp_T, voxels_comp_inf, voxel_T_active, voxel_yinf = \
            inference_analysis(voxel_T_active, voxel_yinf, thres, plot_lims, downsample, plot_slice)

    evt_voxel_FN, evt_voxel_TP, _, _ = evt_voxels_comp
    voxel_FN, voxel_TP, _, _ = voxels_comp_T
    x_lim, y_lim, z_lim = plot_lims

    if fig is not None:
        scatter_voxels_comp(fig, voxels_comp_T, "(Active&True) [MeV]", "Positive [MeV]", plot_slice, view_angle)
        fig.text(0, 0.95, "True Threshold: %.2f"%true_thres)
        fig.text(0., 0.9, "Prediction Threshold: %.2f"%thres)
        fig.text(0., 0.85, "EVENT STATS: ")
        fig.text(0., 0.8, "Sensitivity: %.2f, Purity: %.2f"%(evt_sensitivity, evt_purity))
        fig.text(0., 0.75, "E TP: %.2f, E FN: %.2f"%(sum(evt_voxel_TP.values()), sum(evt_voxel_FN.values())))
        fig.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig.text(0., 0.6, "Sensitivity: %.2f, Purity: %.2f"%(sensitivity, purity))
        fig.text(0., 0.55, "E TP: %.2f, E FN: %.2f"%(sum(voxel_TP.values()), sum(voxel_FN.values())))
        fig.suptitle("(Active&True) vs. Positive [Energy]\nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
        fig.show()

    if fig_histo is not None:
        histo_voxels_comp(fig_histo, fig_histo.add_subplot(), voxels_comp_T, "(Active&True) [MeV]", "Positive [MeV]")
        fig_histo.text(0.4, 0.8, "True Threshold: %.2f"%true_thres)
        fig_histo.text(0.4, 0.75, "Prediction Threshold: %.2f"%thres)
        fig_histo.suptitle("Energy Histo (Active&True) vs. Positive\nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
        fig_histo.show()

    if fig_inf is not None:
        scatter_voxels_comp(fig_inf, voxels_comp_inf, "(Active&True) [Sigmoid]", "Positive [Sigmoid]", plot_slice, view_angle)
        fig_inf.text(0, 0.95, "True Threshold: %.2f"%true_thres)
        fig_inf.text(0., 0.9, "Prediction Threshold: %.2f"%thres)
        fig_inf.text(0., 0.85, "EVENT STATS: ")
        fig_inf.text(0., 0.8, "Sensitivity: %.2f, Purity: %.2f"%(evt_sensitivity, evt_purity))
        fig_inf.text(0., 0.65, "PLOTTED STATS: "+("(Slice X = %d - %d)"%(x_lim[0], x_lim[1]) if plot_slice else ""))
        fig_inf.text(0., 0.6, "Sensitivity: %.2f, Purity: %.2f"%(sensitivity, purity))
        fig_inf.suptitle("(Active&True) vs. Positive [Sigmoid]\nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
        fig_inf.show()

    if fig_histo_inf is not None:
        ax_comp = fig_histo_inf.add_subplot(211)
        histo_voxels_comp(fig_histo_inf, ax_comp, voxels_comp_inf, "(Active&True) [Sigmoid]", "Positive [Sigmoid]")
        fig_histo_inf.text(0.4, 0.8, "True Threshold: %.2f"%true_thres)
        fig_histo_inf.text(0.4, 0.75, "Prediction Threshold: %.2f"%thres)
        ax_inf = fig_histo_inf.add_subplot(212)
        histo_voxels(fig_histo_inf, ax_inf, (voxel_yinf, set_voxel_vals(voxel_T_active, voxel_yinf)),
                ("All Inference [Sigmoid]", "Active&True [Sigmoid]"), ("black", "b"), 
                "All Inference [Sigmoid]", 10, True)
        fig_histo_inf.suptitle("Sigmoid Histo \nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
        fig_histo_inf.show()
    return plot_lims, thres

def plot_yinf_notruth(fig, event=1, plot_active=True, downsample=(1,1,1), thres=0.5, thres_active=0, plot_slice=False, plot_lims=None, view_angle=None, xy_dir = sys.argv[1], yinf_file=sys.argv[2]):
    xy_file = get_xy_file(xy_dir, yinf_file)
    _, voxel_active, event_info = parse_xy(event, 1, xy_file)
    voxel_P = filter_voxel_val(parse_yinf(event_info, yinf_file), thres)
    voxel_active, voxel_P = downsample_voxels(downsample, *filter_voxels_coord(plot_lims, voxel_active, voxel_P))
    voxel_active = filter_voxel_val(voxel_active, thres_active)
    voxel_N = comp_voxels(voxel_active, voxel_P)[0]
    if plot_slice:
         voxel_active, voxel_P, voxel_N = downsample_voxels((-1,1,1), voxel_active, voxel_P, voxel_N)
    projection = "rectilinear" if plot_slice else "3d"
    if plot_active:
        scatter_voxel(fig, fig.add_subplot(311, projection=projection), voxel_P, "Positive [Sigmoid]", plot_slice, view_angle)
        scatter_voxel(fig, fig.add_subplot(312, projection=projection), voxel_active, "Active [Channel Val]", plot_slice, view_angle)
        scatter_voxel(fig, fig.add_subplot(313, projection=projection), voxel_N, "Active but not Positive [Channel Val]", plot_slice, view_angle)
    else:
        scatter_voxel(fig, fig.add_subplot(projection=projection), voxel_P, "Positive ", plot_slice, view_angle)
    fig.text(0.1, 0.75, "Prediction Threshold: %.2f"%thres)
    if type(yinf_file)!=str: yinf_file = yinf_file[0]
    fig.suptitle("Inference (no truth) \nfile: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
    fig.show()
    return voxel_P
    
def plot_yinf_simple(fig, fig_comp=None, event=1, thres=0.5, E_scale=1, true_thres=0, plot_lims=None, xy_dir = sys.argv[1], yinf_file=sys.argv[2]):
    xy_file = get_xy_file(xy_dir, yinf_file)
    voxel_truth, _, event_info = parse_xy(event, E_scale, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)
    voxel_P = filter_voxel_val(parse_yinf(event_info, yinf_file), thres)
    voxel_truth, voxel_P = filter_voxels_coord(plot_lims, voxel_truth, voxel_P)
    coord_lims = get_voxels_lims(voxel_truth, voxel_P)
    scatter_voxel(fig, fig.add_subplot(211, projection="3d"), voxel_truth, "True Voxels",  plot_lims=coord_lims)
    scatter_voxel(fig, fig.add_subplot(212, projection="3d"), voxel_P, "Positive Voxels", plot_lims=coord_lims)
    if fig_comp is not None:
        scatter_voxels_comp(fig_comp, comp_voxels(voxel_truth, voxel_P), "True", "Positive")
    fig.text(0.1, 0.75, "True Threshold: %.2f"%true_thres)
    fig.text(0.1, 0.5, "Purity: %.3f \n Sensitivity: %.3f"%SP_stats(voxel_truth, voxel_P)[:2])
    fig.text(0.1, 0.25, "Inference Threshold: %.2f"%thres)
    if type(yinf_file)!=str: yinf_file = yinf_file[0]
    fig.suptitle("Inference \nfile: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
    fig.show()
    
if __name__=="__main__":
    for i in range(10):
        print(i)
        #plot_yinf(plt.figure(), 0, plot_slice=True, plot_lims=((500+i*20, 500+(i+1)*20), None, None))
    yinf_avg_stats(100, 5, -2, 1, "BeamCosmic", (1,1,1), 0.5/3, True, False, (None, "mc/reco_1GeV_BeamCosmic_xy-tpc1-mc.npz"), True, True, "stats/singleE-BeamCosmic-tpc1_ghost3D-164-lr0.01/inference", "all")
    #yinf_avg_stats(10, 5, -1, 159, None, 0.5/3, False, (1,1,1), False, None, True, "stats/BeamCosmic-tpc159-zseg-10_ghost3D-164-lr0.01/inference", False)
    #compare_yinf_stats(plt.figure()))
    #yinf_stats()
    #plot_yinf_simple(plt.figure(), plt.figure(), event=0, true_thres=0.5/3)
    plt.show()
