import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *
from analysis.products import parse_xy

N_VOXEL = (3585//(1.565/2)//8)*(5984//2//4)*(2300//2//4)
def parse_yinf(event_info, yinf_file=sys.argv[1]):
    event_starti, event_endi, coords_active = event_info
    yinf_batch = np.load(yinf_file)[0]
    if event_endi == -1:
        event_yinf = yinf_batch[event_starti:]
    else:
        event_yinf = yinf_batch[event_starti:event_endi]
    voxel_yinf = {tuple(pt):val[0] for pt, val in zip(coords_active, event_yinf)}
    return voxel_yinf

def get_xy_file(xy_dir, yinf_file, old=False):
    if old:
        return xy_dir + '/' + sorted(os.listdir(xy_dir))[int(yinf_file[yinf_file.find("batch")+5:-4])+200]
    _, f = inf_file_info(yinf_file)
    xy_file = xy_dir + '/' + f.replace("yinf", "xy") + '.npz'
    return xy_file

def yinf_stats(event = 1, true_thres=0, T_weighted=False, downsample=(1,1,1), xy_file=sys.argv[1], yinf_file=sys.argv[2]):
    voxel_truth, voxel_active, event_info = parse_xy(event, 1, xy_file)
    voxel_truth = filter_voxel_val(voxel_truth, true_thres)

    voxel_FN, voxel_T_active, _, voxel_FP = comp_voxels(voxel_truth, voxel_active)

    n_T = len(voxel_truth)
    n_TP_naive = len(voxel_T_active) #naive: treat all active as positive
    n_FP_naive = len(voxel_FP)

    voxel_yinf = parse_yinf(event_info, yinf_file)
    xs, ys, thresholds = SP_curve(voxel_T_active, voxel_yinf, T_weighted, downsample)

    return xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive

def yinf_avg_stats(n_yinf_files = 10, n_events=50, epoch = -1, true_thres = 0, T_weighted=False, downsample=(1,1,1), stats_dir = "stats", plot = False, xy_dir = sys.argv[1], yinf_dir=sys.argv[2]):
    yinf_files = os.listdir(yinf_dir)
    yinf_epoch = files_info(yinf_files, [0], inf_file_info)
    epoch = sorted(yinf_epoch.keys())[epoch]
    yinf_files = yinf_epoch[epoch][:n_yinf_files]
    stats_file = stats_dir+"/epoch%d_stats-thres%.2f%s%s-N%d.npz"% \
        (epoch[0], true_thres, "-weighted" if T_weighted else "", "-downsample%s"%str(downsample) if downsample!=(1,1,1) else "", n_events*n_yinf_files)
    if os.path.exists(stats_file):
        print(stats_file + " Exists")
        return

    print("True Threshold: ", true_thres)
    Xs, Ys, N_T, N_TP_naive, N_FP_naive = [], [], [], [], []
    for yinf_file in yinf_files:
        print(yinf_file)
        xy_file = get_xy_file(xy_dir, yinf_file)
        for event in range(n_events):
            xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive = \
                yinf_stats(event, true_thres, T_weighted, downsample, xy_file, yinf_file=yinf_dir+'/'+yinf_file)
            Xs.append(xs)
            Ys.append(ys)
            N_T.append(n_T)
            N_TP_naive.append(n_TP_naive)
            N_FP_naive.append(n_FP_naive)
    xs = np.mean(Xs, 0)
    ys = np.mean(Ys, 0)
    n_T = np.mean(N_T)
    n_TP_naive = np.mean(N_TP_naive)
    n_FP_naive = np.mean(N_FP_naive)

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

    np.savez(stats_file, xs=xs, ys=ys, thresholds=thresholds, n_T=n_T, n_TP_naive=n_TP_naive, n_FP_naive=n_FP_naive)
    return xs, ys, n_T, thresholds, n_TP_naive, n_FP_naive

def compare_yinf_stats(fig, stats_files = sys.argv[1:], labels=None):
    ax = fig.add_subplot()
    n_T, n_TP_naive, n_FP_naive = [], [], []
    for i, f in enumerate(stats_files):
        if f == "a":
            continue
        with np.load(f) as stats_f:
            xs = stats_f["xs"]
            ys = stats_f["ys"]
            if "n_T" in stats_f:
                n_T.append("%d: %.2f"%(i, stats_f["n_T"]))
            else:
                n_T.append("%d: %.2f"%(i, stats_f["n_P"]))
            n_TP_naive.append("%d: %.2f"%(i, stats_f["n_TP_naive"]))
            n_FP_naive.append("%d: %.2f"%(i, stats_f["n_FP_naive"]))
            if labels is None:
                label = f
            else:
                label = labels[i]
            ax.plot(xs, ys, marker='o', label="%d: %s"%(i, label), markersize=2)

    ax.text(0.1, 0.5, "N True Voxels: " + ", ".join(n_T), transform = ax.transAxes)
    ax.text(0.1, 0.4, "N Active & True: "+ ", ".join(n_TP_naive), transform = ax.transAxes)
    ax.text(0.1, 0.3, "N Active but False : "+ ", ".join(n_FP_naive), transform = ax.transAxes)
    ax.legend()
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency vs. Purity (w.r.t. True and Active Voxels)")
    plt.show()
    #plt.savefig("plots_comp_yinf/comp_yinf.png")
    #plt.clf()

def plot_yinf(fig, fig_histo, fig_inf, fig_histo_inf, event=1, downsample=(1,1,1), thres=None, true_thres=0, plot_slice=False, plot_lims=None, view_angle=None, xy_dir = sys.argv[1], yinf_file=sys.argv[2]):
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

if __name__=="__main__":
    for i in range(10):
        print(i)
        plot_yinf(plt.figure(), 0, plot_slice=True, plot_lims=((500+i*20, 500+(i+1)*20), None, None))
    #yinf_avg_stats(n_yinf_files = 5, epoch=-1)
    #compare_yinf_stats(plt.figure()))
    #yinf_stats()
