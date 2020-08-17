import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from data.util import *
from analysis.util import *

N_VOXEL = (3585//(1.565/2)//8)*(5984//2//4)*(2300//2//4)
def yinf_stats(event = 1, thresholds = np.arange(50)*2E-2, true_thres=0, xy_file=sys.argv[1], yinf_file=sys.argv[2]):
    dict_truth, coords_active, event_starts = parse_xy(event, xy_file=xy_file, true_thres=true_thres)

    yinf_batch = np.load(yinf_file)[0]
    n_events = len(event_starts)
    if event == n_events-1:
        event_yinf = yinf_batch[event_starts[event]:]
    else:
        event_yinf = yinf_batch[event_starts[event]:event_starts[event+1]]
    event_yinf = {tuple(pt):val[0] for pt, val in zip(coords_active, event_yinf)}
    
    coords_active = set(coords_active)
    n_active = len(coords_active)
    coords_T = set(dict_truth.keys())
    n_T = len(coords_T)
    coords_T_active = coords_T.intersection(coords_active)

    n_TP_naive = len(coords_T_active) #naive: treat all active as positive
    n_FP_naive = n_active - n_TP_naive
    coords_FN_naive = coords_T.difference(coords_T_active)
    n_FN_naive = len(coords_FN_naive)
    sensitivity_naive = n_TP_naive/(n_TP_naive + n_FN_naive)
    #print("sensitivity naive: ", sensitivity_naive) #upper bound sensitivity

    xs, ys = [], []
    for thres in thresholds:
        x, y = P_stats(set_thres(event_yinf, thres), coords_T_active, N_VOXEL)
        xs.append(x)
        ys.append(y)
    return xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive

def select_threshold(xs, ys, thresholds):
    sums = np.array(xs) + np.array(ys)
    max_i = np.argmax(sums)
    return thresholds[max_i], xs[max_i], ys[max_i]

def yinf_avg_stats(n_yinf_files = 10, n_events=50, epoch = -1, true_thres = 0, stats_dir = "stats", plot = False, xy_dir = sys.argv[1], yinf_dir=sys.argv[2]):
    yinf_files = os.listdir(yinf_dir)
    yinf_epoch = files_info(yinf_files, [0], inf_file_info)
    epoch = sorted(yinf_epoch.keys())[epoch]
    yinf_files = yinf_epoch[epoch][:n_yinf_files]
    stats_file = stats_dir+"/epoch%d_stats-thres%.2f-N%d.npz"%(epoch[0], true_thres, n_events*n_yinf_files)
    if os.path.exists(stats_file):
        print(stats_file + " Exists")
        return

    print("True Threshold: ", true_thres)
    Xs, Ys, N_T, N_TP_naive, N_FP_naive = [], [], [], [], []
    for yinf_file in yinf_files:
        print(yinf_file)
        _, f = inf_file_info(yinf_file)
        #xy_file = xy_dir + '/' + sorted(os.listdir(xy_dir))[yinf_batch+n_train]
        xy_file = xy_dir + '/' + f.replace("yinf", "xy") + '.npz'
        for event in range(n_events):
            xs, ys, thresholds, n_T, n_TP_naive, n_FP_naive = yinf_stats(event, true_thres = true_thres, yinf_file=yinf_dir+'/'+yinf_file, xy_file=xy_file)
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

def plot_yinf(fig, event=1, save_file=None, thres=None, true_thres=0, plot_lims=None, view_angle=None, xy_dir = sys.argv[1], yinf_file=sys.argv[2]):
    TRUE_THRESHOLD=true_thres
    _, f = inf_file_info(yinf_file)
    xy_file = xy_dir + '/' + f.replace("yinf", "xy") + '.npz'
    dict_truth, coords_active, event_starts = parse_xy(event, xy_file=xy_file, true_thres=TRUE_THRESHOLD)

    yinf_batch = np.load(yinf_file)[0]
    n_events = len(event_starts)
    if event == n_events-1:
        event_yinf = yinf_batch[event_starts[event]:]
    else:
        event_yinf = yinf_batch[event_starts[event]:event_starts[event+1]]
    event_yinf = {tuple(pt):val[0] for pt, val in zip(coords_active, event_yinf)}
    
    coords_active = set(coords_active)
    coords_T = set(dict_truth.keys())
    coords_T_active = coords_T.intersection(coords_active)

    xs, ys, thresholds, _, _, _ = yinf_stats(event, true_thres=true_thres, xy_file=xy_file, yinf_file=yinf_file)
    if thres is None:
        SIGMOID_THRESHOLD, purity, sensitivity = select_threshold(xs, ys, thresholds)
    else: 
        purity, sensitivity = P_stats(set_thres(event_yinf, thres), coords_T_active, N_VOXEL)
        SIGMOID_THRESHOLD = thres

    x_lim, y_lim, z_lim = compare_voxels(fig, coords_T_active, set_thres(event_yinf, SIGMOID_THRESHOLD).keys(), "(Active&True)", "Positive", plot_lims, view_angle)
    fig.text(0.05, 0.7, "True Threshold: %f"%TRUE_THRESHOLD)
    fig.text(0.05, 0.6, "Prediction Threshold: %f"%SIGMOID_THRESHOLD)
    fig.text(0.05, 0.5, "Sensitivity: %.2f, Purity: %.2f"%(sensitivity, purity))
    fig.suptitle("(Active&True) vs. Positive\nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
    Size = fig.get_size_inches()
    if save_file is None:
        plt.show()
    else:
        fig.set_size_inches(Size[0]*2, Size[1]*2, forward=True)
        plt.savefig("plots_yinf/%s_Evt%d.png"%(save_file, event))
    return x_lim, y_lim, z_lim, SIGMOID_THRESHOLD

if __name__=="__main__":
    for i in range(5):
        print(i)
        #plot_yinf(plt.figure(), i)
    #yinf_avg_stats(n_yinf_files = 5, epoch=-1)
    compare_yinf_stats(plt.figure())
