import numpy as np
import os, sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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
        
def compare_xy(event = 1, xy_file = sys.argv[1]):
    with np.load(xy_file, allow_pickle=True) as xy_f:
        pix_batch=xy_f["X"]
        energy_batch=xy_f["Y"]
        event_starts=xy_f["starts"]
        event_pixels = np.array(pix_batch[event_starts[event]:event_starts[event+1]])
        event_energies = np.array(energy_batch[event_starts[event]:event_starts[event+1]])
        event_energys_truth = xy_f["Y_truth"][event]
    sub = 1
    active_pixels = [tuple(pt//sub) for pt in event_pixels[:, :3]]
    active_energys_truth = {(pt[0]//sub, pt[1]//sub, pt[2]//sub):pt[3]  for pt in event_energys_truth}
    total_E_truth = sum(active_energys_truth.values())
    print("total E truth", total_E_truth)
    active_E_coords_truth = set(active_energys_truth.keys())
    print("active energys_truth: ",len(active_E_coords_truth))
    active_energys = {pt:energy for pt, energy in zip(active_pixels, event_energies) if energy[0] != 0}
    total_E = sum(active_energys.values())
    print("total E", total_E)
    active_E_coords = set(active_energys.keys())
    print("active energys: ",len(active_E_coords))
    #print(sorted(list(active_E_coords)))
    active_pixels = set(active_pixels)
    print("active pixels: ",len(active_pixels))
    #print(sorted(list(active_pixels)))
    intersect = active_E_coords_truth.intersection(active_pixels)
    print("intersect_truth: ", len(intersect))
    missing = active_E_coords_truth.difference(intersect)
    missing_E = sum([active_energys_truth[coord] for coord in missing])
    print("missing E _truth", missing_E)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    intersect = np.array([list(pt) for pt in intersect])
    missing = np.array([list(pt) for pt in missing])
    active_pixels = np.array([list(pt) for pt in active_pixels])
    ax = plt.subplot(211, projection="3d")
    if len(intersect) > 0:
        ax.scatter3D(*intersect.T)
    if len(missing) > 0:
        ax.scatter3D(*missing.T)
    ax2 = plt.subplot(212, projection="3d")
    ax2.scatter3D(*active_pixels.T)
    plt.show()

def yinf_stats(yinf_filter, n_train = 200, xy_dir = sys.argv[1], yinf_dir=sys.argv[2]):
    yinf_files = [f for f in os.listdir(yinf_dir) if yinf_filter(f)]
    Xs, Ys, N_P, N_TP_naive, N_FP_naive = [], [], [], [], []
    for yinf_file in yinf_files:
        yinf_batch = int(yinf_file[yinf_file.find("batch")+5: yinf_file.find(".")])
        print(yinf_batch)
        xy_file = xy_dir + '/' + sorted(os.listdir(xy_dir))[yinf_batch+n_train]
        print(xy_file)
        with np.load(xy_file, allow_pickle=True) as xy_f:
            pix_batch=xy_f["X"]
            energy_batch=xy_f["Y"]
            energys_truth = xy_f["Y_truth"]
            event_starts=xy_f["starts"]
        n_events = len(event_starts)
        yinf_batch = np.load(yinf_dir+'/'+yinf_file)[0]
        for event in range(n_events):
            print(event)
            if event == n_events-1:
                event_pixels = pix_batch[event_starts[event]:]
                event_energies = energy_batch[event_starts[event]:]
                event_yinf = yinf_batch[event_starts[event]:]
            else:
                event_pixels = pix_batch[event_starts[event]:event_starts[event+1]]
                event_energies = energy_batch[event_starts[event]:event_starts[event+1]]
                event_yinf = yinf_batch[event_starts[event]:event_starts[event+1]]
            event_energys_truth = {(pt[0], pt[1], pt[2]):pt[3]  for pt in energys_truth[event]}
            active_pixels = [tuple(pt) for pt in event_pixels[:, :3]]
            energys_active = {pt:energy[0] for pt, energy in zip(active_pixels, event_energies) if energy[0] != 0}
            yinf = {pt:energy[0] for pt, energy in zip(active_pixels, event_yinf)}

            E_P = sum(event_energys_truth.values())
            #print("P E", E_P)
            coords_P = set(event_energys_truth.keys())
            n_P = len(coords_P)
            #print("P N: ", n_P)

            #total_E_active = sum(energys_active.values())
            #print("total E", total_E)
            coords_PY_naive = set(active_pixels)
            n_PY_naive = len(coords_PY_naive)
            coords_T_active = set(energys_active.keys())
            n_TP_naive = len(coords_T_active)
            n_FP_naive = n_PY_naive - n_TP_naive
            coords_FN_naive = coords_P.difference(coords_T_active)
            n_FN_naive = len(coords_FN_naive)
            #print("FN_naive N: ", n_FN_naive)
            sensitivity_naive = n_TP_naive/(n_TP_naive + n_FN_naive)
            #print("sensitivity naive: ", sensitivity_naive) #upper bound sensitivity

            #print("active energys: ",len(active_E_coords))
            #print(sorted(list(active_E_coords)))
            #total_E_yinf = sum(yinf.values())
            #print("FP E ",total_E_yinf - total_E)
            #active_yinf_coords = set(active_yinf.keys())
            #print("active yinf: ",len(active_yinf_coords))
            #print(sorted(list(active_pixels)))

            N_VOXEL = (3585//(1.565/2)//8)*(5984//2//4)*(2300//2//4)
            thresholds = np.arange(100)*1E-2
            #thresholds = [0.5]
            xs, ys = [], []
            for thres in thresholds:
                x, y = Y_thres_stats(yinf, thres, coords_T_active, N_VOXEL)
                xs.append(x)
                ys.append(y)
            Xs.append(xs)
            Ys.append(ys)
            N_P.append(n_P)
            N_TP_naive.append(n_TP_naive)
            N_FP_naive.append(n_FP_naive)
    xs = np.mean(Xs, 0)
    ys = np.mean(Ys, 0)
    n_P = np.mean(N_P)
    n_TP_naive = np.mean(N_TP_naive)
    n_FP_naive = np.mean(N_FP_naive)

    ax = plt.subplot()
    ax.plot(xs, ys, marker='o')
    ax.text(0.2, 0.5, "N True Voxels: %.2f"% n_P, transform = ax.transAxes)
    ax.text(0.2, 0.4, "N Active & True: %.2f"% n_TP_naive, transform = ax.transAxes)
    ax.text(0.2, 0.3, "N Active but False : %.2f"% n_FP_naive, transform = ax.transAxes)
    #ax.set_xlabel("N FP(Ghosts)")
    #ax.set_ylabel("N TP")
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    #ax.set_title("Efficiency vs. Purity (w.r.t. True and Active Voxels) \nInference file: %s, Event: %d"%(yinf_file[yinf_file.rfind('/')+1:], event))
    ax.set_title("Efficiency vs. Purity (w.r.t. True and Active Voxels)")
    #ax.set_xlim(0, 500)
    #ax.set_xlim(0, 500)
    #plt.show()
    plt.savefig("plots_ghost/ratio_%s_Evt%d.png"%(yinf_file[yinf_file.rfind('/')+1:], event))
    plt.clf()

    """
    #print("intersect_truth: ", len(intersect))
    missing = E_coords_truth.difference(intersect)
    missing_E = sum([event_energys_truth[coord] for coord in missing])
    print("E _truth", missing_E)
    intersect = np.array([list(pt) for pt in intersect])
    missing = np.array([list(pt) for pt in missing])
    active_pixels = np.array([list(pt) for pt in active_pixels])
    ax = plt.subplot(211, projection="3d")
    if len(intersect) > 0:
        ax.scatter3D(*intersect.T)
    if len(missing) > 0:
        ax.scatter3D(*missing.T)
    ax2 = plt.subplot(212, projection="3d")
    ax2.scatter3D(*active_pixels.T)
    #plt.show()
    """


def Y_thres_stats(Y, thres, coords_T, N): 
    #print("THRESHOLD: ", thres)
    n_T = len(coords_T)
    P = {pt: E for pt, E in Y.items() if E > thres}
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
    specificity = n_TN/(n_FP+n_TN)
    purity = n_TP/n_P
    """
    print("TP N: ", n_TP)
    print("FP N: ", n_FP)
    print("FN N: ", n_FN)
    print("TN N: ", n_TN) 
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    """
    #return sensitivity, 1-specificity
    return purity, sensitivity

for i in range(10):
    print(i)
    #compare_energys(i)
    #compare_channels(i)
    #compare_xy(i)
yinf_stats(lambda f: "epoch33" in f and int(f[f.find("batch")+5: f.find(".")])<5)
