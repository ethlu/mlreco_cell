import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import square
from square import IDENTITY
import pixel
import itertools
import math
import numpy as np

def plot_simple_overlap(wires, tiling):
    angle, pitch, wire_0_offset, num_wires, w_map = wires
    min_x, max_x, min_y, max_y = tiling
    cos = np.cos(angle*np.pi/180)
    sin = np.sin(angle*np.pi/180)
    e1 = np.array([-sin, cos])
    e2 = np.array([cos, sin])
    x_offset = wire_0_offset/cos
    hop_x_wire = pitch/cos

    x = np.arange(min_x, max_x+1)
    y = np.arange(min_y, max_y+1)
    pts = itertools.product(x, y)

    fig, ax = plt.subplots()
    ax.scatter(*zip(*pts), marker='o', s=30, color='red') #https://stackoverflow.com/questions/9923378/plotting-a-grid-in-python
    ax.grid()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, num_wires)]
    for w in range(num_wires):
        offset = wire_0_offset + w*pitch
        yl = (offset - e2[0]*min_x)/e2[1]
        yr = (offset - e2[0]*max_x)/e2[1]
        ax.plot((min_x, max_x), (yl, yr))

        offset = wire_0_offset + (w-0.5)*pitch
        yr = (offset - e2[0]*5*max_x)/e2[1]
        lower_l = (5*max_x, yr)
        maxlen = np.linalg.norm((max_x-min_x, max_y-min_y))

        ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w], alpha=0.5, label="Region: "+str(w_map(w))))

    overlaps = square.wireplane_overlap(square.fixed_pitch_wireplane(wires), tiling)
    for k, v in overlaps.items():
        ax.text(k[0]+0.2, k[1] + 0.5, v, size=5)

    ax.legend()
    plt.show()

def plot_wrapped_wires(seed_wires, tiling):
    min_x, max_x, min_y, max_y = tiling
    x = np.arange(min_x, max_x+1)
    y = np.arange(min_y, max_y+1)
    pts = itertools.product(x, y)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    wrapped_wires = square.wrap_fixed_pitch_wires(seed_wires, tiling)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, seed_wires[3])]
    for wires in wrapped_wires:
        angle, pitch, wire_0_offset, num_wires, w_map = wires
        cos = np.cos(angle*np.pi/180)
        sin = np.sin(angle*np.pi/180)
        e1 = np.array([-sin, cos])
        e2 = np.array([cos, sin])
        x_offset = wire_0_offset/cos
        hop_x_wire = pitch/cos

        for w in range(num_wires):
            x_crossing = np.array((x_offset + w*hop_x_wire, 0))
            #ax.text(*x_crossing, "wire: "+str(w))
            maxlen = 2*np.linalg.norm((max_x-min_x, max_y-min_y))
            upper = x_crossing + maxlen*e1
            lower = x_crossing - maxlen*e1
            ax.plot(*zip(upper, lower), color = colors[w_map(w)])

            lower_l = lower - np.array((hop_x_wire/2, 0))
            ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w_map(w)], alpha=0.5))

    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.show()

def plot_merged_wireplane(wire_sets, tiling):
    min_x, max_x, min_y, max_y = tiling
    x = np.arange(min_x, max_x+1)
    y = np.arange(min_y, max_y+1)
    pts = itertools.product(x, y)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.scatter(*zip(*pts), marker='o', s=30, color='red') 

    for wires in wire_sets:
        angle, pitch, wire_0_offset, num_wires, w_map = wires
        cos = np.cos(angle*np.pi/180)
        sin = np.sin(angle*np.pi/180)
        e1 = np.array([-sin, cos])
        e2 = np.array([cos, sin])
        x_offset = wire_0_offset/cos
        hop_x_wire = pitch/cos

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, num_wires)]

        for w in range(num_wires):
            x_crossing = np.array((x_offset + w*hop_x_wire, 0))
            #ax.text(*x_crossing, "wire: "+str(w))
            maxlen = 2*np.linalg.norm((max_x-min_x, max_y-min_y))
            upper = x_crossing + maxlen*e1
            lower = x_crossing - maxlen*e1
            ax.plot(*zip(upper, lower), color = colors[w_map(w)])

            lower_l = lower - np.array((hop_x_wire/2, 0))
            ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w_map(w)], alpha=0.5, label="Region: "+str(w_map(w))))

    overlaps = square.wireplane_overlap(square.merged_fp_wireplane(wire_sets), tiling)
    for k, v in overlaps.items():
        ax.text(k[0]+0.2, k[1] + 0.5, v, size=5)

    ax.legend()
    plt.show()

def plot_pixelator(wire_sets, tiling, channel_vals):
    min_x, max_x, min_y, max_y = tiling

    x = np.arange(min_x, max_x+1)
    y = np.arange(min_y, max_y+1)
    pts = itertools.product(x, y)

    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [50, 1]})
    ax = axes[0]
    ax.grid()
    ax.set_xlim(min_x-2, max_x+2)
    ax.set_ylim(min_y-2, max_y+2)
    ax.scatter(*zip(*pts), marker='o', s=30, color='red') 

    for wires in wire_sets:
        angle, pitch, wire_0_offset, num_wires, w_map = wires
        cos = np.cos(angle*np.pi/180)
        sin = np.sin(angle*np.pi/180)
        e1 = np.array([-sin, cos])
        e2 = np.array([cos, sin])
        x_offset = wire_0_offset/cos
        hop_x_wire = pitch/cos

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 3*num_wires)]

        for w in range(num_wires):
            offset = wire_0_offset + w*pitch
            label = "Wire: "+str(w_map(w)) + ", Channel val: " + str(channel_vals[w_map(w)])
            if angle == 0:
                xu = (offset - e2[1]*max_y)/e2[0]
                xd = (offset - e2[1]*min_y)/e2[0]
                ax.plot((xu, xd), (max_y, min_y), label=label)
                ax.text(xu-.5, max_y+1, label, size = 8)
                offset = wire_0_offset + (w-0.5)*pitch
                xd = (offset - e2[1]*(min_y-2))/e2[0]
                lower_l = (xd, min_y-2)
            else:
                yl = (offset - e2[0]*min_x)/e2[1]
                yr = (offset - e2[0]*max_x)/e2[1]
                ax.plot((min_x, max_x), (yl, yr), label=label)
                offset = wire_0_offset + (w-0.5)*pitch
                if angle > 0:
                    ax.text(max_x+1, yr-0.5, label, size = 8)
                    yr = (offset - e2[0]*(2+max_x))/e2[1]
                    lower_l = (max_x+2, yr)
                else:
                    ax.text(min_x-2, yl, label, size = 8)
                    yl = (offset - e2[0]*(min_x-2))/e2[1]
                    lower_l = (min_x-2, yl)

            maxlen = np.linalg.norm((max_x-min_x, max_y-min_y))
            ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = cmap(channel_vals[w_map(w)]/10), alpha=0.5))

    pix = pixel.Pixelator(pixel.Geom.create(wire_sets, tiling, len(channel_vals)), active_none = False)
    img = pix(channel_vals)
    #pix = pixel.Pixelator(pixel.Geom.create(wire_sets, tiling, len(channel_vals)), sparse_output = True)
    #img = pix.sparse_to_dense(pix(channel_vals))
    x = np.arange(min_x, max_x)
    y = np.arange(min_y, max_y)
    pts = itertools.product(x, y)
    for x, y in pts:
        ax.text(x+0.2, y + 0.5, [round(i, 3) for i in img[x+min_x, y+min_y]], size=5)

    colors = cmap(np.arange(cmap.N))
    axes[1].imshow([colors], extent=[0, 10, 0, 1])

    #ax.legend(bbox_to_anchor=(.9, 1.05))
    ax.legend()
    plt.show()

#print(plot_simple_overlap((90, 1.8, -5.1, 7, lambda w: w-3), (-5, 5, -5, 5)))
#print(plot_simple_overlap((45, math.sqrt(2)+0.0001, 0, 5, square.IDENTITY), (-5, 5, -10, 10)))
DUNE_TILE = (0, 10, -26, 0) 
#plot_wrapped_wires((35.7, 1, 0.01, 8, IDENTITY), DUNE_TILE)
V_SEED_WIRE = square.shift_wires_origin((-35.7+180, 2, -0.001, 4, IDENTITY), (10, 0), (0, 0))
#print(V_SEED_WIRE)
#plot_wrapped_wires(V_SEED_WIRE, DUNE_TILE)
#plot_wrapped_wires((-35.7, 2, 2, 4, IDENTITY), (0, 10, -26, 0))
wrapped = square.wrap_fixed_pitch_wires((35.7, 2, 0.01, 4, IDENTITY), DUNE_TILE)
pos_wrapped = [wires for wires in wrapped if wires[0]<0]
#plot_merged_wireplane(pos_wrapped, DUNE_TILE)
channel_vals = [0, 7,6, 4]
np.random.shuffle(channel_vals)
channel_vals2 = [0,3, 9, 5]
np.random.shuffle(channel_vals2)
channel_vals3 = [0,2, 9, 6]
np.random.shuffle(channel_vals3)
#plot_pixelator([(0, 1, 0.5, 4, IDENTITY), (90, 1, 0.5, 4, lambda w: w+4)], (0, 4, 0, 4), np.append(channel_vals, channel_vals2))
#plot_pixelator([(60, 2, -3.501, 4, IDENTITY), (-60, 2, -3.501, 4, lambda w: w+4), (0, 2,-3, 4, lambda w: w+8)], (-4, 4, -4, 4), np.concatenate((channel_vals, channel_vals2, channel_vals3)))
