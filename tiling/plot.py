import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import square
from square import IDENTITY
import itertools
import math
import numpy as np

def plot_simple_overlap(wire_plane, tiling):
    angle, pitch, wire_0_offset, num_wires, w_map = wire_plane
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
        x_crossing = np.array((x_offset + w*hop_x_wire, 0))
        #ax.text(*x_crossing, "wire: "+str(w))
        maxlen = np.linalg.norm((max_x-min_x, max_y-min_y))
        upper = x_crossing + maxlen*e1
        lower = x_crossing - maxlen*e1
        ax.plot(*zip(upper, lower))

        lower_l = lower - np.array((hop_x_wire/2, 0))
        ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w], alpha=0.5, label="Region: "+str(w_map(w))))

    overlaps = square.wireplane_overlap(square.fixed_pitch_wireplane(wire_plane), tiling)
    for k, v in overlaps.items():
        ax.text(k[0]+0.2, k[1] + 0.5, v, size=5)

    ax.legend()
    plt.show()

def plot_wrapped_wires(wrapped_wires, tiling):
    min_x, max_x, min_y, max_y = tiling
    x = np.arange(min_x, max_x+1)
    y = np.arange(min_y, max_y+1)
    pts = itertools.product(x, y)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    for wire_plane in wrapped_wires:
        angle, pitch, wire_0_offset, num_wires, w_map = wire_plane
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
            ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w_map(w)], alpha=0.5))

    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.show()

print(plot_simple_overlap((35, 1.8, -5.1, 7, lambda w: w-3), (-5, 5, -5, 5)))
#print(plot_simple_overlap((45, math.sqrt(2)+0.0001, 0, 5, square.IDENTITY), (-5, 5, -10, 10)))
DUNE_TILE = (0, 10, -26, 0) 
plot_wrapped_wires(square.wrap_fixed_pitch_wires((35.7, 1, 0.01, 8, IDENTITY), DUNE_TILE), DUNE_TILE)
