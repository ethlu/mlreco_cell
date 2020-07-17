import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import square
import itertools
import math
import numpy as np

def plot_simple_overlap(wire_plane, tiling):
    angle, pitch, wire_0_offset, min_wire, max_wire = wire_plane
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
    colors = [cmap(i) for i in np.linspace(0, 1, max_wire-min_wire+1)]
    for w in range(min_wire, max_wire+1):
        x_crossing = np.array((x_offset + w*hop_x_wire, 0))
        #ax.text(*x_crossing, "wire: "+str(w))
        maxlen = np.linalg.norm((max_x-min_x, max_y-min_y))
        upper = x_crossing + maxlen*e1
        lower = x_crossing - maxlen*e1
        ax.plot(*zip(upper, lower))

        lower_l = lower - np.array((hop_x_wire/2, 0))
        ax.add_patch(Rectangle(lower_l, pitch, 5*maxlen, angle, color = colors[w-min_wire], alpha=0.5, label="Region: "+str(w)))

    overlaps = square.simple_overlap(wire_plane, tiling)
    for k, v in overlaps.items():
        ax.text(k[0]+0.2, k[1] + 0.5, v, size=5)

    ax.legend()
    plt.show()

print(plot_simple_overlap((35, 1.8, 0.3, -3, 3), (-5, 5, -5, 5)))
