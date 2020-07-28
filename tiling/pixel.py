import numpy as np
from square import * 
from numba import jit, prange
from numba.typed import List
from functools import partial

from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class Geom:
    NONE = -1
    def __init__(self, pixel_overlaps, channel_associations):
        self.num_x, self.num_y = pixel_overlaps.shape
        self.num_chans, self.num_colors = channel_associations.shape
        self.num_chans -= 1 #last one is for None
        self.pix_ol = pixel_overlaps
        self.chan_assoc = channel_associations

    def save(self, file_name = "geom.npz"):
        np.savez_compressed(file_name, pix_ol = self.pix_ol, chan_assoc = self.chan_assoc)

    def to_numba(self):
        print("converting Geom to Numba")
        pix_ol = List()
        for x in range(self.num_x):
            xl = List()
            for y in range(self.num_y):
                yl = List()
                for ol in self.pix_ol[x, y]:
                    yl.append((float(ol[0]), float(ol[1]), float(ol[2])))
                xl.append(yl)
            pix_ol.append(xl)
        chan_assoc = List()
        for chan in range(self.num_chans+1):
            chl = List()
            for color in range(self.num_colors):
                chset = self.chan_assoc[chan][color]
                chl.append(chset if chset else {(Geom.NONE, Geom.NONE)})
            chan_assoc.append(chl)
        return pix_ol, chan_assoc, self.num_x, self.num_y, self.num_chans, self.num_colors, Geom.NONE

    @staticmethod
    def load(file_name = "geom.npz"):
        print("loading: ", file_name)
        geom = np.load(file_name, allow_pickle = True)
        return Geom(geom["pix_ol"], geom["chan_assoc"])

    @staticmethod
    def create(wire_sets, tiling, num_channels):
        wires_by_plane = dict() 
        for wires in wire_sets:
            angle = wires[0]
            try:
                wires_by_plane[angle].append(wires)
            except KeyError:
                wires_by_plane[angle] = [wires]
        overlaps = [wireplane_overlap(merged_fp_wireplane(wires_by_plane[i]), tiling) \
                for i in sorted(wires_by_plane.keys())]
        min_x, max_x, min_y, max_y = tiling 
        num_x, num_y, num_colors = max_x - min_x, max_y - min_y, len(overlaps)
        pix_ol = np.empty((num_x, num_y), dtype = object)
        chan_assoc = np.empty((num_channels+1, num_colors), dtype = object)
        for x in range(num_x):
            for y in range(num_y):
                pix_ol[x, y] = []
        for chan in range(num_channels+1):
            for color in range(num_colors):
                chan_assoc[chan, color] = set()
        for color, color_ol in enumerate(overlaps):
            for coord, ol in color_ol.items():
                x, y = coord
                x -= min_x
                y -= min_y
                for channel, area in ol:
                    if area == 0: continue
                    if channel is None: channel = Geom.NONE
                    chan_assoc[channel, color].add((x, y))
                    pix_ol[x, y].append((channel, color, area))
        return Geom(pix_ol, chan_assoc)

class Pixelator:
    ACTIVE_THRESHOLD = 0
    def __init__(self, geom, active_none = True, sparse_output = False):
        self.geom = geom
        self.active_none = active_none
        self.sparse = sparse_output

    def to_numba(self):
        geom = self.geom.to_numba()
        return partial(Pixelator.pixelate_numba, geom, (self.ACTIVE_THRESHOLD, self.active_none))

    def pixelate(self, channel_vals):
        active_by_color = [set() for _ in range(self.geom.num_colors)]
        for chan in range(self.geom.num_chans):
            if channel_vals[chan] <= self.ACTIVE_THRESHOLD: continue
            for color, pixels in enumerate(self.geom.chan_assoc[chan]):
                active_by_color[color].update(pixels)
        if self.active_none:
            for color in range(self.geom.num_colors):
                active_by_color[color].update(self.geom.chan_assoc[Geom.NONE][color])
        active_pixels = active_by_color[0]
        for i in range(1, len(active_by_color)):
            active_pixels = active_pixels.intersection(active_by_color[i])
        active_pixels = list(active_pixels)

        channel_overlaps = np.zeros(self.geom.num_chans)
        for pixel in active_pixels:
            for chan, color, area in self.geom.pix_ol[pixel]: 
                if chan == Geom.NONE: continue
                channel_overlaps[chan] += area
        with np.errstate(divide='ignore', invalid='ignore'):
            channel_density = np.divide(channel_vals, channel_overlaps)
        if self.sparse: 
            point_cloud = []
        else:
            img = np.zeros((self.geom.num_x, self.geom.num_y, self.geom.num_colors))
        for pixel in active_pixels:
            if self.sparse: 
                color_vals = np.zeros(self.geom.num_colors)
            for chan, color, area in self.geom.pix_ol[pixel]:
                if chan == Geom.NONE: continue
                if self.sparse: color_vals[color] += channel_density[chan]*area
                else: img[pixel+(color,)] += channel_density[chan]*area
            if self.sparse:
                point_cloud.append((pixel, color_vals))
        if self.sparse: 
            return point_cloud
        return img
    __call__ = pixelate

    @staticmethod
    @jit(nopython=True, parallel=True)
    def pixelate_numba(geom, pix, channel_vals_batch):
        pix_ol, chan_assoc, num_x, num_y, num_chans, num_colors, NONE = geom
        threshold, active_none = pix
        point_cloud_batch = List([List([[-1. for _ in range(2+num_colors)]]) for _ in range(len(channel_vals_batch))]) 
        for batch_ind in prange(len(channel_vals_batch)):
            channel_vals = channel_vals_batch[batch_ind]
            active_by_color = [{(NONE, NONE)}   for _ in range(num_colors)]
            for chan in range(num_chans):
                if channel_vals[chan] <= threshold: continue
                for color, pixels in enumerate(chan_assoc[chan]):
                    active_by_color[color].update(pixels)
            if active_none:
                for color in range(num_colors):
                    active_by_color[color].update(chan_assoc[NONE][color])
            active_pixels = active_by_color[0]
            for i in range(1, len(active_by_color)):
                active_pixels = active_pixels.intersection(active_by_color[i])
            active_pixels = list(active_pixels)

            channel_overlaps = np.zeros(num_chans)
            for pixel in active_pixels:
                if pixel[0] == NONE: continue
                for chan, color, area in pix_ol[pixel[0]][pixel[1]]: 
                    if chan == NONE: continue
                    channel_overlaps[int(chan)] += area
            channel_density = [v/ol if ol!=0 else np.inf for v, ol in zip(channel_vals, channel_overlaps)]
            point_cloud = List()
            for pixel in active_pixels:
                if pixel[0] == NONE: continue
                color_vals = [0.0 for _ in range(num_colors)] 
                for chan, color, area in pix_ol[pixel[0]][pixel[1]]: 
                    if chan == NONE: continue
                    color_vals[int(color)] += channel_density[int(chan)]*area
                point_cloud.append(list(map(float, pixel)) + color_vals)
            point_cloud_batch[batch_ind] = point_cloud
        return point_cloud_batch

    def sparse_to_dense(self, point_cloud, use_numba = False):
        img = np.zeros((self.geom.num_x, self.geom.num_y, self.geom.num_colors))
        if use_numba:
            for pt in point_cloud:
                pixel = (int(pt[0]), int(pt[1]))
                color_vals = pt[2:]
                for color, color_val in enumerate(color_vals):
                    img[pixel+(color,)] = color_val
        else:
            for pixel, color_vals in point_cloud:
                for color, color_val in enumerate(color_vals):
                    img[pixel+(color,)] = color_val
        return img

def make_pdsp_geom():
    HEIGHT = 5984
    WIDTH = 2300
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    PIXEL_SIZE = 2 
    tiling = (0, WIDTH//PIXEL_SIZE, -HEIGHT//PIXEL_SIZE, 0) 
    U_SEED_WIRE_1 = (35.7, UV_PITCH/PIXEL_SIZE, 1.2, 400, IDENTITY)
    V_SEED_WIRE_1 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, 1.2, 400, lambda w: 799-w), (tiling[1], 0), (0, 0))
    U_SEED_WIRE_2 = (35.7, UV_PITCH/PIXEL_SIZE, 1.2, 400, lambda w: w+1280)
    V_SEED_WIRE_2 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, 1.2, 400, lambda w: 2079-w), (tiling[1], 0), (0, 0))
    APA = {
        1: {
            "U": wrap_fixed_pitch_wires(U_SEED_WIRE_1, tiling),
            "V": wrap_fixed_pitch_wires(V_SEED_WIRE_1, tiling),
            "X": [(0, X_PITCH/PIXEL_SIZE, 1.2, 480, lambda w: w+800)]
            },
        2: {
            "U": wrap_fixed_pitch_wires(U_SEED_WIRE_2, tiling),
            "V": wrap_fixed_pitch_wires(V_SEED_WIRE_2, tiling),
            "X": [(0, X_PITCH/PIXEL_SIZE, 1.2, 480, lambda w: w+2080)]
            }
        }

    filt_pos_ang = lambda wire_sets: [wires for wires in wire_sets if wires[0] > 0]
    filt_neg_ang = lambda wire_sets: [wires for wires in wire_sets if wires[0] < 0]
    Face1 = {
            "X1": APA[1]["X"],
            "U1": filt_pos_ang(APA[1]["U"]),
            "V1": filt_neg_ang(APA[1]["V"]), 
            "U2": filt_neg_ang(APA[2]["U"]),
            "V2": filt_pos_ang(APA[2]["V"])
            }
    return Geom.create(np.concatenate(list(Face1.values())), tiling, 2560)
    

if __name__ == "__main__":
    import timeit, sys, time
    t0 = time.time()
    N = 200
    geo = Geom.create([(45, 2, 0.0001, N//2, IDENTITY), (90, 2, 0, N//2, lambda w: w+N//2)], (0, N, 0, N), N)
    #geo.save()
    #geo = Geom.load()

    #geo = make_pdsp_pixelator()
    #geo.save()
    #geo = Geom.load("geom_pdsp.npz")
    print("geom: ", sys.getsizeof(geo))
    print("pixel overlap: ", sys.getsizeof(geo.pix_ol))
    print("chan assoc: ", sys.getsizeof(geo.chan_assoc))
    t1 = time.time()
    print(t1-t0)
    N = geo.num_chans
    pix = Pixelator(geo, sparse_output = True)
    #print(timeit.timeit('pix(np.random.randint(0, 5, N))', globals=globals(), number=500)/500)
    pix = pix.to_numba()
    t2 = time.time()
    print("to numba: ", t2-t1)
    #pixelate_numba(*pix.to_numba(), np.random.randint(0, 5, N))

    print("numba avg: ")
    #print("random imag size: ", sys.getsizeof(pix(np.random.randint(0, 5, N))))
    print("dense input average time")
    #print(timeit.timeit('pix(np.random.randint(0, 5, (50, N)))', globals=globals(), number=1)/50)
    batch_size = 1000
    from copy import deepcopy
    t = 0
    for i in range(10):
        sparse = np.concatenate((np.ones((batch_size, 1*N//10)), np.zeros((batch_size, (9*N//10)))), axis=1)
        rng = np.random.default_rng()
        rng.shuffle(sparse, 1)
        sparse = deepcopy(sparse)
        t0 = time.time()
        pix(sparse)
        t += time.time()-t0
    print("sparse avg time", t/10/batch_size)
    #print(timeit.timeit('sparse = deepcopy(sparse); pix(sparse)', globals=globals(), number=10)/500)
    """
    print(timeit.timeit("sparse = [1]*(1*N//10) + [0]*(9*N//10); \
                np.random.shuffle(sparse); \
                pix(np.tile(sparse, (1000, 1)))", globals=globals(), number=5)/500)
                """

