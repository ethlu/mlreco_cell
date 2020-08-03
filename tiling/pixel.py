import numpy as np
from tiling.square import * 
from numba import jit, prange
from numba.typed import List, Dict
from numba.core import types
from numba.types.containers import UniTuple
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
        with np.load(file_name, allow_pickle = True) as geom:
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
    def __init__(self, geom, active_none = True, sparse_output = True):
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
                point_cloud.append(np.concatenate((pixel, color_vals)))
        if self.sparse: 
            return np.array(point_cloud)
        return img
    __call__ = pixelate

    def sparse_to_dense(self, point_cloud):
        img = np.zeros((self.geom.num_x, self.geom.num_y, self.geom.num_colors))
        for pt in point_cloud:
            pixel = (int(pt[0]), int(pt[1]))
            color_vals = pt[2:]
            for color, color_val in enumerate(color_vals):
                img[pixel+(color,)] = color_val
        return img

    @staticmethod
    @jit(nopython=True, parallel=True)
    def pixelate_numba(geom, pix, channel_vals_batch):
        pix_ol, chan_assoc, num_x, num_y, num_chans, num_colors, NONE = geom
        threshold, active_none = pix
        num_events = len(channel_vals_batch)
        num_slices = len(channel_vals_batch[0])
        batch_output = List([List([List([[-1. 
            for _ in range(2+num_colors)]]) 
            for _ in range(num_slices)])
            for _ in range(num_events)])
        for event_i in prange(num_events):
            for slice_i in prange(num_slices):
                channel_vals = channel_vals_batch[event_i][slice_i]
                active_by_color = [{(NONE, NONE)} for _ in range(num_colors)]
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
                    point_cloud.append(list(map(float, pixel)) + color_vals) #float array...
                batch_output[event_i][slice_i] = point_cloud
        return batch_output

    @staticmethod
    @jit(nopython=True, parallel=True, num_colors=3)
    def numba_to_numpy(pix_batch):
        num_events = len(pix_batch)
        num_slices = len(pix_batch[0])
        n_points = 0
        slic_starts = []
        event_starts = []
        for event in pix_batch:
            slic_starts_event = []
            for slic in event:
                slic_starts_event.append(n_points)
                n_points += len(slic)
            slic_starts.append(slic_starts_event)
            event_starts.append(slic_starts_event[0])
        output = np.empty((n_points, 4+num_colors))
        for event_i in prange(num_events):
            for slic_i in prange(num_slices):
                slic_start = slic_starts[event_i][slic_i]
                for i, point in enumerate(pix_batch[event_i][slic_i]):
                    output_i = slic_start+i
                    output[output_i, 0] = slic_i
                    output[output_i, 1] = point[0]
                    output[output_i, 2] = point[1]
                    output[output_i, 3] = event_i
                    for c in range(num_colors):
                        output[output_i, 4+c] = point[2+c]
        return output, np.array(event_starts)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def downsamples(pix_batch, downsample=(1,1,1), num_colors=3, kType=UniTuple(types.int64, 2), vType=types.float64[:]):
        num_events = len(pix_batch)
        num_slices = len(pix_batch[0])
        assert num_slices % downsample[0] == 0, "drift downsample incompatible"
        num_slices_down = num_slices//downsample[0]
        batch_output = List([List([List([[-1. 
            for _ in range(2+num_colors)]]) 
            for _ in range(num_slices_down)])
            for _ in range(num_events)])
        for event_i in prange(num_events):
            for slic_i_start in prange(num_slices_down):
                slic_down_pixels = Dict.empty(
                    key_type = kType,
                    value_type = vType,
                )
                for slic_i in range(slic_i_start*downsample[0], (slic_i_start+1)*downsample[0]):
                    for pt in pix_batch[event_i][slic_i]:
                        coord = (pt[0]//downsample[1], pt[1]//downsample[2])
                        vals = np.array(pt[2:])
                        try:
                            slic_down_pixels[coord] += vals
                        except:
                            slic_down_pixels[coord] = vals
                point_cloud = List()
                for k, v in slic_down_pixels.items():
                    pt = list(map(float, k))
                    for c in v:
                        pt.append(c)
                    point_cloud.append(pt)
                batch_output[event_i][slic_i_start] = point_cloud
        return batch_output

if __name__ == "__main__":
    import timeit, time
    import sys
    sys.path.append('..')
    from geom.pdsp import *
    t0 = time.time()
    N = 100
    geo = Geom.create([(45, 2, 0.0001, N//2, IDENTITY), (90, 2, 0, N//2, lambda w: w+N//2)], (0, N, 0, N), N)
    #geo.save()
    #geo = Geom.load()
    #geo = make_APA_geom(1) 
    #geo.save("geom_pdsp_face1")

    #geo = get_APA_geom(2) 
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
    #print("to numba: ", t2-t1)
    #pixelate_numba(*pix.to_numba(), np.random.randint(0, 5, N))

    #print("numba avg: ")
    #print("random imag size: ", sys.getsizeof(pix(np.random.randint(0, 5, N))))
    #print("dense input average time")
    #print(timeit.timeit('pix(np.random.randint(0, 5, (50, N)))', globals=globals(), number=1)/50)
    batch_size = 10
    from copy import deepcopy
    t = 0
    for i in range(0):
        sparse = np.concatenate((np.ones((batch_size, 1*N//10)), np.zeros((batch_size, (9*N//10)))), axis=1)
        rng = np.random.default_rng()
        rng.shuffle(sparse, 1)
        sparse = np.tile(sparse,(10, 1, 1))
        print(sparse.shape)
        t0 = time.time()
        img = pix(sparse)
        img2 = Pixelator.numba_to_numpy(img)
        print(img2[:3])
        print(img2[-3:])
        img = Pixelator.numba_to_numpy(Pixelator.downsamples(img, (2, 2, 2)))
        print("DOWNSAMPLE")
        print(img[:3])
        print(img[-3:])

        t += time.time()-t0
    np.savez_compressed("test.npz", img)
    print("sparse avg time", t/10/batch_size/10)
    #print(timeit.timeit('sparse = deepcopy(sparse); pix(sparse)', globals=globals(), number=10)/500)
    """
    print(timeit.timeit("sparse = [1]*(1*N//10) + [0]*(9*N//10); \
                np.random.shuffle(sparse); \
                pix(np.tile(sparse, (1000, 1)))", globals=globals(), number=5)/500)
                """

