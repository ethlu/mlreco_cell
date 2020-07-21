import numpy as np
from square import * 

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

    def load(file_name = "geom.npz"):
        print("loading: ", file_name)
        geom = np.load(file_name, allow_pickle = True)
        return Geom(geom["pix_ol"], geom["chan_assoc"])

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

    def sparse_to_dense(self, point_cloud):
        img = np.zeros((self.geom.num_x, self.geom.num_y, self.geom.num_colors))
        for pixel, color_vals in point_cloud:
            for color, color_val in enumerate(color_vals):
                img[pixel+(color,)] = color_val
        return img

def make_pdsp_pixelator():
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
    import timeit
    N = 400
    #geo = Geom.create([(45, 2, 0.0001, N//2, IDENTITY), (90, 2, 0, N//2, lambda w: w+N//2)], (0, N, 0, N), N)
    #geo.save()
    #geo = Geom.load()

    #geo = make_pdsp_pixelator()
    #geo.save()
    geo = Geom.load("geom_pdsp.npz")
    N = geo.num_chans
    pix = Pixelator(geo, sparse_output = True)
    print("dense input average time")
    print(timeit.timeit('pix(np.random.randint(0, 5, N))', globals=globals(), number=5)/5)
    print("sparse avg time")
    print(timeit.timeit("sparse = [1]*(N//10) + [0]*(9*N//10); \
                np.random.shuffle(sparse); \
                pix(sparse)", globals=globals(), number=5)/5)

