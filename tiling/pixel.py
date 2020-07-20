import numpy as np
from square import * 

class Pixelator:
    ACTIVE_THRESHOLD = 0
    def __init__(self, wire_sets, tiling, active_none = True):
        self.active_none = active_none
        wires_by_plane = dict() 
        for wires in wire_sets:
            angle = wires[0]
            try:
                wires_by_plane[angle].append(wires)
            except KeyError:
                wires_by_plane[angle] = [wires]
        self.overlaps = [wireplane_overlap(merged_fp_wireplane(wires_by_plane[i]), tiling) \
                for i in sorted(wires_by_plane.keys())]
        min_x, max_x, min_y, max_y = tiling 
        self.image_size = max_x - min_x, max_y - min_y, len(self.overlaps)
        self.image_x_shift, self.image_y_shift = -min_x, -min_y

    def pixelate(self, channel_vals):
        active_pixels = []
        for pixel in self.overlaps[0].keys():
            active_pixel = True
            for color_overlaps in self.overlaps:
                active_color = False
                for associated_channel, area in color_overlaps[pixel]:
                    if area == 0: continue
                    if associated_channel is None:
                        if self.active_none:
                            active_color = True
                            break
                        else: continue
                    if channel_vals[associated_channel] > self.ACTIVE_THRESHOLD:
                        active_color = True
                        break
                if not active_color:
                    active_pixel = False
                    break
            if active_pixel:
                active_pixels.append(pixel)

        channel_overlaps = [0] * len(channel_vals)
        for pixel in active_pixels:
            for color_overlaps in self.overlaps:
                for associated_channel, area in color_overlaps[pixel]:
                    if associated_channel is None or area == 0: continue
                    channel_overlaps[associated_channel] += area

        image = np.zeros(self.image_size)
        for pixel in active_pixels:
            for i, color_overlaps in enumerate(self.overlaps):
                for channel, area in color_overlaps[pixel]:
                    if channel is None or area == 0: continue
                    image[pixel[0]+self.image_x_shift, pixel[1]+self.image_y_shift, i] += \
                        channel_vals[channel]*area/channel_overlaps[channel]
        return image
    __call__ = pixelate

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
    return Pixelator(np.concatenate(list(Face1.values())), tiling)

if __name__ == "__main__":
    pix = Pixelator([(0, 1, 0, 3, IDENTITY), (90, 1, 0, 3, lambda w: w+4)], (0, 3, 0, 3))
    #print(pix.pixelate([1, 2,0,0,1,0,0,0]))
    pix = make_pdsp_pixelator()


