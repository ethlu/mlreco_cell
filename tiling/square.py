import numpy as np
import math
from bisect import bisect

PRECISION = 3
IDENTITY = lambda w: w
def wireplane_overlap(wire_plane, tiling):
    """offset of a line is perp distance to origin = e2 dot {pt on line} """

    angle, bound_offsets, w_map = wire_plane 
    min_x, max_x, min_y, max_y = tiling 
    num_wires = len(bound_offsets) - 1

    assert -90 < angle and angle < 90
    cos = np.cos(angle*np.pi/180)
    sin = np.sin(angle*np.pi/180)
    e2 = np.array([cos, sin])
    x_offsets = bound_offsets/cos
    if angle != 0: 
        y_offsets = bound_offsets/sin

    crossings = np.full((max_x-min_x, max_y-min_y, 4), None) #up, down, l, r
    hop_x_grid = -sin/cos
    for y in range(min_y, max_y+1):
        for w, x_offset in enumerate(x_offsets):
            x_crossing = x_offset + y*hop_x_grid 
            x = math.floor(x_crossing)
            if x < min_x or x >= max_x:
                continue
            x_rel = x_crossing - x
            assert x_rel != 0, "ill-posed corner intersection, try perturbing offset"
            if y > min_y:
                assert crossings[x,y-1,0] is None, "more than 2 associated wires? try increase pitch" 
                crossings[x, y-1, 0] = (x_rel, w) 
            if y < max_y:
                assert crossings[x,y,1] is None, "more than 2 associated wires? try increase pitch"
                crossings[x, y, 1] = (x_rel, w)
    if angle != 0:
        hop_y_grid = -cos/sin
        for x in range(min_x, max_x+1):
            for w, y_offset in enumerate(y_offsets):
                y_crossing = y_offset + x*hop_y_grid 
                y = math.floor(y_crossing)
                if y < min_y or y >= max_y:
                    continue
                y_rel = y_crossing - y
                assert y_rel != 0, "ill-posed corner intersection, try perturbing offset"
                if x > min_x:
                    assert crossings[x-1, y, 3] is None, "more than 2 associated wires? try increase pitch"
                    crossings[x-1, y, 3] = (y_rel, w)
                if x < max_x:
                    assert crossings[x,y,2] is None, "more than 2 associated wires? try increase pitch"
                    crossings[x, y, 2] = (y_rel, w)

    overlaps = dict()
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            crossing = crossings[x, y, :]
            cross_i = np.sort(np.nonzero(crossing))[0]
            crossing = crossing[cross_i]

            if len(cross_i) in (0, 1):
                if len(cross_i) == 1:
                    assert crossing[0][0] < 10**-PRECISION or 1-crossing[0][0] < 10**-PRECISION
                w = bisect(bound_offsets, e2.dot((x+1/2, y+1/2))) - 1
                if w >= num_wires or w < 0:
                    w = None
                overlaps[(x, y)] = [(w_map(w), 1)]
                continue

            assert len(cross_i) == 2, "more than 2 associated wires? try increase pitch"
            w1 = crossing[0][1]
            assert crossing[1][1] == w1 
            w0 = w1 - 1
            if w1 >= num_wires:
                w1 = None
            if w0 < 0:
                w0 = None 
            w0 = w_map(w0) if w0 is not None else None
            w1 = w_map(w1) if w1 is not None else None

            if cross_i[0] == 0 and cross_i[1] == 1:
                area = round((crossing[0][0] + crossing[1][0])/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[0] > 0:
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
                else:
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
            elif cross_i[0] == 2 and cross_i[1] == 3:
                area = round((crossing[0][0] + crossing[1][0])/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[1] > 0:
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
                else:
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
            elif cross_i[0] == 0 and cross_i[1] == 3:
                area = round((1-crossing[0][0]) * (1-crossing[1][0])/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[0] > 0:
                    assert e2[1] > 0
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
                else:
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
            elif cross_i[0] == 1 and cross_i[1] == 2:
                area = round(crossing[0][0] * crossing[1][0]/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[0] > 0:
                    assert e2[1] > 0
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
                else:
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
            elif cross_i[0] == 0 and cross_i[1] == 2:
                area = round(crossing[0][0] * (1- crossing[1][0])/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[0] > 0:
                    assert e2[1] < 0
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
                else:
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
            elif cross_i[0] == 1 and cross_i[1] == 3:
                area = round((1-crossing[0][0]) * crossing[1][0]/2, PRECISION)
                area_comp = round(1-area, PRECISION)
                if e2[0] > 0:
                    assert e2[1] < 0
                    overlaps[(x, y)] = [(w0, area_comp), (w1, area)]
                else:
                    overlaps[(x, y)] = [(w0, area), (w1, area_comp)]
    return overlaps

def fixed_pitch_wireplane(wires):
    angle, pitch, wire_0_offset, num_wires, w_map = wires 
    if angle > 0:
        assert pitch > np.cos((angle-45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe pixel"
    else:
        assert pitch > np.cos((angle+45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe pixel"
    wire_0_offset += 10**(-2*PRECISION) #avoid corner cases
    bound_offsets = np.arange(num_wires+1)*pitch + wire_0_offset - 0.5*pitch
    return angle, bound_offsets, w_map

def wrap_fixed_pitch_wires(seed_wires, tiling):
    angle, pitch, wire_0_offset, num_wires, w_map = seed_wires 
    min_x, max_x, min_y, max_y = tiling 
    assert -90 < angle and angle < 90
    cos = np.cos(angle*np.pi/180)
    sin = np.sin(angle*np.pi/180)

    wrapped_wires = [seed_wires]
    if angle == 0:
        return wrapped_planes

    def w_map_reversal(w_map_i):
        if w_map_i == 1:
            return w_map
        else:
            return lambda w: w_map(num_wires-1-w)

    def crawl_up(e2_2, angle, w_map_i, min_offset):
        max_offset = min_offset + (num_wires-1)*pitch
        if angle > 0:
            left_crossing_min_y = (min_offset - cos*min_x)/e2_2
            if left_crossing_min_y > max_y:
                return
            min_offset = 2*cos*min_x - max_offset
        else:
            right_crossing_min_y = (max_offset - cos*max_x)/e2_2
            if right_crossing_min_y > max_y:
                return
            min_offset = 2*cos*max_x - max_offset
        e2_2 *= -1
        angle *= -1
        w_map_i *= -1
        wrapped_wires.append((angle, pitch, min_offset, num_wires, w_map_reversal(w_map_i)))
        crawl_up(e2_2, angle, w_map_i, min_offset)

    def crawl_down(e2_2, angle, w_map_i, min_offset):
        max_offset = min_offset + (num_wires-1)*pitch
        if angle > 0:
            right_crossing_max_y = (max_offset - cos*max_x)/e2_2
            if right_crossing_max_y < min_y:
                return
            min_offset = 2*cos*max_x - max_offset
        else:
            left_crossing_max_y = (min_offset - cos*min_x)/e2_2
            if left_crossing_max_y < min_y:
                return
            min_offset = 2*cos*min_x - max_offset
        e2_2 *= -1
        angle *= -1
        w_map_i *= -1
        wrapped_wires.append((angle, pitch, min_offset, num_wires, w_map_reversal(w_map_i)))
        crawl_down(e2_2, angle, w_map_i, min_offset)

    crawl_up(sin, angle, 1, wire_0_offset)
    crawl_down(sin, angle, 1, wire_0_offset)
    return wrapped_wires

"""
def merge_wires(ol1, ol2):
    def find_none_area(overlap):
        none_a = [overlap[i][1] for i in range(len(overlap)) if overlap[i][0] is None]
        return none_a[0] if none_a else 0

    merged = dict()
    for k, v1 in ol1.items():
        v1_none_a = find_none_area(v1)
        v2_none_a = find_none_area(v2)
        assert v1_none_a + v2_none_a >= 1 - 1**(-PRECISION), "merge conflict"
        if not v1_none_a:
            merged[k] = v1
        elif not v2_none_a:
            merged[k] = v2
"""
if __name__ == "__main__":
    #print(simple_overlap((30, 2, 0,  -100, 100), (-100, 100, -100, 100)))
    #print(simple_overlap((45, 1.42, 0., -100, 100), (0, 2, 0, 2)))
    #print(wrap_fixed_pitch_wires((45, math.sqrt(2)+0.0001, 0, 5, IDENTITY), (-5, 5, -10, 10)))
    DUNE_TILE = (0, 1150, -2992, 0) 
    print(wrap_fixed_pitch_wires((35.7, 2.3345, 0.1, 380, IDENTITY), DUNE_TILE))
    print(len(wrap_fixed_pitch_wires((35.7, 2.3345, 0.1, 380, IDENTITY), DUNE_TILE)))
