import numpy as np
import math
from bisect import bisect

PRECISION = 3
IDENTITY = lambda w: w
def wireplane_overlap(wire_plane, tiling):
    """calculates the overlap (area) b/t every unit square and the wire regions of wire_plane
    assumes each square overlaps with at most 2 regions
    offset of a (boundary) line is perp distance to origin = e2 dot {pt on line} """
    angle, bound_offsets, w_map = wire_plane 
    min_x, max_x, min_y, max_y = tiling 
    num_wires = len(bound_offsets) - 1
    print("computing overlaps for wire plane of angle: {}, num wires: {}".format(angle, num_wires))

    cos = round(np.cos(np.radians(angle)), 2*PRECISION)
    sin = round(np.sin(np.radians(angle)), 2*PRECISION)
    e2 = np.array([cos, sin])
    if cos != 0: 
        x_offsets = bound_offsets/cos
    if sin != 0: 
        y_offsets = bound_offsets/sin

    crossings = np.full((max_x-min_x, max_y-min_y, 4), None) #up, down, l, r
    if cos != 0:
        hop_x_grid = -sin/cos
        for y in range(min_y, max_y+1):
            for w, x_offset in enumerate(x_offsets):
                x_crossing = x_offset + y*hop_x_grid 
                x = math.floor(x_crossing)
                if x < min_x or x >= max_x:
                    continue
                x_rel = x_crossing - x
                if x_rel == 0 and sin == 0:
                    continue
                assert x_rel != 0, "ill-posed corner intersection, try perturbing offset"
                if y > min_y:
                    assert crossings[x,y-1,0] is None, "more than 2 associated wires? try increase pitch" 
                    crossings[x, y-1, 0] = (x_rel, w) 
                if y < max_y:
                    assert crossings[x,y,1] is None, "more than 2 associated wires? try increase pitch"
                    crossings[x, y, 1] = (x_rel, w)
    if sin != 0:
        hop_y_grid = -cos/sin
        for x in range(min_x, max_x+1):
            for w, y_offset in enumerate(y_offsets):
                y_crossing = y_offset + x*hop_y_grid 
                y = math.floor(y_crossing)
                if y < min_y or y >= max_y:
                    continue
                y_rel = y_crossing - y
                if y_rel == 0 and cos == 0:
                    continue
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
                else: w = w_map(w)
                overlaps[(x, y)] = [(w, 1)]
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
    """convert fixed-pitch wires to wireplane (i.e. region boundary) representation"""
    angle, pitch, wire_0_offset, num_wires, w_map = wires 
    if angle > 0:
        assert pitch > np.cos((angle-45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe pixel"
    else:
        assert pitch > np.cos((angle+45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe pixel"
    wire_0_offset += 10**(-2*PRECISION) #avoid corner cases
    bound_offsets = np.arange(num_wires+1)*pitch + wire_0_offset - 0.5*pitch
    return angle, bound_offsets, w_map

def merged_fp_wireplane(wire_sets):
    """ merge the fixed pitch wires (i.e. segments) of wire_sets, gaps are "filled" """
    wire_sets_angle = wire_sets[0][0]
    wire_sets = sorted(wire_sets, key = lambda off: off[2])
    bound_offsets, w_map_map = [], []
    prev_max_offset = None 
    for i, wires in enumerate(wire_sets):
        angle, pitch, wire_0_offset, num_wires, w_map = wires 
        assert angle == wire_sets_angle, "wireplane must have single angle"
        curr_bound_offsets = np.arange(num_wires+1)*pitch + wire_0_offset - 0.5*pitch
        if prev_max_offset is not None:
            assert wire_0_offset > prev_max_offset, "merge conflict"
            gap_pitch = wire_0_offset - prev_max_offset
            bound_offsets.pop()
            curr_bound_offsets[0] += (pitch - gap_pitch)/2
        bound_offsets.extend(curr_bound_offsets)
        w_map_map.extend([(i, n) for n in range(num_wires)])
        prev_max_offset = wire_0_offset + (num_wires - 1)*pitch
    bound_offsets = np.array(bound_offsets)
    w_map = lambda w: wire_sets[w_map_map[w][0]][4](w_map_map[w][1])
    return wire_sets_angle, bound_offsets, w_map

def wrap_fixed_pitch_wires(seed_wires, tiling):
    """returns set of wrapped wires by reflecting seed_wires left and right, until out of tiling box."""
    angle, pitch, wire_0_offset, num_wires, w_map = seed_wires 
    min_x, max_x, min_y, max_y = tiling 
    cos = round(np.cos(np.radians(angle)), 2*PRECISION)
    sin = round(np.sin(np.radians(angle)), 2*PRECISION)
    e2 = np.array([cos, sin])

    wrapped_wires = [seed_wires]
    if cos == 0 or sin == 0:
        return wrapped_wires
    assert -90<angle and angle<90

    def crawl_up(angle, e2, min_offset, num_wires, w_map):
        new_num_wires = num_wires 
        max_offset = min_offset + (num_wires-1)*pitch
        if angle > 0:
            l_cross_upper = e2.dot((min_x, max_y))
            if min_offset > l_cross_upper: return
            if max_offset > l_cross_upper:
                new_num_wires = math.floor((l_cross_upper - min_offset)/pitch) + 1
                max_offset = min_offset + (new_num_wires-1)*pitch
            min_offset = 2*cos*min_x - max_offset
            new_w_map = lambda w: w_map(new_num_wires - w - 1)
        else:
            r_cross_lower = e2.dot((max_x, max_y))
            if max_offset < r_cross_lower: return
            if min_offset < r_cross_lower:
                new_num_wires = math.floor((max_offset - r_cross_lower)/pitch) + 1
            min_offset = 2*cos*max_x - max_offset
            new_w_map = lambda w: w_map(num_wires - w - 1)
        e2 = e2 * [1, -1]
        angle *= -1
        min_offset = round(min_offset, PRECISION*2)
        wrapped_wires.append((angle, pitch, min_offset, new_num_wires, new_w_map))
        crawl_up(angle, e2, min_offset, new_num_wires, new_w_map)

    def crawl_down(angle, e2, min_offset, num_wires, w_map):
        new_num_wires = num_wires 
        max_offset = min_offset + (num_wires-1)*pitch
        if angle > 0:
            r_cross_lower = e2.dot((max_x, min_y))
            if max_offset < r_cross_lower: return
            if min_offset < r_cross_lower:
                new_num_wires = math.floor((max_offset - r_cross_lower)/pitch) + 1
            min_offset = 2*cos*max_x - max_offset
            new_w_map = lambda w: w_map(num_wires - w - 1)
        else:
            l_cross_upper = e2.dot((min_x, min_y))
            if min_offset > l_cross_upper: return
            if max_offset > l_cross_upper:
                new_num_wires = math.floor((l_cross_upper - min_offset)/pitch) + 1
                max_offset = min_offset + (new_num_wires-1)*pitch
            min_offset = 2*cos*min_x - max_offset
            new_w_map = lambda w: w_map(new_num_wires - w - 1)
        e2 = e2 * [1, -1]
        angle *= -1
        min_offset = round(min_offset, PRECISION*2)
        wrapped_wires.append((angle, pitch, min_offset, new_num_wires, new_w_map))
        crawl_down(angle, e2, min_offset, new_num_wires, new_w_map)

    crawl_up(angle, e2, wire_0_offset, num_wires, w_map)
    crawl_down(angle, e2, wire_0_offset, num_wires, w_map)
    return wrapped_wires

def shift_wires_origin(wires, old_origin, new_origin):
    angle, pitch, wire_0_offset, num_wires, w_map = wires 
    cos = round(np.cos(np.radians(angle)), 2*PRECISION)
    sin = round(np.sin(np.radians(angle)), 2*PRECISION)
    e2 = np.array([cos, sin])
    wire_n_offset = wire_0_offset + (num_wires-1)*pitch
    
    origin_offset = np.subtract(new_origin, old_origin).dot(e2)
    wire_0_offset -= origin_offset
    wire_n_offset -= origin_offset 
    new_w_map = w_map
    if abs(angle) > 90:
        angle -= 180
        wire_0_offset *= -1
        wire_n_offset *= -1
        new_w_map = lambda w: w_map(num_wires-1-w)
    angle = round(angle, PRECISION*2)
    return angle, pitch, round(min(wire_0_offset, wire_n_offset), PRECISION*2), num_wires, new_w_map


if __name__ == "__main__":
    #print(simple_overlap((30, 2, 0,  -100, 100), (-100, 100, -100, 100)))
    #print(simple_overlap((45, 1.42, 0., -100, 100), (0, 2, 0, 2)))
    #print(wrap_fixed_pitch_wires((45, math.sqrt(2)+0.0001, 0, 5, IDENTITY), (-5, 5, -10, 10)))
    DUNE_TILE = (0, 1150, -2992, 0) 
    #print(wrap_fixed_pitch_wires((35.7, 2.3345, 0.1, 380, IDENTITY), DUNE_TILE))
    #print(len(wrap_fixed_pitch_wires((35.7, 2.3345, 0.1, 380, IDENTITY), DUNE_TILE)))
    wrapped = wrap_fixed_pitch_wires((35.7, 2.3345, 0.1, 380, IDENTITY), DUNE_TILE)
    pos_wrapped = [wires for wires in wrapped if wires[0]>0]
    print(pos_wrapped)
    print(merged_fp_wireplane(pos_wrapped))
