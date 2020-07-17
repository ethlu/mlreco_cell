import numpy as np
import math

PRECISION = 3
def simple_overlap(wire_plane, tiling):
    angle, pitch, wire_0_offset, min_wire, max_wire = wire_plane
    min_x, max_x, min_y, max_y = tiling 
    wire_0_offset += 10**(-2*PRECISION) #avoid corner cases

    assert -90 < angle and angle < 90
    if angle > 0:
        assert pitch > np.cos((angle-45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe wire"
    else:
        assert pitch > np.cos((angle+45)*np.pi/180)*np.sqrt(2), "pitch should be wide enough to circumscribe wire"

    cos = np.cos(angle*np.pi/180)
    sin = np.sin(angle*np.pi/180)
    e1 = np.array([-sin, cos])
    e2 = np.array([cos, sin])

    region_offset = wire_0_offset - 0.5*pitch
    x_offset = region_offset/cos
    if angle != 0: #ignore vertical, possible coincidence
        y_offset = region_offset/sin

    crossings = np.full((max_x-min_x, max_y-min_y, 4), None) #up, down, l, r

    hop_x_wire = pitch/cos
    hop_x_grid = -sin/cos
    
    #fictitious boundary wire
    max_wire += 1

    for y in range(min_y, max_y+1):
        wire_0_crossing = x_offset + y*hop_x_grid
        for w in range(min_wire, max_wire+1):
            x_crossing = wire_0_crossing + w*hop_x_wire
            x = math.floor(x_crossing)
            if x < min_x or x >= max_x:
                continue
            x_rel = x_crossing - x
            assert x_rel != 0, "corner intersection, please perturb offset"
            if y > min_y:
                assert crossings[x,y-1,0] is None
                crossings[x, y-1, 0] = (x_rel, w)
            if y < max_y:
                assert crossings[x,y,1] is None
                crossings[x, y, 1] = (x_rel, w)

    if angle != 0:
        hop_y_wire = pitch/sin
        hop_y_grid = -cos/sin
        
        for x in range(min_x, max_x+1):
            wire_0_crossing = y_offset + x*hop_y_grid
            for w in range(min_wire, max_wire+1):
                y_crossing = wire_0_crossing + w*hop_y_wire
                y = math.floor(y_crossing)
                if y < min_y or y >= max_y:
                    continue
                y_rel = y_crossing - y
                if x > min_x:
                    assert crossings[x-1, y, 3] is None
                    crossings[x-1, y, 3] = (y_rel, w)
                if x < max_x:
                    assert crossings[x,y,2] is None
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
                w = int(round((e2.dot((x+1/2, y+1/2)) - wire_0_offset)/pitch))
                if w >= max_wire or w < min_wire:
                    w = None
                overlaps[(x, y)] = [(w, 1)]
                continue

            w1 = crossing[0][1]
            assert crossing[1][1] == w1 
            w0 = w1 - 1
            if w1 == max_wire:
                w1 = None
            if w1 == min_wire:
                w0 = None 

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
                

if __name__ == "__main__":
    print(simple_overlap((30, 2, 0,  -100, 100), (-100, 100, -100, 100)))
    print(simple_overlap((45, math.sqrt(2), 0., -100, 100), (0, 2, 0, 2)))
