import sys, os
dirname = os.path.dirname(__file__)
import numpy as np

TPC_LENGTH = 3587 #a guess, should be distance to wire simulation
MAX_TICK = 6000
DRIFT_SPEED = 1.565 #mm/us, from detsim.fcl
SIM_T0 = 250 #us, wcls-sim-drift-simchannel.jsonnet
SAMPLING_RATE = 2 #MHz
DRIFT_PER_TICK = DRIFT_SPEED/SAMPLING_RATE
APA_HEIGHT = 5984
APA_WIDTH = 2300
APA_Y = 76
APA_Z0 = 3
APA_Z_INC = 2320 #fit from simchan
APA_N_CHAN = 2560

PIXEL_SIZE = 2  
VOXEL_SIZE = (DRIFT_PER_TICK, PIXEL_SIZE, PIXEL_SIZE)
DOWNSAMPLE = (8, 4, 4) #used for ML

FACE1 = os.path.join(dirname, "geom_pdsp_face1-v3.npz")
FACE2 = os.path.join(dirname, "geom_pdsp_face2-v3.npz")

def get_TPC_box(tpc_num = 1): #offline TPC number
    assert tpc_num in np.arange(12)
    apa = tpc_num//2
    face = tpc_num%2 + 1 
    r_min = np.array((TPC_LENGTH if apa%2 else -TPC_LENGTH, APA_Y, APA_Z0 + (apa//2)*APA_Z_INC))
    if face == 1:
        r_min[0]-=TPC_LENGTH
    r_max = r_min + np.array((TPC_LENGTH, APA_HEIGHT, APA_WIDTH))
    return r_min, r_max #r_min < box < r_max

def get_TPC_coord_transform(tpc_num = 1, coord_lims = True): 
    r_min, r_max = get_TPC_box(tpc_num) 
    face = tpc_num%2 + 1 #face 1 flips, 2 doesn't 
    max_x = DRIFT_PER_TICK * MAX_TICK 
    def transform(coord, time):
        if coord_lims and (any(coord > r_max) or any(coord < r_min)): 
            return None
        if face == 1:
            coord[0] = r_max[0] - coord[0]
        else:
            coord[0] -= r_min[0]
        coord[2] -= r_min[2]
        coord[1] -= r_min[1]
        coord[0] += DRIFT_SPEED * (time + SIM_T0)
        if coord_lims and (coord[0] < 0 or coord[0] > max_x):
            return None
        return coord
    return transform

def get_TPC_inverse_coords_transform(tpc_num, voxel_size, r0, t0=-SIM_T0): 
    r_min, r_max = get_TPC_box(tpc_num) 
    face = tpc_num%2 + 1 #face 1 flips, 2 doesn't 
    if face == 1:
        r_min[0] = r_max[0]
    r_offset = r_min - r0
    r_offset[0] += DRIFT_SPEED*(t0+SIM_T0) * (1 if face == 1 else -1)
    r_offset_vox = np.round(r_offset/voxel_size)
    r_vox_diff = r_offset_vox * voxel_size - r_offset
    def transform(coords):
        if len(coords) == 0: return coords
        coords = np.asarray(coords)
        if face == 1:
            coords[:, 0] = r_offset_vox[0] - coords[:, 0]
        else:
            coords[:, 0] += r_offset_vox[0]
        coords[:, 2] += r_offset_vox[2]
        coords[:, 1] += r_offset_vox[1]
        return coords
    return transform, r_vox_diff

def get_TPC_chans(tpc_num = 1): #offline TPC number
    assert tpc_num in np.arange(12)
    apa = tpc_num//2
    channels = range(apa*APA_N_CHAN, (apa+1)*APA_N_CHAN)
    return channels

def get_APA_chan_transform(tpc_num = 1):
    APA_VWIRE_XOFFSET = 4.5
    APA_XWIRE_XOFFSET = 9
    face = tpc_num%2 + 1 
    v_tick_offset = round(APA_VWIRE_XOFFSET/DRIFT_PER_TICK)
    v_fills = np.zeros(v_tick_offset)
    x_tick_offset = round(APA_XWIRE_XOFFSET/DRIFT_PER_TICK)
    x_fills = np.zeros(x_tick_offset)
    def transform(apa_chan, vals):
        if face == 1:
            return vals
        else:
            return vals
            """ignored; simulation doesn't offset """
            assert 0 <= apa_chan and apa_chan < APA_N_CHAN
            if 800 <= apa_chan and apa_chan < 1600:
                return np.concatenate((vals[v_tick_offset:], v_fills))
            if 2080 <= apa_chan and apa_chan < 2560:
                return np.concatenate((vals[x_tick_offset:], x_fills))
            return vals
    return transform

def get_APA_wireplane_maps(tpc_num = 1):
    face = tpc_num%2 + 1 
    if face == 1:
        pass
    else:
        def u_wmap(w):
            if w < 400:
                return 400+w
            return w-400
        def v_wmap(w):
            if w < 748:
                return 1547-w
            return 1599-(w-748)
        x_wmap = lambda w: w+2080
        return list(map(u_wmap, np.arange(1148))), \
            list(map(v_wmap, np.arange(1148))),\
            list(map(x_wmap, np.arange(480))),
    
def get_APA_wireplane_projectors(tpc_num = 1):
    from tiling.square import shift_wires_origin, IDENTITY, PRECISION
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    R0 = (APA_Z0/PIXEL_SIZE, APA_Y/PIXEL_SIZE)
    def make_projector(wires):
        angle, pitch, wire_0_offset, num_wires, w_map = wires 
        cos = round(np.cos(np.radians(angle)), 2*PRECISION)
        sin = round(np.sin(np.radians(angle)), 2*PRECISION)
        e2 = np.array([cos, sin])
        return lambda coord: int((e2.dot(coord)-wire_0_offset)//pitch)
    face = tpc_num%2 + 1 
    if face == 1:
        pass
    else:
        U_WIRE = shift_wires_origin(
                (-35.7, UV_PITCH/PIXEL_SIZE, -3535.01/PIXEL_SIZE, 1148, IDENTITY),
                (0, 0), R0)
        V_WIRE = shift_wires_origin(
                (35.7, UV_PITCH/PIXEL_SIZE, 49.8/PIXEL_SIZE, 1148, IDENTITY),
                (0, 0), R0)
        X_WIRE = shift_wires_origin(
                (0, X_PITCH/PIXEL_SIZE, 5.6/PIXEL_SIZE, 480, IDENTITY),
                (0, 0), R0)
        return make_projector(U_WIRE), make_projector(V_WIRE), make_projector(X_WIRE)

def get_APA_geom(tpc_num = 1):
    global FACE1, FACE2
    from tiling.pixel import Geom
    face = tpc_num%2 + 1 
    if face == 1:
        if type(FACE1)==str:
            FACE1 = Geom.load(FACE1) 
        return FACE1
    if type(FACE2)==str:
        FACE2 = Geom.load(FACE2) 
    return FACE2

def make_APA_geom_larsoft_face1():
    from tiling.pixel import Geom
    from tiling.square import shift_wires_origin, wrap_fixed_pitch_wires
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    R0 = (APA_Z0/PIXEL_SIZE, APA_Y/PIXEL_SIZE)
    tiling = (0, APA_WIDTH//PIXEL_SIZE, 0, APA_HEIGHT//PIXEL_SIZE) 
        
    def u_wmap(w):
        if w < 348:
            return 347-w
        return 799-(w-348)
    U_WIRE = shift_wires_origin(
            (35.7, UV_PITCH/PIXEL_SIZE, 52.1/PIXEL_SIZE, 1148, u_wmap),
            (0, 0), R0)

    def v_wmap(w):
        if w < 800:
            return w+800
        return w
    V_WIRE = shift_wires_origin(
            (-35.7, UV_PITCH/PIXEL_SIZE, -3530.801/PIXEL_SIZE, 1148, v_wmap),
            (0, 0), R0)

    X_WIRE = shift_wires_origin(
            (0, X_PITCH/PIXEL_SIZE, 5.75/PIXEL_SIZE, 480, lambda w: w+1600),
            (0, 0), R0)

    return Geom.create([U_WIRE, V_WIRE, X_WIRE], tiling, 2560)

def make_APA_geom_larsoft_face2():
    from tiling.pixel import Geom
    from tiling.square import shift_wires_origin, wrap_fixed_pitch_wires
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    R0 = (APA_Z0/PIXEL_SIZE, APA_Y/PIXEL_SIZE)
    tiling = (0, APA_WIDTH//PIXEL_SIZE, 0, APA_HEIGHT//PIXEL_SIZE) 
        
    def u_wmap(w):
        if w < 400:
            return 400+w
        return w-400
    U_WIRE = shift_wires_origin(
            (-35.7, UV_PITCH/PIXEL_SIZE, -3535.01/PIXEL_SIZE, 1148, u_wmap),
            (0, 0), R0)

    def v_wmap(w):
        if w < 748:
            return 1547-w
        return 1599-(w-748)
    V_WIRE = shift_wires_origin(
            (35.7, UV_PITCH/PIXEL_SIZE, 49.8/PIXEL_SIZE, 1148, v_wmap),
            (0, 0), R0)

    X_WIRE = shift_wires_origin(
            (0, X_PITCH/PIXEL_SIZE, 5.6/PIXEL_SIZE, 480, lambda w: w+2080),
            (0, 0), R0)

    return Geom.create([U_WIRE, V_WIRE, X_WIRE], tiling, 2560)


if __name__ == "__main__":
    geo = make_APA_geom_larsoft_face1()
    geo.save(FACE1)
    geo = make_APA_geom_larsoft_face2()
    geo.save(FACE2)


""" 
This is 'v1' geom based on wire wrapping, unfortunately not as accurate as the larsoft ones above

def make_APA_geom(face):
    from tiling.pixel import Geom
    from tiling.square import shift_wires_origin, wrap_fixed_pitch_wires
    assert face in (1,2)
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    UV_OFFSET = 0.5
    tiling = (0, APA_WIDTH//PIXEL_SIZE, -APA_HEIGHT//PIXEL_SIZE, 0) 
    U_SEED_WIRE_1 = (35.7, UV_PITCH/PIXEL_SIZE, UV_OFFSET, 400, lambda w: 399-w)
    U_SEED_WIRE_2 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, UV_OFFSET, 400, lambda w: 799-w), (tiling[1], 0), (0, 0))
    V_SEED_WIRE_1 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, UV_OFFSET, 400, lambda w: 1199-w), (tiling[1], 0), (0, 0))
    V_SEED_WIRE_2 = (35.7, UV_PITCH/PIXEL_SIZE, UV_OFFSET, 400, lambda w: 1599-w)
    APA = {
        1: {
            "U": wrap_fixed_pitch_wires(U_SEED_WIRE_1, tiling),
            "V": wrap_fixed_pitch_wires(V_SEED_WIRE_1, tiling),
            "X": [(0, X_PITCH/PIXEL_SIZE, 1.2, 480, lambda w: w+1600)]
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
            "U2": filt_pos_ang(APA[2]["U"]),
            "V2": filt_neg_ang(APA[2]["V"])
            }
    Face2 = {
            "X2": APA[2]["X"],
            "U1": filt_neg_ang(APA[1]["U"]),
            "V1": filt_pos_ang(APA[1]["V"]), 
            "U2": filt_neg_ang(APA[2]["U"]),
            "V2": filt_pos_ang(APA[2]["V"])
            }
    if face == 1:
        return Geom.create(np.concatenate(list(Face1.values())), tiling, 2560)
    return Geom.create(np.concatenate(list(Face2.values())), tiling, 2560)
"""

