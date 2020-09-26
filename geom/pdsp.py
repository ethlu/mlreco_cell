import sys, os
dirname = os.path.dirname(__file__)
import numpy as np

TPC_LENGTH = 3585
DRIFT_SPEED = 1.565 #mm/us, from detsim.fcl
T0 = 252 #us, a guess
SAMPLING_RATE = 2 #MHz
DRIFT_PER_TICK = DRIFT_SPEED/SAMPLING_RATE
APA_HEIGHT = 5984
APA_WIDTH = 2300
APA_Y = 76
APA_Z0 = 3
APA_Z_INC = APA_WIDTH + 20
APA_N_CHAN = 2560

PIXEL_SIZE = 2  
VOXEL_SIZE = (DRIFT_PER_TICK, PIXEL_SIZE, PIXEL_SIZE)

FACE1 = os.path.join(dirname, "geom_pdsp_face1.npz")
FACE2 = os.path.join(dirname, "geom_pdsp_face2-v2.npz")

def get_TPC_box(tpc_num = 1): #offline TPC number
    assert tpc_num in np.arange(12)
    apa = tpc_num//2
    face = tpc_num%2 + 1 
    r_min = np.array((TPC_LENGTH if apa%2 else -TPC_LENGTH, APA_Y, APA_Z0 + (apa//2)*APA_Z_INC))
    if face == 1:
        r_min[0]-=TPC_LENGTH
    r_max = r_min + np.array((TPC_LENGTH, APA_HEIGHT, APA_WIDTH))
    return r_min, r_max #r_min < box < r_max

def get_TPC_coord_transform(tpc_num = 1): 
    r_min, r_max = get_TPC_box(tpc_num) 
    face = tpc_num%2 + 1 #face 1 flips, 2 doesn't 
    def transform(coord, time):
        if any(coord > r_max) or any(coord < r_min):
            return None
        if face == 1:
            coord[0] = r_max[0] - coord[0]
            coord[2] = r_max[2] - coord[2]
        else:
            coord[0] -= r_min[0]
            coord[2] -= r_min[2]
        coord[1] -= r_min[1]
        coord[0] += DRIFT_SPEED * (time+T0)
        return coord
    return transform

def get_APA_chans(tpc_num = 1): #offline TPC number
    assert tpc_num in np.arange(12)
    apa = tpc_num//2
    channels = range(apa*APA_N_CHAN, (apa+1)*APA_N_CHAN)
    return channels

def get_APA_geom(tpc_num = 1):
    from tiling.pixel import Geom
    face = tpc_num%2 + 1 
    if face == 1:
        return Geom.load(FACE1)
    return Geom.load(FACE2)

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
        return 1599-w
    V_WIRE = shift_wires_origin(
            (35.7, UV_PITCH/PIXEL_SIZE, 49.8/PIXEL_SIZE, 1148, v_wmap),
            (0, 0), R0)

    X_WIRE = shift_wires_origin(
            (0, X_PITCH/PIXEL_SIZE, 5.6/PIXEL_SIZE, 480, lambda w: w+2080),
            (0, 0), R0)

    return Geom.create([U_WIRE, V_WIRE, X_WIRE], tiling, 2560)


    
def make_APA_geom_larsoft(face):
    """Too complicated to get it exactly right, use 'simple' model for now..."""
    APA_HEIGHT = 5991
    APA_WIDTH = 2303
    UV_PITCH = 4.669 
    X_PITCH = 4.79 
    Z_OFFSET = 4.5
    PIXEL_SIZE=0.1
    tiling = (0, APA_WIDTH//PIXEL_SIZE, -APA_HEIGHT//PIXEL_SIZE, 0) 
    z_wire_offset = Z_OFFSET/PIXEL_SIZE
    uv_wire_offset = z_wire_offset*np.cos(35.7)
    #active_tiling = (0, APA_WIDTH//PIXEL_SIZE-2, -APA_HEIGHT//PIXEL_SIZE, 0) 
    active_tiling = (0,1,0,1)

    U_SEED_WIRE_1 = (35.7, UV_PITCH/PIXEL_SIZE, uv_wire_offset, 400, lambda w: 399-w)
    U_SEED_WIRE_2 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, uv_wire_offset, 400, lambda w: 799-w), (tiling[1], 0), (0, 0))
    V_SEED_WIRE_1 = shift_wires_origin((-35.7+180, UV_PITCH/PIXEL_SIZE, uv_wire_offset, 400, lambda w: 1199-w), (tiling[1], 0), (0, 0))
    V_SEED_WIRE_2 = (35.7, UV_PITCH/PIXEL_SIZE, uv_wire_offset, 400, lambda w: 1599-w)
    APA = {
        1: {
            "U": wrap_fixed_pitch_wires(U_SEED_WIRE_1, tiling),
            "V": wrap_fixed_pitch_wires(V_SEED_WIRE_1, tiling),
            "X": [(0, X_PITCH/PIXEL_SIZE, z_wire_offset, 480, lambda w: w+1600)]
            },
        2: {
            "U": wrap_fixed_pitch_wires(U_SEED_WIRE_2, tiling),
            "V": wrap_fixed_pitch_wires(V_SEED_WIRE_2, tiling),
            "X": [(0, X_PITCH/PIXEL_SIZE, z_wire_offset, 480, lambda w: w+2080)]
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
        pass
        return Geom.create(np.concatenate(list(Face1.values())), active_tiling, 2560)
    return Geom.create(np.concatenate(list(Face2.values())), active_tiling, 2560)

