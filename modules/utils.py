import numpy as np
import math

def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(max(min(cos,1),-1)))

def calc_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def px_to_in(px, cfg):
    return px / cfg['pixels_per_inch']
