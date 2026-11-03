#!/usr/bin/env python3
"""SFUISE's EstimationInterface::getPoseInUWB returns the TAG position, q_UW (R(t) offset + p(t)) + t_UW, while the UTIL
Vicon ground truth (and KIMS, the devkit ESKF) refer to the body origin. Convert every sfuise_<seq>_pose.txt to the body
origin, p_body = p_tag - R(q) offset with SFUISE's own UTIL offset (config_test_util.yaml, the devkit lever arm), and write
sfuisebody_<seq>_pose.txt next to it.   usage: util_sfuise_body.py"""
import os, glob, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D = os.path.join(PKG, "result/util/diag")
OFF = np.array([-0.012, 0.001, 0.091])          # config_test_util.yaml 'offset'
def R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
n = 0
for f in sorted(glob.glob(os.path.join(D, "sfuise_*_pose.txt"))):
    P = np.loadtxt(f, comments="#")
    B = P.copy()
    for i in range(len(P)): B[i, 1:4] = P[i, 1:4] - R(P[i, 4:8]) @ OFF
    out = f.replace("sfuise_", "sfuisebody_")
    np.savetxt(out, B, fmt="%.6f", header="time x y z qx qy qz qw  (SFUISE converted from the tag to the body origin, util_sfuise_body.py)")
    n += 1
print(n, "files converted")
