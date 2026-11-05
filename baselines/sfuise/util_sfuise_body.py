#!/usr/bin/env python3
import os, glob, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D = os.path.join(PKG, "result/util")
OFF = np.array([-0.012, 0.001, 0.091])
def R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
n = 0
for f in sorted(glob.glob(os.path.join(D, "sfuisetag_*_pose.txt"))):
    P = np.loadtxt(f, comments="#")
    B = P.copy()
    for i in range(len(P)): B[i, 1:4] = P[i, 1:4] - R(P[i, 4:8]) @ OFF
    out = f.replace("sfuisetag_", "sfuise_")
    np.savetxt(out, B, fmt="%.6f", header="time x y z qx qy qz qw  (SFUISE converted from the tag to the body origin, util_sfuise_body.py)")
    n += 1
print(n, "files converted")
