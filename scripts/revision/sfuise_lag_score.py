#!/usr/bin/env python3
"""UTIL: position RMSE vs output lag, SFUISE (sfuise_lag_run.sh, tag converted to the body origin as in util_sfuise_body.py)
and KIMS (lag 0 = state at the newest window epoch, lag 2 s = the table output with flush). No alignment, vs Vicon.
usage: sfuise_lag_score.py <seq> [<seq> ...]   -> table on stdout"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from util_score import score
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D = os.path.join(PKG, "result/util/diag")
OFF = np.array([-0.012, 0.001, 0.091]); LAGS = ["0", "0.5", "1", "2", "3.3", "5", "lastwin"]
def R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def body(f):
    out = f.replace("sfuiselag_", "sfuiselagbody_")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(f):
        P = np.loadtxt(f, comments="#"); B = P.copy()
        for i in range(len(P)): B[i, 1:4] = P[i, 1:4] - R(P[i, 4:8]) @ OFF
        np.savetxt(out, B, fmt="%.6f", header="time x y z qx qy qz qw  (SFUISE at a fixed output lag, body origin)")
    return out
rows = []
for s in sys.argv[1:]:
    r = {"seq": s}
    for L in LAGS:
        f = os.path.join(D, f"sfuiselag_{s}_{'lastwin' if L == 'lastwin' else 'lag' + L}_pose.txt")
        r["S" + L] = score(s, body(f), 0) if os.path.exists(f) else None
    r["K0"] = score(s, os.path.join(D, f"u1_{s}_now_pose.txt"), 0); r["K2"] = score(s, os.path.join(D, f"u1_{s}_pose.txt"), 99)
    r["Sold"] = score(s, os.path.join(D, f"sfuisebody_{s}_pose.txt"), 0); rows.append(r)
def line(key, name):
    v = [r[key] for r in rows if r.get(key)]; 
    if not v: return
    a3 = np.array([x["rmse"] for x in v]); axy = np.array([x["xy"] for x in v])
    print(f"{name:28s} | " + " ".join(f"{x:6.3f}" for x in a3) + f" | median 3D {np.median(a3):.3f} xy {np.median(axy):.3f}")
print(f"{'3D RMSE [m]':28s} | " + " ".join(f"{r['seq'][:6] + r['seq'][13:18]:>6s}"[:6] for r in rows))
for L in LAGS: line("S" + L, f"SFUISE lag {L if L != 'lastwin' else '~10 s (last window)'}" + (" s" if L != "lastwin" else ""))
line("Sold", "SFUISE table run (~10 s)"); line("K0", "KIMS lag 0 (newest epoch)"); line("K2", "KIMS lag 2 s (table)")
