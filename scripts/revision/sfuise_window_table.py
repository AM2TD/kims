#!/usr/bin/env python3
"""SFUISE with its published 10 s window vs the same configuration with a window matched to KIMS (window_size 30 knots = 3 s
on MILUV, 20 = 2 s elsewhere; nothing else changed), against KIMS, table scoring (MILUV/IULUI whole span, NTU VIRAL Leica
prism 4-DoF gt+10 s .. end-10 s, UTIL whole span with SFUISE moved from the tag to the body origin). SFUISE's committed
estimate is the last one before a stamp leaves its window, so its output delay equals its window (about 10 s or 2-3 s).
usage: python3 scripts/revision/sfuise_window_table.py > result/runlogs/sfuise_window.txt   (package root)"""
import os, sys, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
from util_score import score as us
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("MIL = ["); exec(src[i:src.index("\n", i) + 1])
def u(s, f, lag):
    try: return us(s, f, lag)["rmse"] if os.path.exists(f) else np.nan
    except Exception: return np.nan
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]; IU = ["SquareTurn", "CircleSpin", "AC3_125000"]
UT8 = ["const1-trial1-tdoa2", "const1-trial1-tdoa3", "const2-trial1-tdoa2", "const2-trial1-tdoa3", "const3-trial1-tdoa2", "const3-trial1-tdoa3", "const4-trial1-tdoa2-traj1", "const4-trial1-tdoa3-traj1"]
D = {"miluv": [(s, sc("miluv", s, f"result/miluv/diag/f32a_{s}_pose.txt", 228), sc("miluv", s, f"result/miluv/diag/sfuise_{s}_pose.txt", 0), sc("miluv", s, f"result/miluv/diag/sfuise_w30_{s}_pose.txt", 0), sc("miluv", s, f"result/miluv/diag/sfuise_w30_g15_{s}_pose.txt", 0)) for s in MIL],
     "ntu": [(s, sc("ntu", s, f"result/ntu/diag/f32a_{s}_pose.txt", 117), sc("ntu", s, f"result/ntu/diag/sfuise_{s}_pose.txt", 0), sc("ntu", s, f"result/ntu/diag/sfuise_w20_{s}_pose.txt", 0), sc("ntu", s, f"result/ntu/diag/sfuise_w20_g15_{s}_pose.txt", 0)) for s in NTU],
     "iului": [(s, sc("iului", s, f"result/iului/diag/f32g_{s}_pose.txt", 199), sc("iului", s, f"result/iului/diag/sfuise_{s}_pose.txt", 0), sc("iului", s, f"result/iului/diag/sfuise_w20_{s}_pose.txt", 0)) for s in IU],
     "util": [(s, u(s, f"result/util/diag/uf1_{s}_pose.txt", 99), u(s, f"result/util/diag/sfuisebody_{s}_pose.txt", 0), u(s, f"result/util/diag/sfuisebody_w20_{s}_pose.txt", 0)) for s in UT8]}
WIN = {"miluv": "3 s", "ntu": "2 s", "iului": "2 s", "util": "2 s"}
f = lambda x: "-" if not np.isfinite(x) else ("diverged" if x > 100 else f"{x:.3f}")
print("3D position RMSE [m]. KIMS window/delay: MILUV 3 s, others 2 s. SFUISE published window 10 s vs the matched window; MILUV/NTU also")
print("with the matched window and the wide 1.5 m gate (published gate 0.3 m) on a few flights, to separate the window from the gate.\n")
for ds in ["miluv", "ntu", "iului", "util"]:
    g = ds in ("miluv", "ntu")
    print(f"{ds.upper():6s} {'sequence':40s} | {'KIMS':>8s} {'SFUISE 10 s':>12s} {'SFUISE ' + WIN[ds]:>12s}" + (f" {WIN[ds] + ' gate 1.5':>14s}" if g else ""))
    for r in D[ds]: print(f"{'':6s} {r[0]:40s} | {f(r[1]):>8s} {f(r[2]):>12s} {f(r[3]):>12s}" + (f" {f(r[4]):>14s}" if g else ""))
    A = np.array([r[1:4] for r in D[ds]], float)
    med = lambda j: np.median(A[np.isfinite(A[:, j]), j]) if np.isfinite(A[:, j]).any() else np.nan
    div = lambda j: int(np.sum(A[:, j] > 100)); low = lambda j: int(np.sum(np.isfinite(A[:, j]) & (A[:, 0] < A[:, j])))
    nok = lambda j: int(np.sum(np.isfinite(A[:, j])))
    print(f"{'':6s} {'median (diverged counted)':40s} | {med(0):8.3f} {med(1):12.3f} {med(2):12.3f}")
    print(f"{'':6s} {'diverged (>100 m) / KIMS lower':40s} | {div(0):8d} {f'{div(1)} / {low(1)}/{nok(1)}':>12s} {f'{div(2)} / {low(2)}/{nok(2)}':>12s}\n")
