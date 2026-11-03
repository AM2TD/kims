#!/usr/bin/env python3
"""Final 3D RMSE summary: KIMS / SFUISE / devkit EKF on MILUV and KIMS / SFUISE on NTU from the final_table_v2.py output,
UTIL (KIMS, SFUISE body output, devkit ESKF) and IULUI (KIMS, SFUISE; without SquareBob). Run from the package root.
Inputs and labels (environment, defaults = the paper's runs):
  TABLE_V2     final_table_v2.py output             (result/runlogs/final_table_v2.txt; run final_table_v2.py <label> first)
  UTIL_LABEL   KIMS run prefix on UTIL              (uf1_)
  IULUI_LABEL  KIMS run prefix on IULUI             (f32g_)
  OUT          summary file                         (result/runlogs/final_rmse_final.txt)
A method without output for a sequence is shown as '-' and left out of its median and counts."""
import sys, os, re, numpy as np; sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
from util_score import score as us
TABLE_V2 = os.environ.get("TABLE_V2", "result/runlogs/final_table_v2.txt"); UL = os.environ.get("UTIL_LABEL", "uf1_")
IL = os.environ.get("IULUI_LABEL", "f32g_"); OUT = os.environ.get("OUT", "result/runlogs/final_rmse_final.txt")
out = []; P = lambda s: out.append(s)
num = lambda x: {">1e4": 1e5, "-": np.nan}.get(x, None) if x in (">1e4", "-") else float(x)
fmt = lambda x: "-" if not np.isfinite(x) else (f"{x:.3f}" if x < 100 else "diverged")
V2 = [l.split() for l in open(TABLE_V2) if re.match(r"^(miluv|ntu)\s", l)]
# MILUV reference filter as published (Shalaby et al., IJRR 2026 / arXiv 2504.14376, ifo001): Table 10 = EKF with IMU, UWB and
# the laser height sensor. (Its Table 11, the same EKF with VINS-Fusion VIO added, is left out: camera methods are not compared.)
# The devkit's get_rmse is a per-axis RMS, sqrt(mean of the squared x/y/z errors), so 3D RMSE = sqrt(3) x the value.
# The filter starts from the ground-truth state; constellation 2 (random3_2, remoteControlHighPace_2) used the devkit anchor order.
T10 = {'default_1_remoteControlHighPace_1': 0.76667, 'default_1_random3_1': 0.14737, 'default_1_random3_0': 0.12541, 'default_1_circular3D_0': 0.17514,
       'default_1_remoteControlLowPace_0_v2': 0.12665, 'default_1_remoteControlLowPace_0': 0.14266, 'default_1_circular2D_0': 0.19352,
       'default_1_remoteControlHighPace_0': 0.78438, 'obstacles_1_random3_0b': 0.14545, 'default_1_random3_2': 0.14877,
       'default_1_remoteControlHighPace_2': 0.81323, 'noAprilTags_1_random3_2b': 0.13901, 'noAprilTags_1_remoteControlHighPace_2b': 0.15775}
def summary(name, rr, cols):
    a = np.array([r[1] for r in rr], float); ok = lambda j: np.isfinite(a[:, j])
    P(f"{name} median | " + " | ".join(f"{c} {np.median(a[ok(j), j]):.3f}" if ok(j).any() else f"{c} -" for j, c in enumerate(cols)))
    P(f"{name} >1 m (diverged >100 m) | " + " | ".join(f"{c} {int(np.sum(a[ok(j), j] > 1))} ({int(np.sum(a[ok(j), j] > 100))})" for j, c in enumerate(cols)))
    P(f"{name} KIMS lower than | " + " | ".join(f"{c} {int(np.sum(a[ok(0) & ok(j), 0] < a[ok(0) & ok(j), j]))}/{int(np.sum(ok(0) & ok(j)))}" for j, c in enumerate(cols) if j))
rr = [(r[1], [num(r[3]), num(r[5]), np.sqrt(3) * T10.get(r[1], np.nan)]) for r in V2 if r[0] == "miluv"]
for s, v in rr: P(f"miluv {s} | " + " | ".join(fmt(x) for x in v))
summary("miluv", rr, ["KIMS", "SFUISE", "EKF IMU+UWB+height (paper)"])
nh = np.array([num(r[6]) for r in V2 if r[0] == "miluv"])
P(f"miluv reference: devkit EKF IMU+UWB without height (our run) median {np.nanmedian(nh):.3f}, KIMS lower {int(np.sum(np.array([v[0] for _, v in rr]) < nh))}/{len(nh)}")
rr = [(r[1], [num(r[3]), num(r[5])]) for r in V2 if r[0] == "ntu"]
for s, v in rr: P(f"ntu {s} | " + " | ".join(fmt(x) for x in v))
summary("ntu", rr, ["KIMS", "SFUISE"])
from util_table4 import T4   # UTIL paper Table 4 ESKF, per flight
def u(s, f, lag):
    try: return us(s, f, lag)["rmse"] if os.path.exists(f) else np.nan
    except Exception: return np.nan
U, UREF = [], []
for s in (sorted(os.listdir("data/util")) if os.path.isdir("data/util") else []):   # UTIL is optional (scripts/prepare_data.sh)
    U.append((s, [u(s, f"result/util/diag/{UL}{s}_pose.txt", 99), u(s, f"result/util/diag/sfuisebody_{s}_pose.txt", 0), T4[s]]))
    UREF.append(u(s, f"result/util/diag/eskf_{s}_pose.txt", 0))
if U:
    for c in ["const1", "const2", "const3", "const4"]:
        for m in ["tdoa2", "tdoa3"]:
            g = np.array([v for s, v in U if s.startswith(c + "-") and m in s], float)
            P(f"util {c} {m} n={len(g)} | " + " | ".join(f"{np.median(g[np.isfinite(g[:, j]), j]):.3f}" if np.isfinite(g[:, j]).any() else "-" for j in range(3)))
    summary("util", U, ["KIMS", "SFUISE", "ESKF (UTIL paper Table 4)"])
    _k = np.array([v[0] for _, v in U]); _d = np.array(UREF)
    P(f"util reference: devkit ESKF, our run of the released code, median {np.median(_d):.3f}, KIMS lower {int(np.sum(_k < _d))}/{len(_k)} (used in the figures)")
else: P("util | no data/util (run scripts/prepare_data.sh util)")
I = []
for s in ["SquareTurn", "CircleSpin", "AC3_125000"]:
    v = [sc("iului", s, f"result/iului/diag/{IL}{s}_pose.txt", 199), sc("iului", s, f"result/iului/diag/sfuise_{s}_pose.txt", 0)]; I.append((s, v)); P(f"iului {s} | {fmt(v[0])} | {fmt(v[1])}")
summary("iului", I, ["KIMS", "SFUISE"])
open(OUT, "w").write("Final 3D RMSE [m] (KIMS FP32 + 8 threads, IMU + UWB; SFUISE published config; MILUV: the MILUV paper's EKF with height (Table 10), x sqrt(3); UTIL: the UTIL paper's ESKF (Table 4); MILUV raw mocap GT; IULUI without SquareBob)\n" + "\n".join(out) + "\n")
print("\n".join(out))
