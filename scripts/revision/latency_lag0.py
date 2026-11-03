#!/usr/bin/env python3
"""(A) Accuracy of KIMS's lag-0 output (state at the newest window epoch, *_now_pose.txt) next to the lag-T table output,
final configuration (FP32 rollout + 8 host threads: MILUV/NTU f32a_*, UTIL uf1_*). MILUV/UTIL: no alignment; NTU: 4-DoF, prism,
gt+10..end-10 s. 'common span' scores the lag-T output only where a lag-0 output exists (after the first full window).
(B) Real-time latency from a single-server FIFO replay: epoch k arrives at its measurement time a_k (newest epoch time),
is processed for the measured standalone time s_k (timing file, first 20 solves dropped) after the previous one finishes;
latency = finish - arrival. Age = time since the measurement behind the newest available lag-0 estimate, sampled every 1 ms.
-> result/runlogs/lag0_latency.txt"""
import sys, os, csv, numpy as np, pandas as pd
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])          # std_score, ntu_score, rows, ef
from util_score import score as uscore, track as utrack
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
MILS = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
def miluv_common(s, f, lag, t0):   # lag-T output (with flush) restricted to t >= t0 (where the lag-0 output exists)
    gt_t, gt = ef.load_gt("miluv", s); R = rows(f); th = R[:-lag, 0]; X = R[lag:, 1:4]
    ff = f.replace("_pose.txt", "_flush_pose.txt")
    if os.path.exists(ff):
        F = rows(ff); n = len(F) // 2; last = F[n:]; ml = last[:, 0] > th[-1]; th = np.concatenate([th, last[ml, 0]]); X = np.concatenate([X, last[ml, 1:4]])
    k = (th >= max(t0, gt_t[0])) & (th <= gt_t[-1]); e = X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)
    return np.sqrt(np.mean(np.sum(e ** 2, 1)))
P("(A) lag-0 vs lag-T 3D RMSE [m], final configuration")
P(f"{'sequence':40s} | lag-T table | lag-T common span | lag-0 | lag-0 - lag-T (common)")
A = {"miluv": [], "ntu": [], "util": []}
for s in MILS:
    f, fn = f"result/miluv/diag/f32a_{s}_pose.txt", f"result/miluv/diag/f32a_{s}_now_pose.txt"
    t0 = rows(fn)[0, 0]; a, b, c = std_score("miluv", s, f, 228), miluv_common(s, f, 228, t0), std_score("miluv", s, fn, 0)
    A["miluv"].append((a, b, c)); P(f"miluv {s:34s} | {a:.3f} | {b:.3f} | {c:.3f} | {1e3*(c-b):+.0f} mm")
for s in ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]:
    a = ntu_score(s, f"result/ntu/diag/f32a_{s}_pose.txt", 117); c = ntu_score(s, f"result/ntu/diag/f32a_{s}_now_pose.txt", 0)
    A["ntu"].append((a, a, c)); P(f"ntu   {s:34s} | {a:.3f} | {a:.3f} | {c:.3f} | {1e3*(c-a):+.0f} mm")
for s in sorted(os.listdir("data/util")):
    a = uscore(s, f"result/util/diag/uf1_{s}_pose.txt", 99)["rmse"]; c = uscore(s, f"result/util/diag/uf1_{s}_now_pose.txt", 0)["rmse"]; A["util"].append((a, a, c))
for ds, v in A.items():
    v = np.array(v); P(f"median {ds:6s} (n {len(v):2d}) | lag-T {np.median(v[:,0]):.3f} | common {np.median(v[:,1]):.3f} | lag-0 {np.median(v[:,2]):.3f} | "
                       f"lag-0 worse in {int(np.sum(v[:,2] > v[:,1]))}/{len(v)}, median change {1e3*np.median(v[:,2]-v[:,1]):+.0f} mm, max {1e3*np.max(v[:,2]-v[:,1]):+.0f} mm")
P("")
P("(B) FIFO replay of the measured standalone solve times. latency = estimate available - measurement time; age = staleness of the")
P("newest lag-0 estimate at any instant (1 ms grid). 'zero-time' = the same with instant processing (age from the sampling alone).")
P(f"{'run':30s} | interval | latency mean  p50   p95   p99   max | late>interval | max backlog | age mean  p95   max | zero-time age mean p95")
CASES = [("MILUV HighPace_1", "miluv", "hp1"), ("MILUV random3_0", "miluv", "r30"), ("MILUV circular2D_0", "miluv", "c2d"), ("NTU eee_01", "ntu", "eee01"),
         ("NTU sbs_01", "ntu", "sbs01"), ("UTIL tdoa2", "util", "u2"), ("UTIL tdoa3", "util", "u3")]
def replay(a, s):
    fin = np.empty_like(a); f = a[0]
    for k in range(len(a)): f = max(a[k], f) + s[k]; fin[k] = f
    lat = fin - a; backlog = np.searchsorted(a, fin, side="left") - np.arange(len(a)) - 1
    grid = np.arange(fin[0], a[-1], 0.001); j = np.searchsorted(fin, grid, side="right") - 1; age = grid - a[j]
    jz = np.searchsorted(a, grid, side="right") - 1; agez = grid - a[jz]
    return lat, backlog, age, agez
for name, ds, sh in CASES:
    for pre, lab in [("tt_k", "FP64 1 thr"), ("f32", "FP32 8 thr")]:
        t = rows(f"result/{ds}/diag/{pre}_{sh}_now_pose.txt")[:, 0]; ms = np.loadtxt(f"result/{ds}/diag/{pre}_{sh}_timing.txt")
        a, s = t[20:], ms[20:] / 1e3; lat, bl, age, agez = replay(a, s); iv = np.median(np.diff(a)) * 1e3; mi = np.mean(np.diff(a)) * 1e3
        q = lambda x, p: 1e3 * np.percentile(x, p)
        P(f"{name + ' ' + lab:30s} | {mi:5.1f} ms | {1e3*lat.mean():6.1f} {q(lat,50):5.1f} {q(lat,95):5.1f} {q(lat,99):5.1f} {1e3*lat.max():6.1f} | "
          f"{100*np.mean(lat > np.r_[np.diff(a), np.inf]):5.1f} % | {int(bl.max()):4d} | {1e3*age.mean():6.1f} {q(age,95):5.1f} {1e3*age.max():6.1f} | {1e3*agez.mean():5.1f} {q(agez,95):5.1f}")
open("result/runlogs/lag0_latency.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
