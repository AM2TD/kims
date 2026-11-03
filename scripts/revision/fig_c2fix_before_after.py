#!/usr/bin/env python3
"""KIMS X/Y/Z before and after the MILUV constellation-2 anchor fix (anchors 2 and 3 swapped back, devkit issue #56) on
default_1_random3_2 and default_1_remoteControlHighPace_2: raw mocap GT, KIMS with the devkit anchor order
(result/miluv/diag/c2_swapped/f32a_*), KIMS with the corrected file (result/miluv/diag/f32a_*); lag 228 + window flush.
-> result/figs/c2fix_kims_before_after_xyz.png"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])            # rows, std_score, sc, ef
def track(f, lag=228):
    P = rows(f); th = P[:-lag, 0]; X = P[lag:, 1:4]; F = rows(f.replace("_pose.txt", "_flush_pose.txt")); n = len(F) // 2
    first, last = F[:n], F[n:]; mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
    return np.concatenate([first[mf, 0], th, last[ml, 0]]), np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
S = ["default_1_random3_2", "default_1_remoteControlHighPace_2"]
fig, ax = plt.subplots(6, 1, figsize=(12, 17))
for i, s in enumerate(S):
    gt_t, gt = ef.load_gt("miluv", s); t0 = gt_t[0]
    fb, fa = f"result/miluv/diag/c2_swapped/f32a_{s}_pose.txt", f"result/miluv/diag/f32a_{s}_pose.txt"
    (tb, Xb), (ta, Xa) = track(fb), track(fa); rb, ra = sc("miluv", s, fb, 228), sc("miluv", s, fa, 228)
    for j, lab in enumerate("xyz"):
        a = ax[3 * i + j]
        a.plot(gt_t - t0, gt[:, j], "k", lw=2.0, label="GT (raw mocap)")
        a.plot(tb - t0, Xb[:, j], color="#9a9a9a", lw=1.0, label=f"KIMS, devkit anchor order ({rb:.3f} m)")
        a.plot(ta - t0, Xa[:, j], color="#1f4e9c", lw=1.0, label=f"KIMS, anchors 2/3 swapped back ({ra:.3f} m)")
        a.set_ylabel(f"{lab} [m]"); a.grid(alpha=0.3); a.set_xlim(0, gt_t[-1] - t0)
        if j == 0: a.set_title(f"MILUV {s}: constellation-2 anchor fix, 3D RMSE in legend")
        if j == 2: a.legend(loc="upper right", fontsize=8)
ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fn = "result/figs/c2fix_kims_before_after_xyz.png"; fig.savefig(fn, dpi=95); print("wrote", fn)
