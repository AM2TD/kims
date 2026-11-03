#!/usr/bin/env python3
"""Altitude on the coplanar-anchor flights of the revised evaluation (IULUI AC1 SquareTurn and CircleSpin, anchors at
1.30 m; MILUV constellation 1 random3_1 and remoteControlHighPace_1, anchors at 1.42 m): ground truth, its mirror image
about the fitted anchor plane, KIMS (final configuration, fixed-lag output with the window flush), SFUISE (published
configuration) and, on MILUV, the devkit EKF with the height sensor (the MILUV paper's Table 10 filter).
usage: fig_zcross_final.py [out_prefix]   (default result/figs/fig_zcross_final -> .png and .pdf)"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import err_series as es
from eval_flip import anchor_plane
PKG = es.PKG; os.chdir(PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/fig_zcross_final"
CASES = [("iului", "SquareTurn", "f32g", 199, "IULUI SquareTurn"), ("iului", "CircleSpin", "f32g", 199, "IULUI CircleSpin"),
         ("miluv", "default_1_random3_1", "f32a", 228, "MILUV random3_1"), ("miluv", "default_1_remoteControlHighPace_1", "f32a", 228, "MILUV remoteControlHighPace_1")]
COL = {"KIMS": "#b03a2e", "SFUISE": "#e08a1e", "devkit EKF + height": "#2a7f62"}
def track(f, lag):
    P = es.rows(f); th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]
    ff = f.replace("_pose.txt", "_flush_pose.txt")
    if lag and os.path.exists(ff):
        F = es.rows(ff); n = len(F) // 2; first, last = F[:n], F[n:]; mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
    return th, X
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, ax = plt.subplots(len(CASES), 1, figsize=(10, 2.3 * len(CASES)))
for a, (ds, s, lab, lag, title) in zip(ax, CASES):
    gt_t, gt = es.ef.load_gt(ds, s); c, n = anchor_plane(ds, s); t0 = gt_t[0]
    mir = gt - 2 * ((gt - c) @ n)[:, None] * n[None, :]
    a.axhline(c[2], color="#41B3BA", ls="--", lw=1.0)
    a.plot(gt_t - t0, mir[:, 2], color="#888", ls=":", lw=1.0, label="mirrored GT")
    a.plot(gt_t - t0, gt[:, 2], color="k", ls="--", lw=1.4, label="ground truth")
    curves = [("KIMS", *track(f"result/{ds}/diag/{lab}_{s}_pose.txt", lag), es.sc(ds, s, f"result/{ds}/diag/{lab}_{s}_pose.txt", lag))]
    f = f"result/{ds}/diag/sfuise_{s}_pose.txt"
    if os.path.exists(f): curves.append(("SFUISE", *track(f, 0), es.sc(ds, s, f, 0)))
    if ds == "miluv":
        r = es.err_devkit_miluv(s)
        if r is not None:
            D = pd.read_csv(os.path.join(os.environ.get("MILUV_DEVKIT", os.path.expanduser("~/miluv_devkit")), f"results/ekf_{s}_h.csv")).values
            curves.append(("devkit EKF + height", D[:, 0], D[:, 1:4], np.sqrt(np.mean(np.sum((D[:, 1:4] - D[:, 4:7]) ** 2, 1)))))
    lo, hi = min(gt[:, 2].min(), mir[:, 2].min()) - 0.4, max(gt[:, 2].max(), mir[:, 2].max()) + 0.4
    for m, th, X, rmse in reversed(curves):
        k = (th >= gt_t[0]) & (th <= gt_t[-1]); th, z = th[k], X[k, 2]
        lab_m = f"{m} ({rmse:.3f} m)" if rmse < 100 else f"{m} (diverged)"
        bad = np.where((z < lo - 1) | (z > hi + 1))[0]
        if len(bad): lab_m += f", leaves the plot at {th[bad[0]] - t0:.0f} s"
        a.plot(th - t0, z, color=COL[m], lw=1.6 if m == "KIMS" else 1.1, label=lab_m, zorder=3 if m == "KIMS" else 2)
    a.set_ylim(lo, hi); a.set_xlim(0, gt_t[-1] - t0); a.set_ylabel("z (m)"); a.grid(ls="--", lw=0.5, alpha=0.5)
    a.text(0.995, (c[2] - lo) / (hi - lo) + 0.02, f"anchor plane {c[2]:.2f} m", transform=a.transAxes, ha="right", va="bottom", fontsize=8, color="#2a8a90")
    a.set_title(title, loc="left", fontsize=10); a.legend(loc="upper left", fontsize=7.5, ncol=3, frameon=False)
ax[-1].set_xlabel("time since ground-truth start (s)")
fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); print("wrote", out + ".png/.pdf")
