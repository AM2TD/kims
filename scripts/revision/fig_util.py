#!/usr/bin/env python3
"""UTIL figures. xyz: X/Y/Z of Vicon, KIMS, SFUISE and the devkit ESKF for one or more flights.
strip: per-flight 3D RMSE of the three methods grouped by constellation and mode (cf. SFUISE paper Fig. 4).
usage: fig_util.py xyz <out.png> <seq> [<seq> ...] | fig_util.py strip <out.png>"""
import sys, os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from util_score import track, score
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D = os.path.join(PKG, "result/util/diag")
METH = [("KIMS", "u1_", 99, "#1f4e9c"), ("SFUISE", "sfuisebody_", 0, "#e07b00"), ("ESKF", "eskf_", 0, "#2a9d4b")]   # SFUISE at the body origin
if sys.argv[1] == "xyz":
    seqs = sys.argv[3:]; fig, ax = plt.subplots(3 * len(seqs), 1, figsize=(13, 2.4 * 3 * len(seqs)))
    for si, s in enumerate(seqs):
        G = pd.read_csv(os.path.join(PKG, "data/util", s, "gt.csv")).values
        for j, nm in enumerate("xyz"):
            a = ax[3 * si + j]; a.plot(G[:, 0], G[:, 1 + j], color="black", lw=2.2, label="Vicon")
            for name, pre, lag, col in METH:
                f = os.path.join(D, f"{pre}{s}_pose.txt")
                if not os.path.exists(f): continue
                th, X = track(f, lag); r = score(s, f, lag)
                a.plot(th, X[:, j], color=col, lw=1.1, label=f"{name} ({r['rmse']:.3f} m)")
            lo, hi = G[:, 1 + j].min(), G[:, 1 + j].max(); a.set_ylim(lo - 1.0, hi + 1.0)   # a diverged estimate leaves the axis
            a.set_ylabel(f"{nm} [m]"); a.grid(alpha=0.3)
            if j == 0: a.set_title(f"UTIL {s}: position, anchor frame (3D RMSE in legend)")
        ax[3 * si + 2].legend(loc="upper right", fontsize=8, ncol=4)
    ax[-1].set_xlabel("time [s]"); fig.tight_layout(); fig.savefig(sys.argv[2], dpi=105)
else:
    seqs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PKG, "data/util/*")))
    groups = [(c, m) for c in ["const1", "const2", "const3", "const4"] for m in ["tdoa2", "tdoa3"]]
    S = {(s, name): score(s, os.path.join(D, f"{pre}{s}_pose.txt"), lag) for s in seqs for name, pre, lag, col in METH}
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, key, lab in [(axs[0], "rmse", "3D position RMSE [m]"), (axs[1], "xy", "horizontal (xy) RMSE [m]")]:
        for gi, (c, m) in enumerate(groups):
            ss = [s for s in seqs if s.startswith(c + "-") and m in s]
            for mi, (name, pre, lag, col) in enumerate(METH):
                v = np.array([S[(s, name)][key] for s in ss]); x = gi + (mi - 1) * 0.25
                ax.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), np.minimum(v, 3.0), s=14, color=col, alpha=0.8,
                           marker="o", label=name if gi == 0 else None)
                big = v > 3.0
                if big.any(): ax.scatter(np.full(big.sum(), x), np.full(big.sum(), 3.0), s=60, marker="x", color=col)
                ax.hlines(np.median(v), x - 0.1, x + 0.1, color=col, lw=2.5)
        ax.set_yscale("log"); ax.set_ylim(0.015, 3.5); ax.set_ylabel(lab + " (x at 3 = diverged)"); ax.grid(alpha=0.3, which="both")
    axs[1].set_xticks(range(len(groups))); axs[1].set_xticklabels([f"{c}\n{m}" for c, m in groups])
    axs[0].set_title("UTIL flight dataset, all 79 flights: KIMS vs SFUISE vs UTIL devkit ESKF (bar = median)"); axs[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110)
print("saved", sys.argv[2])
