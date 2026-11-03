#!/usr/bin/env python3
"""IULUI AC1: X/Y/Z of the Qualisys GT, KIMS (final configuration: FP32 rollout + 8 host threads, f32a_*, lag T with the
window flush) and SFUISE (given label), whole GT span, no alignment. z panels also show the anchor plane (mean anchor height).
usage: fig_iului_sfuise.py <out.png> <sfuise_label> [seq ...]"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])            # rows, std_score, ef
out, sf = sys.argv[1], sys.argv[2]; seqs = sys.argv[3:] or ["SquareTurn", "CircleSpin", "SquareBob"]
KL = os.environ.get("KIMS_LABEL", "f32g")   # f32g: grouped same-timestamp epochs (group_same_ts:=1) with the end flush
def track(f, lag):
    R = rows(f); th = R[:-lag, 0] if lag else R[:, 0]; X = R[lag:, 1:4]
    ff = f.replace("_pose.txt", "_flush_pose.txt")
    if lag and os.path.exists(ff):
        F = rows(ff); n = len(F) // 2; first, last = F[:n], F[n:]; mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
    return th, X
fig, ax = plt.subplots(3 * len(seqs), 1, figsize=(13, 2.4 * 3 * len(seqs)))
for si, s in enumerate(seqs):
    gt_t, gp = ef.load_gt("iului", s); mp = f"data/iului/AC1_{s}.json" if os.path.exists(f"data/iului/AC1_{s}.json") else f"data/iului/{s}.json"
    anc = np.array(json.load(open(mp))["anchors"]); zp = anc[:, 2].mean() if np.ptp(anc[:, 2]) < 0.1 else None   # plane only for (quasi-)coplanar anchors
    runs = [("KIMS", f"result/iului/diag/{KL}_{s}_pose.txt", 199, "#1f4e9c"), ("SFUISE", f"result/iului/diag/{sf}_{s}_pose.txt", 0, "#e07b00")]
    for j, nm in enumerate("xyz"):
        a = ax[3 * si + j]; a.plot(gt_t - gt_t[0], gp[:, j], color="black", lw=2.2, label="Qualisys GT")
        if j == 2 and zp is not None: a.axhline(zp, color="0.6", lw=0.8, ls=":", label="anchor plane")
        for name, f, lag, col in runs:
            if not os.path.exists(f): continue
            th, X = track(f, lag); k = (th >= gt_t[0]) & (th <= gt_t[-1])
            a.plot(th[k] - gt_t[0], X[k, j], color=col, lw=1.1, label=f"{name} ({std_score('iului', s, f, lag):.3f} m)")
        lo, hi = gp[:, j].min(), gp[:, j].max(); a.set_ylim(lo - 0.5, hi + 0.5); a.set_ylabel(f"{nm} [m]"); a.grid(alpha=0.3)
        if j == 0: a.set_title(f"IULUI {s}: position (3D RMSE in legend, whole GT span, no alignment)")
    ax[3 * si + 2].legend(loc="upper right", fontsize=8, ncol=4)
ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fig.savefig(out, dpi=100); print("wrote", out)
