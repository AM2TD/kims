#!/usr/bin/env python3
"""x, y, z over time for KIMS (sampling) and the matched LM (final runs) on the flights where they differ, with the ground
truth and, in z, the ground truth reflected about the anchor plane (an estimate on that dashed line is in the mirror branch).
usage: python3 scripts/revision/fig_lm_vs_kims.py [out_png]   (package root; default result/figs/lm_vs_kims_final.png)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es, eval_flip as ef
os.chdir(es.PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/lm_vs_kims_final.png"
CASES = [("iului", "SquareTurn", "f32g_SquareTurn", "lmf_SquareTurn", 199), ("iului", "CircleSpin", "f32g_CircleSpin", "lmf_CircleSpin", 199),
         ("miluv", "default_1_random3_0", "f32a_default_1_random3_0", "lmk_r30", 228), ("miluv", "obstacles_1_random3_0b", "f32a_obstacles_1_random3_0b", "lmk_obs", 228)]
def est(ds, lab, lag):
    P = es.rows(f"result/{ds}/diag/{lab}_pose.txt"); return P[:-lag, 0], P[lag:, 1:4]
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(3, len(CASES), figsize=(4.2 * len(CASES), 7.2), sharex="col")
for c, (ds, seq, kl, ll, lag) in enumerate(CASES):
    gt_t, gp = ef.load_gt(ds, seq); cc, n = ef.anchor_plane(ds, seq); h = (gp - cc) @ n; gm = gp - 2 * h[:, None] * n[None, :]
    t0 = gt_t[0]; res = {}
    for name, lab, col in [("KIMS", kl, "#1f4e9c"), ("LM", ll, "#c0392b")]:
        th, X = est(ds, lab, lag); k = (th >= gt_t[0]) & (th <= gt_t[-1]); th, X = th[k], X[k]
        e = X - np.stack([np.interp(th, gt_t, gp[:, j]) for j in range(3)], 1); r = np.sqrt(np.mean(np.sum(e ** 2, 1)))
        for j in range(3): ax[j, c].plot(th - t0, X[:, j], color=col, lw=1.0, label=name, zorder=3)
        res[name] = r
    for j in range(3): ax[j, c].plot(gt_t - t0, gp[:, j], "k--", lw=1.1, label="ground truth", zorder=4)
    ax[2, c].plot(gt_t - t0, gm[:, 2], color="0.55", ls=":", lw=1.2, label="GT reflected about the anchor plane", zorder=2)
    ax[2, c].axhline(cc[2], color="0.75", lw=0.8, zorder=1)
    ax[0, c].set_title(f"{ds.upper()} {seq}\n3D RMSE: KIMS {res['KIMS']:.3f} m, LM {res['LM']:.3f} m", fontsize=9)
    for j, nm in enumerate("xyz"):
        ax[j, c].grid(ls="--", lw=0.4, alpha=0.5)
        if c == 0: ax[j, c].set_ylabel(f"{nm} (m)")
    ax[2, c].set_xlabel("time (s)")
fig.suptitle("KIMS (sampling) vs the matched LM: same objective, window, prior and rollouts; final runs", fontsize=10)
h, l = ax[2, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=150); print("wrote", out)
