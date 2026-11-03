#!/usr/bin/env python3
"""Seed variability of KIMS in the final configuration (sweep_final.sh, phase S): 3D RMSE of every sweep sequence for
seeds 1..10 (dots), their mean and standard deviation (bar and line), and the table seed 1234 (diamond); per dataset.
Bottom row: share of time on the mirror side of the anchor plane (|h_GT| > 0.1 m) on the coplanar-anchor flights with a
ground truth in the anchor frame (IULUI AC1, MILUV constellation 1, NTU VIRAL eee).
usage: fig_seeds_final.py [out_prefix]   (default result/figs/fig_seeds_final -> .png/.pdf and a text summary)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import err_series as es
from eval_flip import anchor_plane
PKG = es.PKG; os.chdir(PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/fig_seeds_final"
SEEDS = list(range(1, 11))
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
SHORT = lambda s: s.replace("default_1_", "").replace("remoteControl", "RC").replace("noAprilTags_1_", "noAT ").replace("obstacles_1_", "obst ")
DS = [("MILUV", "miluv", MIL, "f32a", 228), ("NTU VIRAL", "ntu", ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03'], "f32a", 117),
      ("UTIL", "util", ["const1-trial1-tdoa2", "const1-trial1-tdoa3", "const2-trial1-tdoa2", "const2-trial1-tdoa3", "const3-trial1-tdoa2", "const3-trial1-tdoa3", "const4-trial1-tdoa2-traj1", "const4-trial1-tdoa3-traj1"], "uf1", 99),
      ("IULUI", "iului", ["SquareTurn", "CircleSpin", "AC3_125000"], "f32g", 199)]
def finished(f):   # the run's log says it finished (in-progress outputs would score only their first part)
    lab = f.split("/diag/", 1)[1][:-len("_pose.txt")]; lg = os.path.join(PKG, "result/runlogs", lab + ".log")
    return os.path.exists(lg) and "solves in" in open(lg, errors="ignore").read()
def err(ds, s, f, lag):
    if not finished(f): return None
    r = es.err_ntu(s, f, lag) if ds == "ntu" else es.err_util(s, f, lag) if ds == "util" else es.err_std(ds, s, f, lag)
    return r
def rmse(r): return np.sqrt(np.mean(np.sum(r[1] ** 2, 1))) if r is not None else np.nan
def wrong(ds, s, f, lag):   # share of time on the mirror side (GT in the anchor frame: IULUI, MILUV, NTU eee only)
    if not os.path.exists(f) or not finished(f): return np.nan
    gt_t, gt = es.ef.load_gt(ds, s); c, n = anchor_plane(ds, s); P = es.rows(f); th = P[:-lag, 0]; X = P[lag:, 1:4]
    k = (th >= gt_t[0]) & (th <= gt_t[-1]); G = np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)
    he, hg = (X[k] - c) @ n, (G - c) @ n; v = np.abs(hg) > 0.1
    return float(np.mean(np.sign(he[v]) != np.sign(hg[v])))
COPL = [("iului", "SquareTurn", "f32g", 199), ("iului", "CircleSpin", "f32g", 199), ("miluv", "default_1_random3_1", "f32a", 228),
        ("miluv", "default_1_remoteControlHighPace_1", "f32a", 228), ("ntu", "eee_01", "f32a", 117), ("ntu", "eee_02", "f32a", 117), ("ntu", "eee_03", "f32a", 117)]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(12, 6.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], width_ratios=[13, 9, 8, 3.5])
lines = []
for j, (name, ds, seqs, lab, lag) in enumerate(DS):
    a = fig.add_subplot(gs[0, j])
    for i, s in enumerate(seqs):
        v = np.array([rmse(err(ds, s, f"result/{ds}/diag/sweep/sd{sd}_{s}_pose.txt", lag)) for sd in SEEDS]); d = rmse(err(ds, s, f"result/{ds}/diag/{lab}_{s}_pose.txt", lag))
        ok = np.isfinite(v)
        if ok.any():
            x = i + 1 + np.linspace(-0.18, 0.18, len(v))[ok]
            a.scatter(x, v[ok], s=9, color="#b03a2e", alpha=0.55, lw=0, zorder=2)
            m, sd_ = v[ok].mean(), v[ok].std(); a.plot([i + 0.7, i + 1.3], [m, m], color="#b03a2e", lw=1.6, zorder=3); a.plot([i + 1, i + 1], [m - sd_, m + sd_], color="#b03a2e", lw=1.0, zorder=3)
        a.scatter([i + 1], [d], marker="D", s=26, facecolor="none", edgecolor="k", lw=1.0, zorder=4)
        lines.append(f"{name:9s} {s:40s} default {d:.3f} | seeds n {ok.sum():2d} mean {np.nanmean(v):.3f} sd {np.nanstd(v):.4f} min {np.nanmin(v) if ok.any() else np.nan:.3f} max {np.nanmax(v) if ok.any() else np.nan:.3f}")
    a.set_xticks(range(1, len(seqs) + 1)); a.set_xticklabels([SHORT(s) for s in seqs], rotation=60, ha="right", fontsize=7)
    a.set_title(name); a.grid(axis="y", ls="--", lw=0.5, alpha=0.5)
    if ds == "ntu":   # 0.17-3.1 m: log axis with plain labels
        a.set_yscale("log"); a.set_yticks([0.2, 0.5, 1, 2, 3]); a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else: a.set_ylim(0, a.get_ylim()[1] * 1.1)
    if j == 0: a.set_ylabel("3D RMSE (m)")
b = fig.add_subplot(gs[1, :2])
for i, (ds, s, lab, lag) in enumerate(COPL):
    v = np.array([wrong(ds, s, f"result/{ds}/diag/sweep/sd{sd}_{s}_pose.txt", lag) for sd in SEEDS]); d = wrong(ds, s, f"result/{ds}/diag/{lab}_{s}_pose.txt", lag); ok = np.isfinite(v)
    if ok.any():
        b.scatter(i + 1 + np.linspace(-0.18, 0.18, len(v))[ok], 100 * v[ok], s=9, color="#b03a2e", alpha=0.55, lw=0); m = 100 * v[ok].mean(); b.plot([i + 0.7, i + 1.3], [m, m], color="#b03a2e", lw=1.6)
    b.scatter([i + 1], [100 * d], marker="D", s=26, facecolor="none", edgecolor="k", lw=1.0, zorder=4)
    lines.append(f"mirror side {ds:5s} {s:36s} default {100 * d:5.1f} % | seeds mean {100 * np.nanmean(v):5.1f} % max {100 * np.nanmax(v) if ok.any() else np.nan:5.1f} %")
b.set_xticks(range(1, len(COPL) + 1)); b.set_xticklabels([f"{ {'iului': 'IULUI', 'miluv': 'MILUV', 'ntu': 'NTU'}[d]} {SHORT(s)}" for d, s, _, _ in COPL], rotation=20, ha="right", fontsize=8)
b.set_ylabel("time on mirror side (%)"); b.grid(axis="y", ls="--", lw=0.5, alpha=0.5); b.set_title("coplanar anchors", fontsize=9)
L = fig.add_subplot(gs[1, 2:]); L.axis("off")
L.scatter([], [], s=12, color="#b03a2e", alpha=0.6, label="seeds 1-10"); L.plot([], [], color="#b03a2e", lw=1.6, label="mean ± s.d. over seeds")
L.scatter([], [], marker="D", s=26, facecolor="none", edgecolor="k", label="table seed (1234)"); L.legend(loc="center", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
open(out + ".txt", "w").write("\n".join(lines) + "\n"); print("wrote", out + ".png/.pdf/.txt"); print("\n".join(lines))
