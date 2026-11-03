#!/usr/bin/env python3
"""Distribution of the per-epoch 3D position error on the four datasets (revised paper): KIMS (final configuration),
SFUISE (published configuration) and the dataset filter (MILUV devkit EKF with the height sensor, UTIL devkit ESKF), pooled over
the evaluated sequences of each dataset with the table scoring (err_series.py). Runs with an RMSE above 100 m count as
diverged and are left out of the pooled distribution (their number is printed under the box). Boxes: quartiles and
median; whiskers: 5th and 95th percentiles; no outlier markers.
usage: fig_boxplot_final.py [out_prefix]   (default result/figs/fig_boxplot_final -> .png and .pdf)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from err_series import err_std, err_ntu, err_util, err_devkit_miluv, PKG
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "result/figs/fig_boxplot_final")
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
NTU = ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']
UT = sorted(os.listdir(os.path.join(PKG, "data/util")))
IU = ['SquareTurn', 'CircleSpin', 'AC3_125000']
SF = lambda ds, s: f"result/{ds}/diag/sfuise_{s}_pose.txt"
D = {   # dataset -> list of (method, function(seq) -> (t, e) or None)
    "MILUV": (MIL, [("KIMS", lambda s: err_std("miluv", s, f"result/miluv/diag/f32a_{s}_pose.txt", 228)), ("SFUISE", lambda s: err_std("miluv", s, SF("miluv", s), 0)), ("EKF + height", err_devkit_miluv)]),
    "NTU VIRAL": (NTU, [("KIMS", lambda s: err_ntu(s, f"result/ntu/diag/f32a_{s}_pose.txt", 117)), ("SFUISE", lambda s: err_ntu(s, SF("ntu", s), 0))]),
    "UTIL": (UT, [("KIMS", lambda s: err_util(s, f"result/util/diag/uf1_{s}_pose.txt", 99)), ("SFUISE", lambda s: err_util(s, f"result/util/diag/sfuisebody_{s}_pose.txt", 0)), ("devkit ESKF", lambda s: err_util(s, f"result/util/diag/eskf_{s}_pose.txt", 0))]),
    "IULUI": (IU, [("KIMS", lambda s: err_std("iului", s, f"result/iului/diag/f32g_{s}_pose.txt", 199)), ("SFUISE", lambda s: err_std("iului", s, SF("iului", s), 0))])}
COL = {"KIMS": "#b03a2e", "SFUISE": "#e08a1e", "EKF + height": "#2a7f62", "devkit ESKF": "#2a7f62"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 4, figsize=(11, 3.3))
summary = []
for a, (name, (seqs, methods)) in zip(ax, D.items()):
    data, labels, notes, cols = [], [], [], []
    for m, fn in methods:
        errs, ndiv, nseq = [], 0, 0
        for s in seqs:
            r = fn(s)
            if r is None: continue
            nseq += 1; e = np.linalg.norm(r[1], axis=1)
            if not np.all(np.isfinite(e)) or np.sqrt(np.mean(e ** 2)) > 100: ndiv += 1; continue
            errs.append(e)
        e = np.concatenate(errs) if errs else np.array([np.nan])
        data.append(e); labels.append(m); cols.append(COL[m]); notes.append(f"{ndiv} div." if ndiv else "")
        summary.append(f"{name:9s} {m:12s} seqs {nseq:2d} diverged {ndiv} | median {np.nanmedian(e):.3f} p95 {np.nanpercentile(e, 95):.3f} rms {np.sqrt(np.nanmean(e ** 2)):.3f}")
    b = a.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=False, medianprops=dict(linewidth=1.6))
    for i, c in enumerate(cols):
        plt.setp(b["boxes"][i], color=c, linewidth=1.4); plt.setp(b["medians"][i], color=c, linewidth=1.8)
        plt.setp(b["whiskers"][2 * i:2 * i + 2], color=c, linewidth=1.2); plt.setp(b["caps"][2 * i:2 * i + 2], color=c, linewidth=1.2)
    a.set_xticks(range(1, len(labels) + 1)); a.set_xticklabels([l.replace(" + ", "\n+ ").replace("devkit ", "devkit\n") for l in labels], rotation=0, fontsize=9)
    top = max(np.nanpercentile(x, 95) for x in data) * 1.15
    a.set_ylim(0, top)
    for i, nt in enumerate(notes):
        if nt: a.text(i + 1, top * 0.97, nt, ha="center", va="top", fontsize=8, color="#555")
    a.set_title(f"{name} ({len(seqs)})"); a.grid(axis="y", ls="--", lw=0.5, alpha=0.6)
ax[0].set_ylabel("3D position error (m)")
fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); print("wrote", out + ".png/.pdf")
print("\n".join(summary))
