#!/usr/bin/env python3
"""Paper figure (replaces traj_all): per-axis position over time for one representative sequence per dataset, final
configuration: KIMS (fixed-lag output with the window flush), SFUISE (published configuration) and the dataset filter
(MILUV devkit EKF with the height sensor, UTIL devkit ESKF), against the ground truth. NTU VIRAL: Leica prism point in the UWB
anchor frame (dataset transform). The column titles give each method's 3D RMSE with the table scoring.
usage: fig_traj_final.py [out_prefix]   (default result/figures/fig_traj -> .pdf and .png)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figures/fig_traj"
src = open("scripts/revision/fig_final_xyz.py").read().split("for page in range(")[0]
sys.argv = ["fig_final_xyz.py", "miluv", "unused"]; ns = {"__file__": os.path.join(PKG, "scripts/revision/fig_final_xyz.py")}
exec(src, ns)                                   # series(seq) of fig_final_xyz.py, dataset chosen through ns["ds"]
CASES = [("miluv", "default_1_random3_0", "MILUV / random3_0"), ("ntu", "eee_01", "NTU VIRAL / eee_01"),
         ("util", "const4-trial1-tdoa2-traj1", "UTIL / const4-trial1-tdoa2-traj1"), ("iului", "CircleSpin", "IULUI / CircleSpin")]
COL = {"KIMS": "#c0392b", "SFUISE": "#e08a1e", "devkit": "#2a7f62"}
NAME = {"miluv": "EKF + height", "util": "devkit ESKF"}
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(3, 4, figsize=(7.16, 4.3), sharex="col")
for c, (ds, s, title) in enumerate(CASES):
    ns["ds"] = ds; gt_t, gp, series, _ = ns["series"](s); t0 = gt_t[0]
    v = {n: ("div." if not np.isfinite(r) or r > 100 else f"{r:.3f}") for n, _, _, r in series}
    rm = f"KIMS {v['KIMS']}, SFUISE {v.get('SFUISE', '-')}" + (f"\n{NAME[ds]} {v['devkit']}" if "devkit" in v else "\n")
    for j, lab in enumerate("xyz"):
        a = ax[j, c]
        for n, th, X, r in reversed(series):
            k = (th >= gt_t[0]) & (th <= gt_t[-1])
            a.plot(th[k] - t0, X[k, j], color=COL[n], lw=0.9 if n == "KIMS" else 0.7, zorder=3 if n == "KIMS" else 2)
        a.plot(gt_t - t0, gp[:, j], "k--", lw=0.8, zorder=4)
        lo, hi = np.nanmin(gp[:, j]), np.nanmax(gp[:, j]); pad = 0.25 * (hi - lo) + 0.3; a.set_ylim(lo - pad, hi + pad)
        a.grid(ls="--", lw=0.4, alpha=0.5); a.tick_params(labelsize=6.5, length=2)
        if c == 0: a.set_ylabel(f"{lab} (m)")
        if j == 2: a.set_xlabel("time (s)")
    ax[0, c].set_title(f"{title}\n{rm}", fontsize=6.5)
from matplotlib.lines import Line2D
h = [Line2D([], [], color="k", ls="--", lw=0.8), Line2D([], [], color=COL["KIMS"], lw=1.2), Line2D([], [], color=COL["SFUISE"], lw=1.0), Line2D([], [], color=COL["devkit"], lw=1.0)]
fig.legend(h, ["ground truth", "KIMS", "SFUISE", "dataset filter (MILUV: devkit EKF + height, UTIL: devkit ESKF)"], loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.005),
           title="column titles: 3D RMSE in m, table scoring", title_fontsize=6.5)
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.4, w_pad=0.6)
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); print("wrote", out + ".pdf/.png")
