#!/usr/bin/env python3
"""z of GT, KIMS (sampling) and the matched LM (same target, same window) on the sequences where the two differ most, plus
nya_01 / nya_03. MILUV: anchor frame, mocap GT. NTU: prism in the UWB anchor frame, GT through the 4-DoF fit of the better run,
GT reflected about the anchor plane (z = 1.5 m) dashed. Legend: 3D RMSE as in final_table_v2.txt.
usage: fig_lm_examples.py <out.png>"""
import sys, os, csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); sys.path[:0] = ["scripts", "scripts/revision"]
import eval_flip as ef
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])            # rows, Rof, T_BP, sc
exec(open("scripts/revision/fig_ntu_baselines.py").read().split("def rmse_str")[0].split("import eval_flip as ef")[1])  # prism, fit4
CASES = [("ntu", "nya_01", "n_nya_01", "lmy_nya_01", 117), ("ntu", "nya_03", "n_nya_03", "lmy_nya_03", 117), ("ntu", "sbs_02", "n_sbs_02", "lmy_s02", 117),
         ("miluv", "default_1_random3_0", "fx_r30", "lmk_r30", 228), ("miluv", "obstacles_1_random3_0b", "fx_obstacles_1_random3_0b", "lmk_obs", 228)]
fig, ax = plt.subplots(len(CASES), 1, figsize=(13, 2.7 * len(CASES)))
for a, (ds, s, k, l, lag) in zip(ax, CASES):
    fk, fl = f"result/{ds}/diag/{k}_pose.txt", f"result/{ds}/diag/{l}_pose.txt"; rk, rl = sc(ds, s, fk, lag), sc(ds, s, fl, lag)
    if ds == "ntu":
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"data/ntu/{s}/leica.csv")))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
        runs = {n: prism(f, lag) for n, f in [("KIMS", fk), ("LM", fl)]}; fits = {n: fit4(lt, lp, *runs[n]) for n in runs}
        best = min(fits, key=lambda n: fits[n][0]); gt_t, gp = lt, fits[best][1]
        a.plot(gt_t - gt_t[0], 3.0 - gp[:, 2], color="0.45", lw=1.2, ls="--", label="GT reflected about the anchor plane"); a.axhline(1.5, color="0.6", lw=0.8, ls=":")
        tr = {n: runs[n] for n in runs}
    else:
        gt_t, gp = ef.load_gt("miluv", s); tr = {}
        for n, f in [("KIMS", fk), ("LM", fl)]:
            P = rows(f); tr[n] = (P[:-lag, 0], P[lag:, 1:4])
    t0 = gt_t[0]; a.plot(gt_t - t0, gp[:, 2], color="black", lw=2.2, label="GT")
    for n, col, r in [("KIMS", "#1f4e9c", rk), ("LM", "#c0392b", rl)]:
        th, X = tr[n]; m = (th >= gt_t[0]) & (th <= gt_t[-1]); a.plot(th[m] - t0, X[m, 2], color=col, lw=1.1, label=f"{n} ({r:.3f} m)")
    a.set_title(f"{ds.upper()} {s}: z, KIMS (sampling) vs matched LM; 3D RMSE in legend"); a.set_ylabel("z [m]"); a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=8, ncol=4)
ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=100); print("wrote", sys.argv[1])
