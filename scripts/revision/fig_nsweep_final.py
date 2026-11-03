#!/usr/bin/env python3
"""Sample-count sensitivity of KIMS in the final configuration (sweep_final.sh, phases T and N).
(a) 3D RMSE against N: median over the sweep sequences of each dataset (band: 25th-75th percentile); N 8192 = the table runs.
(b) per-epoch solve time against N, each run alone on the GPU (one flight per dataset): median (filled) with the 5th-95th
    percentile range and mean (open); dashed lines = mean epoch interval of that flight (the real-time budget). First 20 solves dropped.
usage: fig_nsweep_final.py [out_prefix]   (default result/figs/fig_nsweep_final -> .png/.pdf and a text summary)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import err_series as es
PKG = es.PKG; os.chdir(PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/fig_nsweep_final"
NS = [512, 1024, 2048, 4096, 8192, 16384]
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
DS = [("MILUV", "miluv", MIL, "f32a", 228, "default_1_random3_0", "#1f5fa8"),
      ("NTU VIRAL", "ntu", ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03'], "f32a", 117, "eee_01", "#6b8e23"),
      ("UTIL", "util", ["const1-trial1-tdoa2", "const1-trial1-tdoa3", "const2-trial1-tdoa2", "const2-trial1-tdoa3", "const3-trial1-tdoa2", "const3-trial1-tdoa3", "const4-trial1-tdoa2-traj1", "const4-trial1-tdoa3-traj1"], "uf1", 99, "const1-trial1-tdoa2", "#7b5ea7"),
      ("IULUI", "iului", ["SquareTurn", "CircleSpin", "AC3_125000"], "f32g", 199, "SquareTurn", "#b03a2e")]
def run_file(ds, s, lab, n, tseq):
    if n == 8192: return f"result/{ds}/diag/{lab}_{s}_pose.txt"
    return f"result/{ds}/diag/sweep/{'tn' if s == tseq else 'n'}{n}_{s}_pose.txt"
def finished(f):   # the run's log says it finished (in-progress outputs would score only their first part)
    lab = f.split("/diag/", 1)[1][:-len("_pose.txt")]; lg = os.path.join(PKG, "result/runlogs", lab + ".log")
    return os.path.exists(lg) and "solves in" in open(lg, errors="ignore").read()
def rmse(ds, s, f, lag):
    if not finished(f): return np.nan
    r = es.err_ntu(s, f, lag) if ds == "ntu" else es.err_util(s, f, lag) if ds == "util" else es.err_std(ds, s, f, lag)
    if r is None: return np.nan
    e = np.sqrt(np.mean(np.sum(r[1] ** 2, 1))); return e if np.isfinite(e) else 1e6
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4)); lines = []
for name, ds, seqs, lab, lag, tseq, col in DS:
    med, q1, q3, ns = [], [], [], []
    for n in NS:
        v = np.array([rmse(ds, s, run_file(ds, s, lab, n, tseq), lag) for s in seqs]); ok = np.isfinite(v)
        if ok.sum() < len(seqs): lines.append(f"{name} N {n}: {ok.sum()}/{len(seqs)} runs present")
        if not ok.any(): continue
        ns.append(n); med.append(np.median(v[ok])); q1.append(np.percentile(v[ok], 25)); q3.append(np.percentile(v[ok], 75))
        lines.append(f"{name:9s} N {n:5d}: median 3D RMSE {med[-1]:.3f} (IQR {q1[-1]:.3f}-{q3[-1]:.3f}), max {v[ok].max():.3f}, n {ok.sum()}")
    a1.plot(ns, med, "o-", color=col, lw=1.6, ms=4, label=name); a1.fill_between(ns, q1, q3, color=col, alpha=0.12, lw=0)
    mean, med_t, lo, hi, nt = [], [], [], [], []
    for n in NS:
        f = f"result/{ds}/diag/sweep/tn{n}_{tseq}"
        if not os.path.exists(f + "_timing.txt"): continue
        ms = np.loadtxt(f + "_timing.txt", ndmin=1)[20:]; P = es.rows(f + "_pose.txt")
        nt.append(n); mean.append(ms.mean()); med_t.append(np.median(ms)); lo.append(np.percentile(ms, 5)); hi.append(np.percentile(ms, 95)); bud = (P[-1, 0] - P[0, 0]) / (len(P) - 1) * 1e3
        lines.append(f"{name:9s} {tseq:22s} N {n:5d}: mean {ms.mean():6.2f} ms median {np.median(ms):6.2f} p95 {np.percentile(ms, 95):6.2f} | budget {bud:5.1f} ms, load {ms.mean() / bud:.2f}, over budget {100 * np.mean(ms > bud):4.1f} %")
    if nt:
        a2.errorbar(nt, med_t, yerr=[np.array(med_t) - lo, np.array(hi) - med_t], fmt="s-", color=col, ms=4, lw=1.4, capsize=3, label=f"{name} ({tseq.replace('default_1_', '')})")
        a2.plot(nt, mean, "o:", color=col, mfc="none", ms=4, lw=0.8)
        a2.axhline(bud, color=col, ls="--", lw=1.0)
for a in (a1, a2):
    a.set_xscale("log", base=2); a.set_xticks(NS); a.set_xticklabels([str(n) for n in NS]); a.set_xlabel("rollout samples N"); a.grid(ls="--", lw=0.5, alpha=0.5)
a1.set_yscale("log"); a1.set_ylabel("3D RMSE (m)"); a1.set_title("(a) accuracy vs. sample count"); a1.legend(fontsize=8, frameon=False)
a2.set_ylabel("solve time per epoch (ms)"); a2.set_title("(b) solve time per epoch, run alone", fontsize=10)
from matplotlib.lines import Line2D
h, l = a2.get_legend_handles_labels()
h += [Line2D([], [], color="#666", marker="o", mfc="none", ls=":", lw=0.8, ms=4), Line2D([], [], color="#666", ls="--", lw=1.0)]
l += ["mean (median: filled, bars: 5-95 %)", "epoch interval (budget)"]
a2.legend(h, l, fontsize=7, frameon=False, loc="upper right")
a2.set_ylim(0, 62)   # room above the data for the legend
fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
open(out + ".txt", "w").write("\n".join(lines) + "\n"); print("wrote", out + ".png/.pdf/.txt"); print("\n".join(lines))
