#!/usr/bin/env python3
"""NTU VIRAL X/Y/Z of Leica GT, KIMS, SFUISE and the fixed-lag ESKF.
Positions are the prism point in the UWB anchor frame. GT enters that frame through the dataset transform (eee) or through the
4-DoF fit of the run with the lowest RMSE (nya, sbs; the dataset gives no transform there). The z panels also draw GT reflected
about the anchor plane (z = 1.5 m): an estimate on that dashed line is in the mirror branch. Legend: 3D RMSE with each method's
own 4-DoF fit (as in final_table_v2.txt) and the mirror-branch share of the time GT is more than 0.5 m from the plane.
usage: fig_ntu_baselines.py <out_prefix> [eee nya sbs]  ->  <out_prefix>_<site>.png"""
import sys, os, csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(PKG, "scripts"))
import eval_flip as ef
T_BP = np.array([-0.293656, -0.012288, -0.273095]); PLANE = 1.5
METH = [("KIMS", "result/ntu/diag/{k}{s}_pose.txt", 117, "#1f4e9c"), ("SFUISE", "result/ntu/diag/sfuise_{s}_pose.txt", 0, "#e07b00"),
        ("ESKF fixed-lag", "result/baselines/ekfl_{s}_pose.txt", 0, "#2a9d4b")]
def Rof(q):
    x, y, z, w = q / np.linalg.norm(q); return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def prism(f, lag):
    P = np.array([[float(x) for x in l.split()[:8]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])
    th = P[:-lag, 0] if lag else P[:, 0]; X, Q = P[lag:, 1:4], P[lag:, 4:8]
    return th, np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
def fit4(lt, lp, th, E):   # 4-DoF (yaw + translation) fit of Leica to the estimate over gt+10 .. end-10 s; returns rmse, mapped GT
    k = (th > lt[0] + 10) & (th < lt[-1] - 10); G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = E[k]
    if not np.all(np.isfinite(E)) or np.abs(E).max() > 1e3: return np.inf, None
    gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    return np.sqrt(np.mean(np.sum((E - ((Rz @ G.T).T + t)) ** 2, 1))), (Rz @ lp.T).T + t
def rmse_str(r, E): return "diverged" if not np.isfinite(r) else f"{r:.3f} m"
sites = sys.argv[2:] or ["eee", "nya", "sbs"]
for site in sites:
    seqs = [f"{site}_0{i}" for i in (1, 2, 3)]; fig, ax = plt.subplots(3 * len(seqs), 1, figsize=(13, 2.4 * 3 * len(seqs)))
    for si, s in enumerate(seqs):
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(os.path.join(PKG, f"data/ntu/{s}/leica.csv"))))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
        runs = {}
        for name, pat, lag, col in METH:
            f = os.path.join(PKG, pat.format(k="w118_" if site == "eee" else "n_", s=s))
            if os.path.exists(f): th, E = prism(f, lag); r, Gm = fit4(lt, lp, th, E); runs[name] = (th, E, r, Gm, col)
        if site == "eee": gt_t, gp = ef.load_gt("ntu", s); ref = "GT via the dataset transform"
        else:
            best = min(runs, key=lambda n: runs[n][2]); gt_t, gp = lt, runs[best][3]; ref = f"GT via the 4-DoF fit of {best} ({runs[best][2]:.2f} m)"
        t0 = gt_t[0]; span = (gt_t > gt_t[0] + 10) & (gt_t < gt_t[-1] - 10)
        for j, nm in enumerate("xyz"):
            a = ax[3 * si + j]; a.plot(gt_t - t0, gp[:, j], color="black", lw=2.2, label="Leica GT")
            if j == 2:
                a.plot(gt_t - t0, 2 * PLANE - gp[:, 2], color="0.45", lw=1.2, ls="--", label="GT reflected about the anchor plane")
                a.axhline(PLANE, color="0.6", lw=0.8, ls=":")
            for name, (th, E, r, Gm, col) in runs.items():
                k = (th >= gt_t[0]) & (th <= gt_t[-1]); lab = f"{name} (3D {rmse_str(r, E)}"
                if j == 2:
                    kk = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10); hg = np.interp(th[kk], gt_t, gp[:, 2]) - PLANE; he = E[kk, 2] - PLANE
                    far = np.abs(hg) > 0.5; ok = np.isfinite(he) & (np.abs(he) < 1e3)
                    mir = far & ok & (np.abs(he + hg) + 0.3 < np.abs(he - hg))
                    lab += ", mirror " + (f"{100 * mir.sum() / max(far.sum(), 1):.1f}%" if ok.all() else "n/a") + ")"
                else: lab += ")"
                a.plot(th[k] - t0, E[k, j], color=col, lw=1.1, label=lab)
            lo, hi = gp[span, j].min(), gp[span, j].max()
            if j == 2: lo, hi = min(lo, 2 * PLANE - hi), max(hi, 2 * PLANE - lo)
            a.set_ylim(lo - 1.0, hi + 1.0); a.set_ylabel(f"{nm} [m]"); a.grid(alpha=0.3)   # a diverged estimate leaves the axis
            if j == 0: a.set_title(f"NTU {s}: prism position in the UWB anchor frame ({ref}); dotted z = anchor plane")
        ax[3 * si + 2].legend(loc="upper right", fontsize=7.5, ncol=3)
    ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fig.savefig(f"{sys.argv[1]}_{site}.png", dpi=100); plt.close(fig)
    print("wrote", f"{sys.argv[1]}_{site}.png")
