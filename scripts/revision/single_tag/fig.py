#!/usr/bin/env python3
import os
"""X/Y/Z (one sequence) or Z-only (all sequences) of GT, the 4-tag final KIMS run and the single-tag runs, each estimate
mapped into the Leica frame with its own 4-DoF alignment. The dashed line is the anchor plane (from the best run's alignment).
usage: plot1tag.py <plan.txt> xyz <seq> <out.png> | plot1tag.py <plan.txt> z <out.png>"""
import sys, os, csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); T_BP = np.array([-0.293656, -0.012288, -0.273095])
COL = {"4 tags": "#1f4e9c", "2000": "#e07b00", "2001": "#2a9d4b", "2010": "#c2185b", "2011": "#7b52ab"}
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def load(seq, f, lag, lt, lp):
    P = rows(f); th = P[:-lag, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    E = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); Ek = E[k]
    gc = G.mean(0); ec = Ek.mean(0); g2 = G[:, :2] - gc[:2]; e2 = Ek[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    rm = np.sqrt(np.mean(np.sum((Ek - ((Rz @ G.T).T + t)) ** 2, 1)))
    return th, E, (Rz, t), rm                                    # estimate kept in its own UWB frame
def runs(seq, plan):
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'{PKG}/data/ntu/{seq}/leica.csv')))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    R = [("4 tags", f"{PKG}/result/ntu/diag/" + ("w118_" if seq.startswith("eee") else "n_") + f"{seq}_pose.txt", 117)]
    R += [(p[1], f"{PKG}/result/ntu/diag/t1_{seq}_{p[1]}_pose.txt", int(p[4]) - 1) for p in plan if p[0] == seq]
    out = []
    for name, f, lag in R:
        if not os.path.exists(f) or len(rows(f)) <= lag + 10: continue
        out.append((name,) + load(seq, f, lag, lt, lp))
    A = np.loadtxt(f"{PKG}/data/ntu/" + ("anchors_eee.txt" if seq.startswith("eee") else f"anchors_{seq}.txt"), comments="#")
    zplane = A[:, 2].mean()
    if seq.startswith("eee"):
        sys.path.insert(0, PKG + "/scripts"); from eval_flip import load_gt
        gt_t, gt_p = load_gt("ntu", seq)
    else:
        Rz, t = min(out, key=lambda r: r[4])[3]; gt_t, gt_p = lt, (Rz @ lp.T).T + t
    return gt_t, gt_p, out, zplane
plan = [l.split() for l in open(sys.argv[1])]
if sys.argv[2] == "xyz":
    seq, png = sys.argv[3], sys.argv[4]; lt, lp, out, zp = runs(seq, plan)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    for j, nm in enumerate("xyz"):
        a = ax[j]; a.plot(lt - lt[0], lp[:, j], color="black", lw=2.2, label="GT (Leica prism, UWB frame)")
        for name, th, E, t, rm in out:
            lab = f"{name} ({rm:.2f} m)" if name == "4 tags" else f"tag {name} only ({rm:.2f} m)"
            a.plot(th - lt[0], E[:, j], color=COL[name], lw=1.1 if name != "4 tags" else 1.6, label=lab)
        if nm == "z": a.axhline(zp, color="gray", ls="--", lw=1, label="anchor plane")
        a.set_ylabel(f"{nm} [m]"); a.grid(alpha=0.3)
    ax[0].set_title(f"NTU {seq}: KIMS with 4 tags vs one tag (window held at the same length in seconds), UWB frame, RMSE (4-DoF aligned) in legend")
    ax[2].legend(loc="upper right", fontsize=8, ncol=3); ax[2].set_xlabel("time since GT start [s]")
    fig.tight_layout(); fig.savefig(png, dpi=110)
else:
    png = sys.argv[3]; seqs = ["eee_01","eee_02","eee_03","nya_01","nya_02","nya_03","sbs_01","sbs_02","sbs_03"]
    fig, ax = plt.subplots(9, 1, figsize=(13, 22))
    for i, seq in enumerate(seqs):
        lt, lp, out, zp = runs(seq, plan); a = ax[i]
        a.plot(lt - lt[0], lp[:, 2], color="black", lw=2.2, label="GT")
        for name, th, E, t, rm in out:
            a.plot(th - lt[0], E[:, 2], color=COL[name], lw=1.0 if name != "4 tags" else 1.5, label=(f"{name} {rm:.2f}" if name == "4 tags" else f"tag {name} {rm:.2f}"))
        a.axhline(zp, color="gray", ls="--", lw=1); a.set_ylabel(f"{seq}\nz [m]"); a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=7, ncol=6)
    ax[0].set_title("NTU z (UWB frame): GT, KIMS 4 tags, KIMS one tag; dashed = anchor plane; RMSE [m] in legend")
    ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fig.savefig(png, dpi=100)
print("saved", png)
