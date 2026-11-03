#!/usr/bin/env python3
import os
"""Score the single-tag NTU runs against the 4-tag final runs: 4-DoF aligned prism RMSE (same scorer as the final table),
xy / z split, and the time on the mirror branch (estimate closer to the reflection of GT about the anchor plane by
> 0.3 m while GT is > 0.5 m from the plane). usage: score1tag.py <plan.txt> <out.txt>"""
import sys, os, csv, numpy as np
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def score(seq, f, lag):
    if not os.path.exists(f): return None
    P = rows(f)
    if len(P) <= lag + 10: return None
    th = P[:-lag, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'{PKG}/data/ntu/{seq}/leica.csv')))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    if not np.all(np.isfinite(E)): return dict(rmse=np.inf, xy=np.inf, z=np.inf, mirror=np.nan)
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    Ga = (Rz @ G.T).T + t; e = E - Ga
    A = np.loadtxt(f"{PKG}/data/ntu/" + ("anchors_eee.txt" if seq.startswith("eee") else f"anchors_{seq}.txt"), comments="#")
    zp = A[:, 2].mean(); hg = Ga[:, 2] - zp; he = E[:, 2] - zp
    d_true = np.abs(he - hg); d_mir = np.abs(he + hg)
    mir = (np.abs(hg) > 0.5) & (d_mir + 0.3 < d_true)
    dt = np.diff(th[k], append=th[k][-1])
    return dict(rmse=np.sqrt(np.mean(np.sum(e ** 2, 1))), xy=np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1))), z=np.sqrt(np.mean(e[:, 2] ** 2)),
                mirror=float(np.sum(dt[mir])), span=float(th[k][-1] - th[k][0]))
plan = [l.split() for l in open(sys.argv[1])]
seqs = ["eee_01","eee_02","eee_03","nya_01","nya_02","nya_03","sbs_01","sbs_02","sbs_03"]
out = []
hdr = f"{'sequence':8s} {'run':>10s} | {'rmse':>7s} {'xy':>6s} {'z':>6s} | mirror s"
out.append(hdr); out.append("-" * len(hdr))
for s in seqs:
    base = f"{PKG}/result/ntu/diag/" + ("w118_" if s.startswith("eee") else "n_") + f"{s}_pose.txt"
    r = score(s, base, 117)
    out.append(f"{s:8s} {'4 tags':>10s} | {r['rmse']:7.3f} {r['xy']:6.3f} {r['z']:6.3f} | {r['mirror']:6.1f}")
    for p in plan:
        if p[0] != s: continue
        tg, T = p[1], int(p[4]); f = f"{PKG}/result/ntu/diag/t1_{s}_{tg}_pose.txt"
        r = score(s, f, T - 1)
        if r is None: out.append(f"{s:8s} {'tag ' + tg:>10s} |   (not run: tag data gap)" if not os.path.exists(f) else f"{s:8s} {'tag ' + tg:>10s} |   (incomplete)"); continue
        out.append(f"{s:8s} {'tag ' + tg:>10s} | {r['rmse']:7.3f} {r['xy']:6.3f} {r['z']:6.3f} | {r['mirror']:6.1f}")
    out.append("")
r = score("eee_03", f"{PKG}/result/ntu/diag/t4_eee_03_pose.txt", 117)
if r: out.append(f"rebuild check, eee_03 4 tags with the new binary: rmse {r['rmse']:.3f} (final table 1.845)")
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("\n".join(out))
