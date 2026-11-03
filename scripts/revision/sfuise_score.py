#!/usr/bin/env python3
"""Score an SFUISE output (body origin, lag 0) exactly as final_table.py scores the other columns.
MILUV: whole GT span, anchor frame, GT from eval_flip.load_gt (constellation z bias). NTU: prism with the estimate's own
attitude, Leica aligned by yaw + translation (4-DoF), window gt+10 s .. end-10 s.
usage: sfuise_score.py <miluv|ntu> <seq> <pose_file>  -> rmse xy z"""
import sys, os, csv, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(PKG, "scripts")); import eval_flip as ef
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()[:8]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])
def Rof(q):
    x, y, z, w = q / np.linalg.norm(q); return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def score(ds, seq, f):
    P = rows(f); th, X, Q = P[:, 0], P[:, 1:4], P[:, 4:8]
    if ds == "ntu":
        E = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'{PKG}/data/ntu/{seq}/leica.csv')))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
        k = (th > lt[0] + 10) & (th < lt[-1] - 10); G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = E[k]
        if not np.all(np.isfinite(E)) or len(E) < 10: return dict(rmse=np.inf, xy=np.inf, z=np.inf)
        gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
        yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
        c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); e = E - ((Rz @ G.T).T + (ec - Rz @ gc))
    else:
        gt_t, gt = ef.load_gt(ds, seq); k = (th >= gt_t[0]) & (th <= gt_t[-1])
        if k.sum() < 10 or not np.all(np.isfinite(X[k])): return dict(rmse=np.inf, xy=np.inf, z=np.inf)
        e = X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)
    return dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), xy=float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1)))), z=float(np.sqrt(np.mean(e[:, 2] ** 2))))
if __name__ == "__main__":
    r = score(sys.argv[1], sys.argv[2], sys.argv[3]); print(f"{r['rmse']:.3f} {r['xy']:.3f} {r['z']:.3f}")
