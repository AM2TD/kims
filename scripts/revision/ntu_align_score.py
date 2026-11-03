#!/usr/bin/env python3
"""NTU VIRAL scoring with a 4-DoF (yaw + translation) Umeyama alignment of the Leica prism track to the estimate
(estimate converted to the prism with its own attitude and T_Body_Prism), as the dataset's evaluation does for the
estimate. usage: ntu_align_score.py <seq> <pose_file> <lag_steps> [--leica path]  -> rmse xy z, alignment"""
import sys, os, csv, argparse, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('seq'); ap.add_argument('pose'); ap.add_argument('lag', type=int); ap.add_argument('--leica', default=None); ap.add_argument('--quiet', action='store_true')
a = ap.parse_args()
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
P = rows(a.pose); lag = a.lag
th = P[:-lag, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
lf = a.leica or os.path.join(PKG, 'data', 'ntu', a.seq, 'leica.csv')
L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(lf)))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
k = (th > lt[0] + 10) & (th < lt[-1] - 10)
G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
# 4-DoF alignment G' = Rz(yaw) G + t  minimising |E - G'|: yaw from the 2D Procrustes of centred xy, t from the means
gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
Ga = (Rz @ G.T).T + t; e = E - Ga
res = dict(rmse=np.sqrt(np.mean(np.sum(e ** 2, 1))), xy=np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1))), z=np.sqrt(np.mean(e[:, 2] ** 2)), yaw=np.degrees(yaw), t=t)
if not a.quiet: print(f"{a.seq} {os.path.basename(a.pose)}: 4-DoF aligned rmse {res['rmse']:.3f} xy {res['xy']:.3f} z {res['z']:.3f} | align yaw {res['yaw']:.2f} deg t ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
else: print(f"{res['rmse']:.3f} {res['xy']:.3f} {res['z']:.3f} {res['yaw']:.2f} {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}")
