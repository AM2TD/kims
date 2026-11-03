#!/usr/bin/env python3
"""X/Y/Z of NTU sequences under the 4-DoF alignment (Leica prism track aligned to the estimate converted to the prism).
usage: fig_ntu_xyz_aligned.py <out.png> <seq:pose:lag> ..."""
import sys, os, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
specs = [s.split(':') for s in sys.argv[2:]]
fig, ax = plt.subplots(len(specs) * 3, 1, figsize=(12, 2.2 * 3 * len(specs)))
for si, (seq, pose, lag) in enumerate(specs):
    lag = int(lag); P = rows(pose); th = P[:-lag, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(os.path.join(PKG, 'data', 'ntu', seq, 'leica.csv'))))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    La = (Rz @ lp.T).T + t; Ga = (Rz @ G.T).T + t; e = E - Ga
    rm = np.sqrt(np.mean(np.sum(e ** 2, 1)))
    for j, name in enumerate('xyz'):
        a = ax[3 * si + j]; t0 = lt[0]
        a.plot(lt - t0, La[:, j], 'k', lw=1.2, label='Leica GT (4-DoF aligned)')
        a.plot(th - t0, prism[:, j], color='tab:red', lw=0.9, label=f'KIMS (prism)  {name} rms {np.sqrt(np.mean(e[:, j] ** 2)):.3f} m')
        a.set_ylabel(f'{name} [m]'); a.grid(alpha=0.3); a.legend(fontsize=8, loc='upper right', ncol=2); a.set_xlim(-2, lt[-1] - t0 + 2)
        if j == 0: a.set_title(f'{seq}: 3D rmse {rm:.3f} m (xy {np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1))):.3f}, z {np.sqrt(np.mean(e[:, 2] ** 2)):.3f})', loc='left', fontsize=10)
ax[-1].set_xlabel('time since GT start [s]')
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=120); print('wrote', sys.argv[1])
