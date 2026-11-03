#!/usr/bin/env python3
"""X/Y/Z of KIMS (final configuration) vs GT for a list of sequences. usage: fig_xyz_all.py <out.png> <ds:seq:label:T> ...
NTU uses the 4-DoF prism alignment; MILUV/IULUI the mocap frame directly."""
import sys, os, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import eval_flip as ef
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])[:, :8]
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
specs = [s.split(':') for s in sys.argv[2:]]
fig, ax = plt.subplots(len(specs) * 3, 1, figsize=(12, 2.0 * 3 * len(specs)))
for si, (ds, seq, lab, T) in enumerate(specs):
    T = int(T); P = rows(f'result/{ds}/diag/{lab}_pose.txt'); th = P[:-(T-1), 0]; X = P[T-1:, 1:4]; Q = P[T-1:, 4:8]
    if ds == 'ntu':
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'data/ntu/{seq}/leica.csv')))[1:]]); gt_t = L[:, 0]; gt = L[:, 1:4]
        est = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))]); k = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10)
        G = np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); E = est[k]
        gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
        yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc; gt = (Rz @ gt.T).T + t; G = (Rz @ G.T).T + t; gtname = 'Leica GT (4-DoF aligned)'
    else:
        gt_t, gt = ef.load_gt(ds, seq); est = X; k = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10); G = np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); E = est[k]; gtname = 'mocap GT'
    e = E - G; t0 = gt_t[0]
    for j, name in enumerate('xyz'):
        a = ax[3 * si + j]; a.plot(gt_t - t0, gt[:, j], 'k', lw=1.1, label=gtname); a.plot(th - t0, est[:, j], color='tab:red', lw=0.8, label=f'KIMS  {name} rms {np.sqrt(np.mean(e[:, j] ** 2)):.3f} m')
        a.set_ylabel(f'{name} [m]'); a.grid(alpha=0.3); a.legend(fontsize=7, loc='upper right', ncol=2); a.set_xlim(-2, gt_t[-1] - t0 + 2)
        if j == 0: a.set_title(f'{ds} {seq}: 3D rmse {np.sqrt(np.mean(np.sum(e ** 2, 1))):.3f} m (xy {np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1))):.3f}, z {np.sqrt(np.mean(e[:, 2] ** 2)):.3f})', loc='left', fontsize=9)
ax[-1].set_xlabel('time since GT start [s]'); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=100); print('wrote', sys.argv[1])
