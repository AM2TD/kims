#!/usr/bin/env python3
"""(dp_z, dv_z) landscape of one solve: negative log target on a grid, the posterior samples, LM path and start gradient.
usage: fig_landscape.py <out.png> <dump files...> (SquareTurn, T 200)"""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import eval_flip as ef
out, files = sys.argv[1], sys.argv[2:]
T = 200; gt_t, gt = ef.load_gt('iului', 'SquareTurn'); c, n = ef.anchor_plane('iului', 'SquareTurn')
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
P = rows(os.environ.get('KIMS_DIAG_POSE', 'result/iului/diag/sb_SquareTurn_pose.txt'))
fig, ax = plt.subplots(2, len(files), figsize=(5.2 * len(files), 9.5), squeeze=False)
for col, f in enumerate(files):
    hdr = open(f).readline().split(); H = dict(zip(hdr[1::2], hdr[2::2])); x0z, x0vz = float(H['x0z']), float(H['x0vz']); ess = float(H['ess'])
    nsolve = int(f.rsplit('_', 1)[1].split('.')[0]); th = P[nsolve - T, 0]
    S, G = [], {0: [], 1: []}; L = []; start = pmean = grad = None
    for l in open(f):
        t = l.split()
        if t[0] == 'S': S.append([float(x) for x in t[1:]])
        elif t[0] in ('G0', 'G1'): G[int(t[0][1])].append([float(x) for x in t[1:]])
        elif t[0] == 'L': L.append([float(x) for x in t[1:]])
        elif t[0] == 'START': start = np.array([float(x) for x in t[1:]])
        elif t[0] == 'PMEAN': pmean = np.array([float(x) for x in t[1:]])
        elif t[0] == 'GRAD': grad = np.array([float(x) for x in t[1:]])
    S = np.array(S); L = np.array(L)
    tz = np.interp(th, gt_t, gt[:, 2]) - x0z; tvz = (np.interp(th + 0.25, gt_t, gt[:, 2]) - np.interp(th - 0.25, gt_t, gt[:, 2])) / 0.5 - x0vz
    for r, which in enumerate((0, 1)):
        a = ax[r, col]; g = np.array(G[which]); z = g[:, 0]; v = g[:, 1]; J = g[:, 2]
        NG = int(np.sqrt(len(g))); Z = z.reshape(NG, NG); Vv = v.reshape(NG, NG); JJ = J.reshape(NG, NG); JJ = JJ - np.nanmin(JJ)
        cs = a.contourf(Z, Vv, np.minimum(JJ, 60), levels=np.linspace(0, 60, 31), cmap='viridis_r'); a.contour(Z, Vv, JJ, levels=[1, 2, 5, 10, 20, 40], colors='w', linewidths=0.4)
        base = start if which == 0 else pmean
        w = np.exp(S[:, 6] - S[:, 6].max()); keep = w > 1e-3
        a.scatter(S[keep, 2], S[keep, 5], s=6 * w[keep] ** 0.5 + 1, c='tab:red', alpha=0.35, lw=0, label=f'posterior samples (ESS {ess:.0f})')
        a.plot(L[:, 2], L[:, 5], 'o-', color='tab:blue', ms=4, lw=1.2, label='LM iterates from the start')
        if grad is not None:
            d = -grad[[2, 5]]; d = 0.25 * d / (np.linalg.norm(d) + 1e-12)
            a.annotate('', xy=(start[2] + d[0], start[5] + d[1]), xytext=(start[2], start[5]), arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))
            a.plot([], [], color='cyan', lw=1.5, label='-gradient at the start')
        a.plot(start[2], start[5], 'ws', ms=7, mec='k', label='start (carried mean)')
        a.plot(pmean[2], pmean[5], 'r*', ms=12, mec='k', label='sampling posterior mean')
        a.plot(tz, tvz, 'k+', ms=14, mew=2, label='true offset (GT)')
        a.axvline(c[2] - x0z, color='w', ls='--', lw=0.8); a.text(c[2] - x0z + 0.02, -0.95, 'anchor plane', color='w', fontsize=8)
        a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_xlabel('head offset dp_z [m]'); a.set_ylabel('head offset dv_z [m/s]')
        a.set_title(f'solve {nsolve}, head t = gt+{th - gt_t[0]:.1f} s, x0 z {x0z:.2f} (GT {x0z + tz:.2f})\nslice through the {"start" if which == 0 else "posterior mean"}', fontsize=9)
        if r == 0 and col == 0: a.legend(fontsize=7, loc='upper left', framealpha=0.85)
fig.colorbar(cs, ax=ax.ravel().tolist(), shrink=0.6, label='J - min J (nats)')
fig.savefig(out, dpi=120); print('wrote', out)
