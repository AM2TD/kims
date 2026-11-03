#!/usr/bin/env python3
"""Posterior strips across the SquareTurn plane crossing: per-solve weighted marginal of the head z (sampling run) and the
Laplace cloud (LM run), with both runs' heads, GT and the anchor plane. usage: fig_strip.py <out.png> [dataset seq pose_file strip_samp strip_lm|- tmin tmax]"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import eval_flip as ef
ds, seq = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('iului', 'SquareTurn')
pose = sys.argv[4] if len(sys.argv) > 4 else 'result/iului/diag/sb_SquareTurn_pose.txt'
fs = sys.argv[5] if len(sys.argv) > 5 else 'result/runlogs/strip_st_samp.txt'
fl = sys.argv[6] if len(sys.argv) > 6 else 'result/runlogs/strip_st_lm.txt'
tmin, tmax = (float(sys.argv[7]), float(sys.argv[8])) if len(sys.argv) > 8 else (43, 66)
T = 229 if ds == 'miluv' else (118 if ds == 'ntu' else 200); gt_t, gt = ef.load_gt(ds, seq); c, n = ef.anchor_plane(ds, seq); t0 = gt_t[0]
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
P = rows(pose)
def load(f):
    E = []; Z = []; cur = None
    for l in open(f):
        t = l.split()
        if t[0] == 'E':
            cur = [float(x) for x in t[1:]]; E.append(cur); Z.append([])
        else: Z[-1].append([float(x) for x in t[1:]])
    return np.array(E), [np.array(z) for z in Z]
Es, Zs = load(fs); El, Zl = load(fl) if fl != '-' else (Es, Zs)
zb = np.arange(min(gt[:, 2].min(), 2 * c[2] - gt[:, 2].max()) - 0.3, max(gt[:, 2].max(), 2 * c[2] - gt[:, 2].min()) + 0.3, 0.02)
def image(E, Z):
    img = np.zeros((len(zb) - 1, len(E))); tt = np.zeros(len(E))
    for k, (e, z) in enumerate(zip(E, Z)):
        nidx = int(e[0]); tt[k] = P[nidx - T, 0] - t0
        w = z[:, 2]; h, _ = np.histogram(e[1] + z[:, 0], bins=zb, weights=w); img[:, k] = h / max(h.max(), 1e-12)
    return tt, img
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1.4]))
for a, (E, Z, name, col) in zip(ax[:2], [(Es, Zs, 'sampling run: weighted posterior of the head z (per solve)', 'tab:red'), (El, Zl, 'LM run: Laplace cloud N(mode, H^-1) of the head z (per solve)', 'tab:blue')]):
    tt, img = image(E, Z)
    a.pcolormesh(tt, 0.5 * (zb[1:] + zb[:-1]), img, cmap='Greys', vmin=0, vmax=1, shading='nearest')
    a.plot(gt_t - t0, gt[:, 2], 'k', lw=1.3, label='GT')
    ts, _ = image(Es, Zs); tl, _ = image(El, Zl)
    a.plot(ts, Es[:, 4], color='tab:red', lw=1.2, label='sampling run head')
    a.plot(tl, El[:, 4], color='tab:blue', lw=1.2, label='LM run head')
    a.axhline(c[2], color='tab:green', ls='--', lw=0.9, label='anchor plane')
    a.set_ylim(zb[0], zb[-1]); a.set_ylabel('z [m]'); a.set_title(name, loc='left', fontsize=10); a.legend(fontsize=8, loc='upper left', ncol=4)
# fraction of posterior mass below the plane
a = ax[2]
for E, Z, name, col in [(Es, Zs, 'sampling', 'tab:red'), (El, Zl, 'LM (Laplace)', 'tab:blue')]:
    tt, _ = image(E, Z); fr = np.array([np.sum(z[:, 2] * ((e[1] + z[:, 0]) < c[2])) / np.sum(z[:, 2]) for e, z in zip(E, Z)])
    a.plot(tt, fr, color=col, lw=1.2, label=name)
a.plot(gt_t - t0, (gt[:, 2] < c[2]).astype(float), 'k', lw=1.0, label='GT below the plane')
a.set_ylim(-0.05, 1.05); a.set_ylabel('P(z < plane)'); a.set_xlabel('time since GT start [s]'); a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
ax[0].set_xlim(tmin, tmax)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=130); print('wrote', sys.argv[1])
