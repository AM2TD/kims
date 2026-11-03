#!/usr/bin/env python3
"""Ablation table: 3D rmse of the final configuration with one component removed at a time."""
import sys, os, csv, numpy as np
sys.path.insert(0, 'scripts'); import eval_flip as ef
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def score_std(ds, seq, f, T):
    gt_t, gt = ef.load_gt(ds, seq); P = rows(f); th = P[:-(T-1), 0]; X = P[T-1:, 1:4]; k = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10)
    e = X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); return np.sqrt(np.mean(np.sum(e ** 2, 1)))
def score_ntu(seq, f, lag=117):
    P = rows(f); th = P[:-lag, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]; prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'data/ntu/{seq}/leica.csv')))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10); G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc; e = E - ((Rz @ G.T).T + t); return np.sqrt(np.mean(np.sum(e ** 2, 1)))
cols = [('full', None), ('-tempering', 'notemp'), ('-prev.posterior', 'noprop'), ('-arrival cov', 'noarr'), ('-head filter', 'hf'), ('-mixture', 'tukey'), ('-still bias', 'nostill'), ('LM instead', 'lm')]
seqs = [('miluv', 'default_1_remoteControlHighPace_1', 'hp1', {'full': 'sb_hp', 'hf': 'nohf_hp', 'lm': 'lm_hp'}),
        ('miluv', 'default_1_remoteControlHighPace_0', 'hp0', {'full': 'new_default_1_remoteControlHighPace_0'}),
        ('iului', 'SquareTurn', 'st', {'full': 'sb_SquareTurn', 'hf': 'nohf_SquareTurn', 'lm': 'lm_SquareTurn'}),
        ('ntu', 'eee_01', 'eee01', {'full': 'w118_eee_01', 'lm': 'lm_eee_01'}),
        ('ntu', 'sbs_01', 'sbs01', {'full': 'n_sbs_01'})]
print("Ablation: one component removed at a time (3D rmse, m). NTU head filter is off in the full configuration, so its column shows '+head filter'. '-' = not applicable / not run.")
print(f"{'sequence':14s} | " + ' '.join(f'{c[0]:>15s}' for c in cols))
for ds, seq, short, special in seqs:
    out = []
    for name, tag in cols:
        lab = special.get(tag or 'full') or (f'ab_{tag}_{short}' if tag else None)
        if ds == 'ntu' and name == '-head filter': lab = f'ab_hf_{short}'
        f = f'result/{ds}/diag/{lab}_pose.txt' if lab else None
        if not f or not os.path.exists(f): out.append('-'); continue
        try: v = score_ntu(seq, f) if ds == 'ntu' else score_std(ds, seq, f, 229 if ds == 'miluv' else 200); out.append(f'{v:.3f}')
        except Exception as ex: out.append('err')
    print(f"{short:14s} | " + ' '.join(f'{o:>15s}' for o in out))
