#!/usr/bin/env python3
"""X/Y/Z of the three constellation-0 sequences: GT, final run (mixture fitted on these flights) and holdout run
(mixture fitted on the other ten MILUV sequences). Output result/figs/mix_holdout_xyz.png."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import eval_flip as ef
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])[:, :8]
LAG = 228
S = [('default_1_random3_0', 'fx_r30', 'random3_0'), ('default_1_circular3D_0', 'fx_default_1_circular3D_0', 'circular3D_0'),
     ('default_1_remoteControlLowPace_0_v2', 'fx_default_1_remoteControlLowPace_0_v2', 'LowPace_0_v2')]
fig, ax = plt.subplots(3, 3, figsize=(15, 8.5), sharex='col')
for c, (seq, lab, name) in enumerate(S):
    gt_t, gt = ef.load_gt('miluv', seq)
    for r, k in enumerate('xyz'):
        a = ax[r, c]; a.plot(gt_t, gt[:, r], 'k-', lw=1.6, label='GT')
        for f, col, lb in [(f'result/miluv/diag/{lab}_pose.txt', '#d62728', 'final (mixture from these flights)'),
                           (f'result/miluv/diag/cv_{seq}_pose.txt', '#1f77b4', 'holdout (mixture from the other 10)')]:
            P = rows(f); a.plot(P[:-LAG, 0], P[LAG:, 1 + r], color=col, lw=0.9, label=lb, alpha=0.85)
        a.set_ylabel(f'{k} [m]'); a.grid(alpha=0.3)
        if r == 0: a.set_title(name)
        if r == 2: a.set_xlabel('t [s]')
ax[0, 0].legend(fontsize=8, loc='best')
fig.suptitle('MILUV constellation 0: range mixture fitted on the evaluated flights vs on the other ten (3D RMSE 0.197/0.197, 0.153/0.157, 0.104/0.105 m)')
fig.tight_layout(); fig.savefig('result/figs/mix_holdout_xyz.png', dpi=110); print('saved')
