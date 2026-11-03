#!/usr/bin/env python3
"""Seed-repetition table of the final configuration: per sequence mean, sd, min, max of the 3D RMSE and
the wrong-side fraction over seeds 1234, 1, 2, 3, 4."""
import sys, os, numpy as np
sys.path.insert(0, 'scripts'); import eval_flip as ef
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])
cases = [('miluv', 'default_1_remoteControlHighPace_1', 229, ['sb_hp', 'sbs1_hp', 'sbs2_hp', 's3_hp', 's4_hp']),
         ('miluv', 'default_1_random3_1', 229, ['sb_r3', 's1_r3', 's2_r3', 's3_r3', 's4_r3']),
         ('iului', 'SquareTurn', 200, ['sb_SquareTurn', 'sbs1_SquareTurn', 'sbs2_SquareTurn', 's3_SquareTurn', 's4_SquareTurn']),
         ('iului', 'CircleSpin', 200, ['sb_CircleSpin'] + [f's{s}_CircleSpin' for s in (1, 2, 3, 4)]),
         ('iului', 'SquareBob', 200, ['sb_SquareBob'] + [f's{s}_SquareBob' for s in (1, 2, 3, 4)]),
         ('ntu', 'eee_01', 118, ['w118_eee_01', 'nf3s1_eee_01', 'nf3s2_eee_01', 's3_eee_01', 's4_eee_01']),
         ('ntu', 'eee_02', 118, ['w118_eee_02'] + [f's{s}_eee_02' for s in (1, 2, 3, 4)]),
         ('ntu', 'eee_03', 118, ['w118_eee_03'] + [f's{s}_eee_03' for s in (1, 2, 3, 4)])]
print("Final configuration, seeds 1234/1/2/3/4; same scorer (gt+10 s .. end-10 s, exact lag). rmse: mean +- sd [min, max]; wrong: mean [min, max]")
print(f"{'dataset':6s} {'sequence':12s} {'n':>2s} | {'rmse mean':>9s} {'sd':>6s} {'min':>6s} {'max':>6s} | {'z mean':>6s} | {'wrong mean':>10s} {'min':>6s} {'max':>6s}")
for ds, seq, T, labs in cases:
    gt_t, gt = ef.load_gt(ds, seq); c, n = ef.anchor_plane(ds, seq); R = []; W = []; Z = []
    for lab in labs:
        f = f'result/{ds}/diag/{lab}_pose.txt'
        if not os.path.exists(f): continue
        P = rows(f)
        if len(P) < T + 100: continue
        th = P[:-(T - 1), 0]; X = P[T - 1:, 1:4]; k = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10)
        g = np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); e = X[k] - g; hz = (X[k] - c) @ n; hg = (g - c) @ n; w = np.abs(hg) > 0.1
        R.append(np.sqrt(np.mean(np.sum(e ** 2, 1)))); Z.append(np.sqrt(np.mean(e[:, 2] ** 2))); W.append(np.mean(np.sign(hz[w]) != np.sign(hg[w])))
    R = np.array(R); W = np.array(W); Z = np.array(Z)
    if len(R) == 0: print(ds, seq, 'none'); continue
    print(f"{ds:6s} {seq[:12]:12s} {len(R):2d} | {R.mean():9.3f} {R.std(ddof=1) if len(R) > 1 else 0:6.3f} {R.min():6.3f} {R.max():6.3f} | {Z.mean():6.3f} | {W.mean():10.3f} {W.min():6.3f} {W.max():6.3f}")
