#!/usr/bin/env python3
import os
"""Final comparison table over all 27 sequences. 3D rmse [m], window gt+10 s .. end-10 s.
NTU rows: every method converted to the prism with its own attitude and Leica aligned with yaw + translation (4-DoF).
Columns: KIMS (lag T), matched LM, published KIMS, NIS-gated ESKF (lag 0), SW-PGO (Cauchy/Tukey, published window), MILUV devkit EKF (+height), literature VINS-Fusion (NTU, 6-DoF)."""
import sys, os, csv, numpy as np, pandas as pd
sys.path.insert(0, 'scripts'); import eval_flip as ef
from datasets import miluv_constellation
T_BP = np.array([-0.293656, -0.012288, -0.273095])
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])[:, :8]
def Rof(q):
    x, y, z, w = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
FULL_SPAN = True   # MILUV / IULUI: score over the whole GT span; KIMS/LM runs add their first/last window flush
def std_score(ds, seq, f, lag):
    gt_t, gt = ef.load_gt(ds, seq); P = rows(f)
    th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]
    if FULL_SPAN:
        ff = f.replace('_pose.txt', '_flush_pose.txt')
        if lag > 0 and os.path.exists(ff):
            F = rows(ff); n = len(F) // 2; first = F[:n]; last = F[n:]
            t0h, t1h = th[0], th[-1]; mf = first[:, 0] < t0h; ml = last[:, 0] > t1h
            th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
        k = (th >= gt_t[0]) & (th <= gt_t[-1])
    else: k = (th > gt_t[0] + 10) & (th < gt_t[-1] - 10)
    if k.sum() < 10: return np.nan
    e = X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); return np.sqrt(np.mean(np.sum(e ** 2, 1)))
def ntu_score(seq, f, lag):
    P = rows(f); th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'data/ntu/{seq}/leica.csv')))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    if k.sum() < 10: return np.nan
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    if not np.all(np.isfinite(E)): return np.nan
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc; e = E - ((Rz @ G.T).T + t); return np.sqrt(np.mean(np.sum(e ** 2, 1)))
def sc(ds, seq, f, lag):
    if not f or not os.path.exists(f): return np.nan
    try: return ntu_score(seq, f, lag) if ds == 'ntu' else std_score(ds, seq, f, lag)
    except Exception: return np.nan
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
kims_lab = {'default_1_remoteControlHighPace_1': 'sb_hp', 'default_1_random3_1': 'sb_r3', 'default_1_random3_0': 'c0_r3', 'default_1_circular3D_0': 'c0_circ', 'default_1_remoteControlLowPace_0_v2': 'lp_0'}
lm_lab = {**{s: 'lm_' + s for s in ['default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']}, 'default_1_remoteControlHighPace_1': 'lm_hp', 'default_1_random3_1': 'lm_r3', 'SquareTurn': 'lm_SquareTurn', 'CircleSpin': 'lm_CircleSpin', 'SquareBob': 'lm_SquareBob', 'eee_01': 'lm_eee_01', 'eee_02': 'lm_eee_02', 'eee_03': 'lm_eee_03', 'nya_01': 'lmy_nya_01', 'nya_02': 'lmy_nya_02', 'nya_03': 'lmy_nya_03', 'sbs_01': 'lmy_sbs_01', 'sbs_02': 'lmy_s02', 'sbs_03': 'lmy_sbs_03'}
lm_lab.update({'default_1_random3_0': 'lmk_r30', 'obstacles_1_random3_0b': 'lmk_obs'})   # LM reruns with the window flush (same trajectory, whole span)
pub_iului = {'SquareTurn': 0.161, 'CircleSpin': 0.185, 'SquareBob': 0.134}
vins = {'eee_01': 0.608, 'eee_02': 0.506, 'eee_03': 0.494, 'nya_01': 0.397, 'nya_02': 0.424, 'nya_03': 0.787, 'sbs_01': 0.508, 'sbs_02': 0.564, 'sbs_03': 0.878}
cases = [('miluv', s) for s in MIL] + [('iului', s) for s in ['SquareTurn', 'CircleSpin', 'SquareBob']] + [('ntu', s) for s in ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']]
hdr = ['KIMS', 'LM', 'pubKIMS', 'EKF-RTS', 'SW-PGO', 'PS', 'EKF+h', 'EKFuwb', 'VINS-F']
print("Final table (3D rmse, m). MILUV / IULUI: whole GT span, KIMS and LM with the first/last window flush; NTU: window gt+10 s .. end-10 s, 4-DoF aligned, prism. KIMS: final configuration, lag T. LM: same cost, Laplace. pubKIMS: published run. EKF-RTS: repo EKF smoother (published runs; our runs on nya/sbs). SW-PGO: Ceres sliding window (IULUI Tukey 20 nodes, published runs; MILUV/NTU plain 15 nodes). PS: particle smoother (published runs). EKF+h / EKFuwb: MILUV devkit EKF with / without the height sensor. VINS-F: dataset paper, 6-DoF aligned (context).")
print(f"{'dataset':6s} {'sequence':40s} {'geom':9s} | " + ' '.join(f'{h:>8s}' for h in hdr))
agg = {}
for ds, seq in cases:
    if ds == 'miluv': k = 'fx_hp' if seq == 'default_1_remoteControlHighPace_1' else ('fx_r30' if seq == 'default_1_random3_0' else 'fx_' + seq); kf = f'result/miluv/diag/{k}_pose.txt'; lag = 228; geom = 'coplanar' if miluv_constellation(seq) == '1' else 'non-copl'
    elif ds == 'iului': kf = f'result/iului/diag/fx_{seq}_pose.txt'; lag = 199; geom = 'quasi-cop'
    else: kf = f'result/ntu/diag/{"w118_" if seq.startswith("eee") else "n_"}{seq}_pose.txt'; lag = 117; geom = 'coplanar3'
    v = [sc(ds, seq, kf, lag), sc(ds, seq, f'result/{ds}/diag/{lm_lab[seq]}_pose.txt', lag) if seq in lm_lab else np.nan]
    pub = np.nan
    if ds == 'iului': pub = sc(ds, seq, f'result/iului/{seq}/kims_pose.txt', 15)   # published run rescored on the same span
    elif ds == 'ntu': pub = sc(ds, seq, f'result/ntu/{seq}/kims_pose.txt' if seq.startswith('eee') else f'result/ntu/diag/pub_{seq}_pose.txt', 18)
    v.append(pub)
    # EKF-RTS: published results for IULUI / NTU eee, our runs for nya/sbs
    ekf = f'result/{ds}/{seq}/ekf_rts_pose.txt' if ds != 'miluv' else None
    if ds == 'ntu' and not seq.startswith('eee'): ekf = f'result/baselines/ekfrts_{seq}_pose.txt'
    v.append(sc(ds, seq, ekf, 0) if ekf else np.nan)
    # SW-PGO: published results for IULUI (Tukey, 20 nodes), our plain runs (15 nodes) for MILUV / NTU
    swp = f'result/iului/{seq}/sw_pgo_pose.txt' if ds == 'iului' else f'result/baselines/swpgo_{seq}_pose.txt'
    v.append(sc(ds, seq, swp, 0))
    v.append(sc(ds, seq, f'result/{ds}/{seq}/particle_smoothing_pose.txt', 0) if ds != 'miluv' else np.nan)
    ekfh = np.nan; ekfu = np.nan
    for var in ('h', 'nh'):
        fe = os.path.join(os.environ.get('MILUV_DEVKIT', os.path.expanduser('~/miluv_devkit')), f'results/ekf_{seq}_{var}.csv')
        if ds == 'miluv' and os.path.exists(fe):
            D = pd.read_csv(fe).to_numpy(); tt = D[:, 0]; kk = np.ones(len(tt), bool) if FULL_SPAN else ((tt > tt[0] + 10) & (tt < tt[-1] - 10)); val = np.sqrt(np.mean(np.sum((D[kk, 1:4] - D[kk, 4:7]) ** 2, 1)))
            if var == 'h': ekfh = val
            else: ekfu = val
    v.append(ekfh); v.append(ekfu); v.append(vins.get(seq, np.nan))
    agg.setdefault((ds, geom), []).append(v)
    print(f"{ds:6s} {seq[:40]:40s} {geom:9s} | " + ' '.join(('%8.3f' % x) if np.isfinite(x) else f'{"-":>8s}' for x in v))
print()
for (ds, geom), vs in agg.items():
    a = np.array(vs, dtype=float)
    print(f"mean {ds:6s} {geom:9s} (n {len(a):2d}) | " + ' '.join(('%8.3f' % np.nanmean(a[:, j])) if np.isfinite(a[:, j]).any() else f'{"-":>8s}' for j in range(a.shape[1])) + '   (means over the sequences where each method has a value)')
