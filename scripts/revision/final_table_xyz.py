#!/usr/bin/env python3
import os
"""Final comparison split into 3D / horizontal (xy) / vertical (z) RMSE [m], same runs and scoring rules as final_table.py
(MILUV / IULUI whole GT span with the window flush; NTU 4-DoF aligned prism, window gt+10 .. end-10)."""
import sys, os, csv, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
src = open('scripts/revision/final_table.py').read()
exec(src.split('cases = [')[0])          # helpers, label maps, constants
NAN3 = np.array([np.nan] * 3)
def split(e): return np.array([np.sqrt(np.mean(np.sum(e ** 2, 1))), np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1))), np.sqrt(np.mean(e[:, 2] ** 2))])
def std_score3(ds, seq, f, lag):
    gt_t, gt = ef.load_gt(ds, seq); P = rows(f)
    th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]
    ff = f.replace('_pose.txt', '_flush_pose.txt')
    if lag > 0 and os.path.exists(ff):
        F = rows(ff); n = len(F) // 2; first = F[:n]; last = F[n:]
        mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
    k = (th >= gt_t[0]) & (th <= gt_t[-1])
    if k.sum() < 10: return NAN3
    return split(X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1))
def ntu_score3(seq, f, lag):
    P = rows(f); th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f'data/ntu/{seq}/leica.csv')))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    if k.sum() < 10: return NAN3
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    if not np.all(np.isfinite(E)): return NAN3
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    return split(E - ((Rz @ G.T).T + t))
def sc3(ds, seq, f, lag):
    if not f or not os.path.exists(f): return NAN3
    try: return ntu_score3(seq, f, lag) if ds == 'ntu' else std_score3(ds, seq, f, lag)
    except Exception: return NAN3
cases = [('miluv', s) for s in MIL] + [('iului', s) for s in ['SquareTurn', 'CircleSpin', 'SquareBob']] + [('ntu', s) for s in ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']]
H = ['KIMS', 'LM', 'pubKIMS', 'EKF-RTS', 'SW-PGO', 'EKF+h', 'EKFuwb']
res = []
for ds, seq in cases:
    if ds == 'miluv': k = {'default_1_remoteControlHighPace_1': 'fx_hp', 'default_1_random3_0': 'fx_r30'}.get(seq, 'fx_' + seq); kf = f'result/miluv/diag/{k}_pose.txt'; lag = 228; geom = 'coplanar' if miluv_constellation(seq) == '1' else 'non-copl'
    elif ds == 'iului': kf = f'result/iului/diag/fx_{seq}_pose.txt'; lag = 199; geom = 'quasi-cop'
    else: kf = f'result/ntu/diag/{"w118_" if seq.startswith("eee") else "n_"}{seq}_pose.txt'; lag = 117; geom = 'coplanar3'
    v = [sc3(ds, seq, kf, lag), sc3(ds, seq, f'result/{ds}/diag/{lm_lab[seq]}_pose.txt', lag) if seq in lm_lab else NAN3]
    if ds == 'iului': v.append(sc3(ds, seq, f'result/iului/{seq}/kims_pose.txt', 15))
    elif ds == 'ntu': v.append(sc3(ds, seq, f'result/ntu/{seq}/kims_pose.txt' if seq.startswith('eee') else f'result/ntu/diag/pub_{seq}_pose.txt', 18))
    else: v.append(NAN3)
    ekf = f'result/{ds}/{seq}/ekf_rts_pose.txt' if ds != 'miluv' else None
    if ds == 'ntu' and not seq.startswith('eee'): ekf = f'result/baselines/ekfrts_{seq}_pose.txt'
    v.append(sc3(ds, seq, ekf, 0) if ekf else NAN3)
    v.append(sc3(ds, seq, f'result/iului/{seq}/sw_pgo_pose.txt' if ds == 'iului' else f'result/baselines/swpgo_{seq}_pose.txt', 0))
    for var in ('h', 'nh'):
        fe = os.path.join(os.environ.get('MILUV_DEVKIT', os.path.expanduser('~/miluv_devkit')), f'results/ekf_{seq}_{var}.csv')
        if ds == 'miluv' and os.path.exists(fe):
            D = pd.read_csv(fe).to_numpy(); v.append(split(D[:, 1:4] - D[:, 4:7]))
        else: v.append(NAN3)
    res.append((ds, seq, geom, np.array(v)))
names = ['3D', 'horizontal xy', 'vertical z']
print("RMSE split [m]: 3D / horizontal (xy) / vertical (z). Same runs and rules as final_table.txt. VINS-Fusion (literature) has no split and is omitted.")
for m in range(3):
    print(f"\n== {names[m]} ==")
    print(f"{'dataset':6s} {'sequence':40s} {'geom':9s} | " + ' '.join(f'{h:>8s}' for h in H))
    agg = {}
    for ds, seq, geom, V in res:
        row = V[:, m]; agg.setdefault((ds, geom), []).append(row)
        print(f"{ds:6s} {seq:40s} {geom:9s} | " + ' '.join(('%8.3f' % x) if np.isfinite(x) else f'{"-":>8s}' for x in row))
    for (ds, geom), rs in agg.items():
        a = np.array(rs); print(f"mean {ds:6s} {geom:9s} (n {len(a):2d})        | " + ' '.join(('%8.3f' % np.nanmean(a[:, j])) if np.isfinite(a[:, j]).any() else f'{"-":>8s}' for j in range(a.shape[1])))
