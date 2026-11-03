#!/usr/bin/env python3
"""Main comparison table v2, IMU + UWB only."""

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
lm_lab = {**{s: 'lm_' + s for s in ['default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']}, 'default_1_remoteControlHighPace_1': 'lm_hp', 'default_1_random3_1': 'lm_r3', 'SquareTurn': 'lm_SquareTurn', 'CircleSpin': 'lm_CircleSpin', 'SquareBob': 'lm_SquareBob', 'eee_01': 'lm_eee_01', 'eee_02': 'lm_eee_02', 'eee_03': 'lm_eee_03', 'nya_01': 'lmy_nya_01', 'nya_02': 'lmy_nya_02', 'nya_03': 'lmy_nya_03', 'sbs_01': 'lmy_sbs_01', 'sbs_02': 'lmy_s02', 'sbs_03': 'lmy_sbs_03'}
lm_lab.update({'default_1_random3_0': 'lmk_r30', 'obstacles_1_random3_0b': 'lmk_obs', 'default_1_random3_2': 'lmf_default_1_random3_2', 'default_1_remoteControlHighPace_2': 'lmf_default_1_remoteControlHighPace_2'})   # LM reruns with the window flush (same trajectory, whole span)

# ---- v2: IMU+UWB-only main table: KIMS, matched LM, SFUISE, dataset reference filter, SW-PGO, particle smoother
cases = [('miluv', s) for s in MIL] + [('ntu', s) for s in ['eee_01', 'eee_02', 'eee_03', 'nya_01', 'nya_02', 'nya_03', 'sbs_01', 'sbs_02', 'sbs_03']]
hdr = ['KIMS', 'LM', 'SFUISE', 'RefFilt', 'SW-PGO', 'PS']
print("Main table v2 (3D rmse, m), IMU + UWB only. MILUV: whole GT span; NTU: gt+10 s .. end-10 s, 4-DoF aligned, prism.")
print("KIMS lag T" + (f" (runs {sys.argv[1]}_*: FP32 rollout + 8 host threads)" if len(sys.argv) > 1 else "") + "; LM same cost; SFUISE published ToA configuration, run alone; RefFilt = MILUV devkit EKF without the height")
print("sensor / NTU our ESKF with fixed-lag RTS over a 2 s sliding window (fresh ranges, 5-sigma gate, std 0.3, still bias); SW-PGO window 15, published design (re-uses the")
print("latest range of every pair); PS fresh-range Gaussian likelihood (sigma 0.3) and still bias. ESKF, SW-PGO, PS start from KIMS's pose.")
print(f"{'dataset':6s} {'sequence':40s} | " + ' '.join(f'{h:>8s}' for h in hdr))
# SFUISE: two configurations, its published ToA one (0.3 m gate) and the same with a 1.5 m gate (KIMS's NTU cut); the column
# uses, per dataset, the one with the lower median 3D rmse (benefit of the doubt for the baseline); both are listed below.
SFV = {}
for ds in ['miluv', 'ntu']:
    seqs = [q for d, q in cases if d == ds]; med = {}
    for var in ['', '_g15']:
        vals = [sc(ds, q, f'result/{ds}/diag/sfuise{var}_{q}_pose.txt', 0) for q in seqs]
        med[var] = np.nanmedian(vals) if np.isfinite(vals).any() else np.inf
    SFV[ds] = min(med, key=med.get); SFV[ds + '_med'] = med
KL = sys.argv[1] if len(sys.argv) > 1 else ''   # KIMS run label prefix; default: the published FP64 runs
agg = {}
for ds, seq in cases:
    if ds == 'miluv': k = 'fx_hp' if seq == 'default_1_remoteControlHighPace_1' else ('fx_r30' if seq == 'default_1_random3_0' else 'fx_' + seq); kf = f'result/miluv/diag/{k}_pose.txt'; lag = 228
    else: kf = f'result/ntu/diag/{"w118_" if seq.startswith("eee") else "n_"}{seq}_pose.txt'; lag = 117
    if KL: kf = f'result/{ds}/diag/{KL}_{seq}_pose.txt'   # e.g. f32a: FP32 rollout + 8 host threads
    v = [sc(ds, seq, kf, lag), sc(ds, seq, f'result/{ds}/diag/{lm_lab[seq]}_pose.txt', lag) if seq in lm_lab else np.nan,
         sc(ds, seq, f'result/{ds}/diag/sfuise{SFV[ds]}_{seq}_pose.txt', 0)]
    if ds == 'miluv':
        fe = os.path.join(os.environ.get('MILUV_DEVKIT', os.path.expanduser('~/miluv_devkit')), f'results/ekf_{seq}_nh.csv'); ref = np.nan
        if os.path.exists(fe):
            D = pd.read_csv(fe).to_numpy(); ref = np.sqrt(np.mean(np.sum((D[:, 1:4] - D[:, 4:7]) ** 2, 1)))
        v.append(ref)
    else: v.append(sc(ds, seq, f'result/baselines/ekfl_{seq}_pose.txt', 0))   # ESKF fixed-lag RTS (2 s, real time): fresh ranges, 5-sigma gate, gyro attitude noise, still bias, std 0.3
    v.append(sc(ds, seq, f'result/baselines/swpgoi_{seq}_pose.txt', 0)); v.append(sc(ds, seq, f'result/baselines/psb_{seq}_pose.txt', 0))
    agg.setdefault(ds, []).append(v)
    print(f"{ds:6s} {seq[:40]:40s} | " + ' '.join(('%8.3f' % x) if np.isfinite(x) and x < 1e4 else (f'{">1e4":>8s}' if np.isfinite(x) else f'{"-":>8s}') for x in v))
print()
for ds, vs in agg.items():
    a = np.array(vs, dtype=float)
    med = ' '.join(('%8.3f' % np.nanmedian(a[:, j])) if np.isfinite(a[:, j]).any() else f'{"-":>8s}' for j in range(a.shape[1]))
    best = [int(np.nansum(a[:, 0] < a[:, j])) for j in range(1, a.shape[1])]
    print(f"median {ds:6s} (n {len(a):2d}) | {med}")
    print(f"       KIMS lower than " + ', '.join(f'{h} in {b}/{len(a)}' for h, b in zip(hdr[1:], best)) + f"; >1 m: " + ', '.join(f'{h} {int(np.nansum(a[:, j] > 1))}' for j, h in enumerate(hdr)))
print()
for ds in ['miluv', 'ntu']:
    m = SFV[ds + '_med']; print(f"SFUISE {ds}: median 3D published config {m['']:.3f}, 1.5 m gate {m['_g15']:.3f} -> column uses {'1.5 m gate' if SFV[ds] == '_g15' else 'published config'}")
