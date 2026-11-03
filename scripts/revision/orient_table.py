#!/usr/bin/env python3
"""Orientation RMSE (deg) against the mocap attitude for MILUV and IULUI (NTU has no attitude GT). Window gt+10 s .. end-10 s.
raw: angle of R_gt^T R_est. offset-removed: a constant body-frame rotation (mounting / frame convention) fitted per run and
removed. Also yaw-only error (heading) after offset removal."""
import sys, os, numpy as np
sys.path.insert(0, 'scripts'); from datasets import CONFIG
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rows(f): return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8])[:, :8]
def Rof(q):
    x, y, z, w = q; n = np.sqrt(x*x+y*y+z*z+w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def load_gt_q(ds, seq):
    d = os.path.join(PKG, CONFIG[ds]['data_dir'].format(seq=seq)); g = np.loadtxt(os.path.join(d, 'mocap.csv'), delimiter=',', skiprows=1); return g[:, 0], g[:, 4:8]
def slerp_nearest(t, gt_t, gq):
    i = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1); return gq[i]
def angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
def karcher(Rs):
    M = np.mean(Rs, 0); U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R
def score(ds, seq, f, lag):
    if not f or not os.path.exists(f): return (np.nan,) * 3
    gt_t, gq = load_gt_q(ds, seq); P = rows(f); th = P[:-lag, 0] if lag > 0 else P[:, 0]; Q = P[lag:, 4:8]
    k = np.where((th > gt_t[0] + 10) & (th < gt_t[-1] - 10))[0][::5]
    if len(k) < 10: return (np.nan,) * 3
    D = [Rof(slerp_nearest(th[i], gt_t, gq)).T @ Rof(Q[i]) for i in k]   # R_gt^T R_est (body-frame offset)
    if not all(np.all(np.isfinite(d)) for d in D): return (np.nan,) * 3
    raw = np.sqrt(np.mean([angle(d) ** 2 for d in D]))
    Rc = karcher(np.array(D)); Dc = [Rc.T @ d for d in D]
    corr = np.sqrt(np.mean([angle(d) ** 2 for d in Dc]))
    yaw = np.sqrt(np.mean([np.degrees(np.arctan2(d[1, 0], d[0, 0])) ** 2 for d in Dc]))
    return raw, corr, yaw
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
kims = {'default_1_remoteControlHighPace_1': 'sb_hp', 'default_1_random3_1': 'sb_r3', 'default_1_random3_0': 'c0_r3', 'default_1_circular3D_0': 'c0_circ', 'default_1_remoteControlLowPace_0_v2': 'lp_0'}
lm = {'default_1_remoteControlHighPace_1': 'lm_hp', 'default_1_random3_1': 'lm_r3', 'SquareTurn': 'lm_SquareTurn', 'CircleSpin': 'lm_CircleSpin', 'SquareBob': 'lm_SquareBob'}
print("Orientation error (deg): raw angle rmse / after removing a constant body-frame offset / yaw-only after offset removal. KIMS lag T, LM lag T, baselines lag 0.")
print(f"{'dataset':6s} {'sequence':38s} | {'KIMS':>16s} | {'LM':>16s} | {'pubKIMS':>16s} | {'EKF-RTS':>16s} | {'SW-PGO':>16s} | {'PS':>16s}")
def fmt(v): return ('%4.1f/%4.1f/%4.1f' % v) if np.isfinite(v[0]) else f'{"-":>14s}'
for ds, seqs, T in [('miluv', MIL, 228), ('iului', ['SquareTurn', 'CircleSpin', 'SquareBob'], 199)]:
    for seq in seqs:
        kf = f'result/{ds}/diag/{kims.get(seq, "new_" + seq) if ds == "miluv" else "sb_" + seq}_pose.txt'
        lmf = f'result/{ds}/diag/{lm[seq]}_pose.txt' if seq in lm else (f'result/miluv/diag/lm_{seq}_pose.txt' if ds == 'miluv' else None)
        pub = f'result/iului/{seq}/kims_pose.txt' if ds == 'iului' else None
        ekf = f'result/iului/{seq}/ekf_rts_pose.txt' if ds == 'iului' else None
        swp = f'result/iului/{seq}/sw_pgo_pose.txt' if ds == 'iului' else f'result/baselines/swpgo_{seq}_pose.txt'
        ps = None   # the particle smoother writes an identity quaternion (no attitude estimate)
        vals = [score(ds, seq, kf, T), score(ds, seq, lmf, T), score(ds, seq, pub, 15), score(ds, seq, ekf, 0), score(ds, seq, swp, 0), score(ds, seq, ps, 0)]
        print(f"{ds:6s} {seq[:38]:38s} | " + ' | '.join(f'{fmt(v):>16s}' for v in vals))
