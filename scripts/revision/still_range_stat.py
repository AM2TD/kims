#!/usr/bin/env python3
"""GT-free range-error statistic from the rest phase: fit the still-phase mean ranges with the tag array
(position + yaw, gravity-aligned attitude from the accelerometer) and report the per-pair residual spread against the
std column. A spread well above the std column means pair-persistent biases dominate -> head filter off."""
import sys, os, json, csv, numpy as np
sys.path.insert(0, 'scripts'); from datasets import CONFIG, anchor_path
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def load(ds, seq):
    cfg = CONFIG[ds]; d = os.path.join(PKG, cfg['data_dir'].format(seq=seq))
    if ds == 'iului':
        meta = json.load(open(os.path.join(PKG, cfg['meta'].format(seq=seq)))); A = np.array(meta['anchors']); aid = list(range(len(A)))
        tags = {int(meta['tag0_id']): np.array(meta['tag0_offset']), int(meta['tag1_id']): np.array(meta['tag1_offset'])}
        U = np.loadtxt(os.path.join(d, 'uwb_range.csv'), delimiter=',', skiprows=1); U = U[:, [0, 1, 2, 3, 4]]   # t tag anchor range std
    else:
        A = np.loadtxt(os.path.join(PKG, anchor_path(ds, seq))); aid = list(cfg['anchor_ids'])
        tags = {int(t): np.array(o) for t, o in zip(cfg['tag_ids'], cfg['tag_offsets'])}
        rows = [r for r in csv.reader(open(os.path.join(d, 'uwb_range.csv')))][1:]
        hdr = next(csv.reader(open(os.path.join(d, 'uwb_range.csv'))))
        if ds == 'ntu': U = np.array([[float(r[0]), float(r[2]), float(r[3]), float(r[1]), float(r[4])] for r in rows])   # timestamp,range,from,to,std
        else:
            ci = {k: hdr.index(k) for k in hdr}
            it = ci.get('from_id', 1); ia = ci.get('to_id', 2); ir = ci.get('range', 3); isd = ci.get('std', None); its = ci.get('timestamp', 0)
            U = np.array([[float(r[its]), float(r[it]), float(r[ia]), float(r[ir]), float(r[isd]) if isd is not None else 0.069] for r in rows])
    imu = np.loadtxt(os.path.join(d, 'imu_px4.csv' if ds == 'miluv' else 'imu.csv'), delimiter=',', skiprows=1)
    return A, aid, tags, U, imu
def Rz(y): c, s = np.cos(y), np.sin(y); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
for ds, seq in [('miluv', 'default_1_remoteControlHighPace_1'), ('miluv', 'default_1_random3_1'), ('iului', 'SquareBob'), ('iului', 'SquareTurn'), ('iului', 'CircleSpin'), ('ntu', 'eee_01'), ('ntu', 'eee_02'), ('ntu', 'eee_03')]:
    A, aid, tags, U, imu = load(ds, seq)
    t0 = U[0, 0]; still = U[(U[:, 0] < t0 + 3.0) & (U[:, 3] > 0)]
    # gravity alignment from the first 3 s of IMU
    am = imu[imu[:, 0] < imu[0, 0] + 3.0][:, 4:7].mean(0)
    zb = am / np.linalg.norm(am); zdown = zb[2] < 0
    if zdown: zb = -zb
    v = np.cross(zb, [0, 0, 1]); sn = np.linalg.norm(v); cs = zb[2]
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); Rt = np.eye(3) + K + K @ K * ((1 - cs) / (sn * sn)) if sn > 1e-9 else np.eye(3)
    if zdown: Rt = Rt @ np.diag([1, -1, -1])
    pairs = {}
    for t, tid, a, r, sd in still:
        if int(tid) in tags and int(a) in aid: pairs.setdefault((int(tid), int(a)), []).append(r)
    keys = sorted(pairs); rm = np.array([np.mean(pairs[k]) for k in keys]); sdc = np.median(still[:, 4])
    best = None
    for yaw in np.radians(np.arange(0, 360, 2.0)):
        R = Rz(yaw) @ Rt
        # linear LS in position given yaw: iterate Gauss-Newton from the anchor centroid
        p = A.mean(0) + np.array([0, 0, -1.0])
        for it in range(20):
            res = []; J = []
            for (tid, a), r in zip(keys, rm):
                tw = p + R @ tags[tid]; d = tw - A[aid.index(a)]; dist = np.linalg.norm(d)
                res.append(r - dist); J.append(d / dist)
            res = np.array(res); J = np.array(J); dp = np.linalg.lstsq(J, res, rcond=None)[0]; p = p + dp
            if np.linalg.norm(dp) < 1e-6: break
        rms = np.sqrt(np.mean(res ** 2))
        if best is None or rms < best[0]: best = (rms, yaw, p, res.copy())
    rms, yaw, p, res = best
    print(f"{ds:6s} {seq[:30]:30s} | pairs {len(keys):2d} | std column {sdc:.3f} | still-fit residual rms {rms:.3f} (ratio {rms/sdc:4.1f}) | per-pair: " + ' '.join(f'{x:+.2f}' for x in res))
