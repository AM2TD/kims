#!/usr/bin/env python3
import sys, os, csv, numpy as np
sys.path.insert(0, 'scripts'); from datasets import CONFIG
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seq = sys.argv[1]; cfg = CONFIG['ntu']; d = os.path.join(PKG, cfg['data_dir'].format(seq=seq))
A = np.loadtxt(sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, cfg['anchor_file'])); aid = list(cfg['anchor_ids'])
tags = {int(t): np.array(o) for t, o in zip(cfg['tag_ids'], cfg['tag_offsets'])}
rows = [r for r in csv.reader(open(os.path.join(d, 'uwb_range.csv')))][1:]
U = np.array([[float(r[0]), float(r[2]), float(r[3]), float(r[1]), float(r[4])] for r in rows])
imu = np.loadtxt(os.path.join(d, 'imu.csv'), delimiter=',', skiprows=1)
t0 = U[0, 0]; still = U[(U[:, 0] < t0 + 3.0) & (U[:, 3] > 0)]
am = imu[imu[:, 0] < imu[0, 0] + 3.0][:, 4:7].mean(0); zb = am / np.linalg.norm(am); zdown = zb[2] < 0
if zdown: zb = -zb
v = np.cross(zb, [0, 0, 1]); sn = np.linalg.norm(v); cs = zb[2]
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); Rt = np.eye(3) + K + K @ K * ((1 - cs) / (sn * sn)) if sn > 1e-9 else np.eye(3)
if zdown: Rt = Rt @ np.diag([1, -1, -1])
pairs = {}
for t, tid, a, r, sd in still:
    if int(tid) in tags and int(a) in aid: pairs.setdefault((int(tid), int(a)), []).append(r)
keys = sorted(pairs); rm = np.array([np.mean(pairs[k]) for k in keys])
def Rz(y): c, s = np.cos(y), np.sin(y); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
best = None
for yaw in np.radians(np.arange(0, 360, 1.0)):
    R = Rz(yaw) @ Rt; p = A.mean(0) + np.array([0, 0, -1.0])
    for it in range(30):
        res = []; J = []
        for (tid, a), r in zip(keys, rm):
            tw = p + R @ tags[tid]; dd = tw - A[aid.index(a)]; dist = np.linalg.norm(dd); res.append(r - dist); J.append(dd / dist)
        res = np.array(res); J = np.array(J); dp = np.linalg.lstsq(J, res, rcond=None)[0]; p = p + dp
        if np.linalg.norm(dp) < 1e-7: break
    rms = np.sqrt(np.mean(res ** 2))
    if best is None or rms < best[0]: best = (rms, yaw, p.copy())
rms, yaw, p = best
if p[2] > A[:, 2].mean(): p[2] = 2 * A[:, 2].mean() - p[2]
print(f"init_x:={p[0]:.3f} init_y:={p[1]:.3f} init_z:={p[2]:.3f} init_yaw:={np.degrees(yaw):.1f}   # still-fit residual rms {rms:.3f} m, {len(keys)} pairs")
