"""Representability bound T_max for the current basis (window-constant + 3 cosines per channel): on the calibration
flights, the accelerometer error (IMU specific force minus the mocap-derived one, body frame) is projected on the
basis over windows of length T; the remainder is double-integrated and its window-end displacement compared with the
effective LOS range sd (0.081 m). Reports the median window-end displacement per T."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
G = np.array([0, 0, 9.81]); SE = 0.081
res = {}
for seq in ["default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2"]:
    d = f"data/miluv/{seq}/ifo001"
    M = np.loadtxt(f"{d}/mocap.csv", delimiter=",", skiprows=1); mt, mp, mq = M[:, 0], M[:, 1:4], M[:, 4:8]; mq /= np.linalg.norm(mq, axis=1, keepdims=True)
    I = np.loadtxt(f"{d}/imu_px4.csv", delimiter=",", skiprows=1); it, acc = I[:, 0], I[:, 4:7]
    dtm = np.median(np.diff(mt)); w = int(0.3 / dtm) | 1
    a_w = savgol_filter(mp, w, 3, deriv=2, delta=dtm, axis=0)                       # world acceleration from mocap
    k = (it > mt[0] + 1) & (it < mt[-1] - 1); ti, ai = it[k], acc[k]
    q = np.stack([np.interp(ti, mt, mq[:, c]) for c in range(4)], 1); q /= np.linalg.norm(q, axis=1, keepdims=True); R = Rot.from_quat(q)
    f_true = R.inv().apply(np.stack([np.interp(ti, mt, a_w[:, c]) for c in range(3)], 1) + G)   # true specific force, body
    e = ai - f_true                                                                  # accelerometer error (body)
    fly = np.interp(ti, mt, mp[:, 2]) > 0.4
    for T in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        out = []
        for t0 in np.arange(ti[0] + 5, ti[-1] - T - 1, T / 2):
            j0, j1 = np.searchsorted(ti, t0), np.searchsorted(ti, t0 + T)
            if not (fly[j0] and fly[j1 - 1]) or j1 - j0 < 50: continue
            tt = ti[j0:j1] - t0; Ts = tt[-1]
            Phi = np.stack([np.ones_like(tt)] + [np.cos(np.pi * m * (tt + 0.5 * np.median(np.diff(tt))) / Ts) for m in (1, 2, 3)], 1)
            ew = R[j0:j1].apply(e[j0:j1])                                            # error in world frame
            resid = ew - Phi @ np.linalg.lstsq(Phi, ew, rcond=None)[0]              # remainder after the basis (per world axis)
            h = np.diff(ti[j0:j1 + 1]) if j1 < len(ti) else np.diff(np.append(ti[j0:j1], ti[j1 - 1] + np.median(np.diff(tt))))
            v = np.cumsum(resid * h[:, None], 0); p = np.cumsum(v * h[:, None], 0)
            out.append(np.linalg.norm(p[-1]))
        res.setdefault(T, []).append(np.median(out))
print("window-end displacement of the IMU error not representable by the basis (median over windows, m; threshold sigma_eff = 0.081):")
for T, v in res.items(): print(f"  T = {T:.0f} s: " + " / ".join(f"{x:.3f}" for x in v) + f"   (mean {np.mean(v):.3f})")
