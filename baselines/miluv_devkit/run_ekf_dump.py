from __future__ import annotations
import sys, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation
from miluv.data import DataLoader
import miluv.utils as utils
import examples.ekfutils.imu_one_robot_models as model
import examples.ekfutils.common as common
exp_name, out = sys.argv[1], sys.argv[2]; use_height = not (len(sys.argv) > 3 and sys.argv[3] == 'noheight')
miluv = DataLoader(exp_name, imu="px4", cam=None, mag=False); data = miluv.data["ifo001"]
qt = np.append(data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()); qt = np.sort(np.unique(qt))
imu_q = miluv.query_by_timestamps(qt, robots="ifo001", sensors="imu_px4")["ifo001"]
accel = imu_q["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]].to_numpy()
gyro = imu_q["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]].to_numpy()
gt_se23 = utils.get_se23_poses(data["mocap_quat"](qt), data["mocap_pos"].derivative(nu=1)(qt), data["mocap_pos"](qt))
ekf = model.EKF(gt_se23[0], miluv.anchors, miluv.tag_moment_arms)
rt = data["uwb_range"]["timestamp"].to_numpy(); rr = data["uwb_range"]["range"].to_numpy(); rto = data["uwb_range"]["to_id"].to_numpy(); rfrom = data["uwb_range"]["from_id"].to_numpy()
ht = data["height"]["timestamp"].to_numpy(); hr = data["height"]["range"].to_numpy()
ri = 0; hi = 0; rows = []
for i in range(len(qt)):
    u = np.array([gyro[i, 1], gyro[i, 2], gyro[i, 3], accel[i, 1], accel[i, 2], accel[i, 3]])
    dt = (qt[i] - qt[i - 1]) if i > 0 else 0
    ekf.predict(u, dt)
    while ri < len(rt) and rt[ri] <= qt[i]:
        if rt[ri] == qt[i]: ekf.correct({"range": float(rr[ri]), "to_id": int(rto[ri]), "from_id": int(rfrom[ri])})
        ri += 1
    while hi < len(ht) and ht[hi] <= qt[i]:
        if ht[hi] == qt[i] and use_height: ekf.correct({"height": float(hr[hi])})
        hi += 1
    P = np.array(ekf.pose); G = np.array(gt_se23[i]); p = P[:3, 4]; g = G[:3, 4]
    q = Rotation.from_matrix(P[:3, :3]).as_quat(); gq = Rotation.from_matrix(G[:3, :3]).as_quat()
    rows.append([qt[i], p[0], p[1], p[2], g[0], g[1], g[2], *q, *gq])
pd.DataFrame(rows, columns=["t", "x", "y", "z", "gx", "gy", "gz", "qx", "qy", "qz", "qw", "gqx", "gqy", "gqz", "gqw"]).to_csv(out, index=False)
e = np.array(rows)[:, 1:4] - np.array(rows)[:, 4:7]
print(exp_name, 'height' if use_height else 'noheight', 'devkit pos_rmse (per-axis RMS) %.3f | 3D rmse %.3f | z rms %.3f' % (np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(np.sum(e ** 2, 1))), np.sqrt(np.mean(e[:, 2] ** 2))))
