#!/usr/bin/env python3
import os
"""Per-update time of the MILUV devkit EKF without the height sensor (same loop as $MILUV_DEVKIT/run_ekf_dump.py
noheight): one update = IMU prediction to a measurement time plus the range correction there. Run from the devkit directory.
usage: timing_devkit_miluv.py <exp_name> <out_ms_file>   (writes 'time_s ms' per update)"""
import sys, os, time, numpy as np
sys.path.insert(0, os.environ.get("MILUV_DEVKIT", os.path.expanduser("~/miluv_devkit")))
from miluv.data import DataLoader
import miluv.utils as utils
import examples.ekfutils.imu_one_robot_models as model
exp_name, out = sys.argv[1], sys.argv[2]
miluv = DataLoader(exp_name, imu="px4", cam=None, mag=False); data = miluv.data["ifo001"]
qt = np.append(data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()); qt = np.sort(np.unique(qt))
imu_q = miluv.query_by_timestamps(qt, robots="ifo001", sensors="imu_px4")["ifo001"]
accel = imu_q["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]].to_numpy()
gyro = imu_q["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]].to_numpy()
gt_se23 = utils.get_se23_poses(data["mocap_quat"](qt), data["mocap_pos"].derivative(nu=1)(qt), data["mocap_pos"](qt))
ekf = model.EKF(gt_se23[0], miluv.anchors, miluv.tag_moment_arms)
rt = data["uwb_range"]["timestamp"].to_numpy(); rr = data["uwb_range"]["range"].to_numpy(); rto = data["uwb_range"]["to_id"].to_numpy(); rfrom = data["uwb_range"]["from_id"].to_numpy()
ri = 0; rows = []
for i in range(len(qt)):
    t0 = time.perf_counter(); nr = 0
    u = np.array([gyro[i, 1], gyro[i, 2], gyro[i, 3], accel[i, 1], accel[i, 2], accel[i, 3]])
    ekf.predict(u, (qt[i] - qt[i - 1]) if i > 0 else 0)
    while ri < len(rt) and rt[ri] <= qt[i]:
        if rt[ri] == qt[i]: ekf.correct({"range": float(rr[ri]), "to_id": int(rto[ri]), "from_id": int(rfrom[ri])}); nr += 1
        ri += 1
    rows.append((qt[i], (time.perf_counter() - t0) * 1e3, nr))
with open(out, "w") as f:
    f.write(f"# MILUV devkit EKF (no height) per-step time: time_s ms n_range_corrections, {len(rows)} steps\n")
    for t, ms, nr in rows: f.write(f"{t:.6f} {ms:.4f} {nr}\n")
print(exp_name, "steps", len(rows), "mean ms %.3f" % np.mean([r[1] for r in rows]))
