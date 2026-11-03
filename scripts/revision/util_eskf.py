#!/usr/bin/env python3
"""UTIL devkit ESKF baseline (utiasDSL/util-uwb-dataset scripts/estimation/eskf_class.py, unmodified) on one flight,
run exactly as the devkit main.py: IMU and TDOA downsampled by 8 (per anchor pair in tdoa2 mode, tdoa3 overall), start
(1.25, 0, 0.07) with identity attitude, posterior at every merged IMU/UWB time. Headless, with dictionary time lookups
(same first-match semantics as main.py's isin). Writes result/util/diag/eskf_<seq>_pose.txt (time x y z qx qy qz qw).
usage: util_eskf.py <seq>      (UTIL_ROOT, default ~/util_raw: dataset/, util-uwb-dataset/ devkit clone, pylib/ with pyquaternion)"""
import sys, os, numpy as np, pandas as pd
UR = os.environ.get("UTIL_ROOT", os.path.expanduser("~/util_raw")); DEV = UR + "/util-uwb-dataset/scripts"
sys.path[:0] = [DEV + "/estimation", DEV, UR + "/pylib"]
from eskf_class import ESKF
from pyquaternion import Quaternion
from utility.praser import extract_gt, extract_tdoa, extract_acc, extract_gyro, interp_meas, extract_tdoa_meas
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seq = sys.argv[1]; const = seq.split("-")[0]
df = pd.read_csv(f"{UR}/dataset/flight-dataset/csv-data/{const}/{seq}.csv")
anchor_position = np.load(f"{UR}/dataset/flight-dataset/survey-results/anchor_{const}.npz")["an_pos"]
def downsamp(x): return x[0::2][0::2][0::2]
gt_pose = extract_gt(df); tdoa = extract_tdoa(df); acc = extract_acc(df); gyr = extract_gyro(df)
t_vicon = gt_pose[:, 0]; t_imu = acc[:, 0]
g_syn = [interp_meas(gyr[:, 0], gyr[:, j], t_imu).reshape(-1, 1) for j in (1, 2, 3)]
imu = np.concatenate([acc[:, 1:]] + g_syn, axis=1)
min_t = min(tdoa[0, 0], t_imu[0], t_vicon[0])
t_imu = (t_imu - min_t).reshape(-1, 1); tdoa[:, 0] = tdoa[:, 0] - min_t
t_imu = downsamp(t_imu); imu = downsamp(imu)
if "tdoa2" in seq:
    parts = extract_tdoa_meas(tdoa[:, 0], tdoa[:, 1:4])
    tdoa_c = np.concatenate([downsamp(p) for p in parts], axis=0)
else:
    tdoa_c = downsamp(tdoa)
o = np.argsort(tdoa_c[:, 0]); t_uwb = tdoa_c[o, 0].reshape(-1, 1); uwb = tdoa_c[o, 1:4]
t = np.unique(np.sort(np.concatenate((t_imu, t_uwb)), axis=None)); K = t.shape[0]
X0 = np.zeros((6, 1)); X0[0] = 1.25; X0[1] = 0.0; X0[2] = 0.07
q0 = Quaternion([1, 0, 0, 0])
P0 = np.diag([0.1 ** 2] * 9)
eskf = ESKF(X0, q0, P0, K)
im, um = {}, {}
for i, v in enumerate(t_imu[:, 0]): im.setdefault(v, i)
for i, v in enumerate(t_uwb[:, 0]): um.setdefault(v, i)
for k in range(1, K):
    ik = im.get(t[k - 1]); uk = um.get(t[k - 1]); dt = t[k] - t[k - 1]
    eskf.predict(imu[ik if ik is not None else 0, :], dt, ik is not None, k)
    if uk is not None: eskf.UWB_correct(uwb[uk, :], anchor_position, k)
os.makedirs(os.path.join(PKG, "result/util/diag"), exist_ok=True)
with open(os.path.join(PKG, "result/util/diag", f"eskf_{seq}_pose.txt"), "w") as f:
    f.write("# time x y z qx qy qz qw  (UTIL devkit ESKF posterior)\n")
    for k in range(K):
        q = eskf.q_list[k]; x = eskf.Xpo[k]
        f.write(f"{t[k] + min_t:.6f} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {q[0]:.6f}\n")
print(seq, "done", K, "steps")
