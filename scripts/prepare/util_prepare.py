#!/usr/bin/env python3
import argparse, json, os, re, numpy as np, pandas as pd
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIFT = 100.0
LEVER = np.array([-0.01245, 0.00127, 0.0908])
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--bin", type=float, default=0.02)
ap.add_argument("--still", type=float, default=1.0)
ap.add_argument("--out", default="data/util", help="output root (default data/util)"); a = ap.parse_args()
seq = os.path.basename(a.csv)[:-4]; const = re.match(r"(const\d)", seq).group(1)
root = a.csv.split("/csv-data/")[0]
A = {}
for l in open(f"{root}/survey-results/anchor_{const}_survey.txt"):
    k, *v = l.strip().split(",")
    if k.endswith("_p"): A[int(k[2:-2])] = np.array([float(x) for x in v])
d = pd.read_csv(a.csv); out = os.path.join(PKG, a.out, seq); os.makedirs(out, exist_ok=True)
ta = d["t_acc"].dropna().values; acc = d[["acc_x", "acc_y", "acc_z"]].dropna().values * 9.81
tg = d["t_gyro"].dropna().values; gyr = d[["gyro_x", "gyro_y", "gyro_z"]].dropna().values * np.pi / 180.0
g = np.stack([np.interp(ta, tg, gyr[:, j]) for j in range(3)], 1)
pd.DataFrame({"timestamp": ta, "angular_velocity.x": g[:, 0], "angular_velocity.y": g[:, 1], "angular_velocity.z": g[:, 2],
              "linear_acceleration.x": acc[:, 0], "linear_acceleration.y": acc[:, 1], "linear_acceleration.z": acc[:, 2]}
             ).to_csv(os.path.join(out, "imu.csv"), index=False, float_format="%.9f")
t = d["t_tdoa"].dropna().values; ia = d["idA"].dropna().values.astype(int); ib = d["idB"].dropna().values.astype(int)
z = d["tdoa_meas"].dropna().values
sw = ia > ib; i = np.where(sw, ib, ia); j = np.where(sw, ia, ib); z = np.where(sw, -z, z)
keys, cnt = np.unique(i * 100 + j, return_counts=True)
keep_keys = keys[cnt >= 0.01 * cnt.max()]
if "tdoa2" in seq:
    keep_keys = np.array([k for k in keep_keys if (k // 100, k % 100) in {(n, n + 1) for n in range(7)} | {(0, 7)}])
sep = np.array([np.linalg.norm(A[ii] - A[jj]) for ii, jj in zip(i, j)])
ok = np.isin(i * 100 + j, keep_keys) & (np.abs(z) <= sep + 0.5)
n_stray = int((~np.isin(i * 100 + j, keep_keys)).sum()); n_phys = int((np.isin(i * 100 + j, keep_keys) & ~(np.abs(z) <= sep + 0.5)).sum())
t, i, j, z = t[ok], i[ok], j[ok], z[ok]
slots = [(int(k // 100), int(k % 100)) for k in keep_keys]; sidx = {s: n for n, s in enumerate(slots)}
s = np.array([sidx[(ii, jj)] for ii, jj in zip(i, j)])
b = np.floor(t / a.bin).astype(np.int64)
df = pd.DataFrame({"b": b, "s": s, "t": t, "z": z})
tb = df.groupby("b")["t"].mean()
agg = df.groupby(["b", "s"])["z"].mean().reset_index()
agg["timestamp"] = tb.loc[agg["b"]].values
agg = agg.sort_values(["timestamp", "s"])
pd.DataFrame({"timestamp": agg["timestamp"].values, "range": agg["z"].values + SHIFT, "from_id": 0, "to_id": agg["s"].values,
              "std": 0.224}).to_csv(os.path.join(out, "uwb_range.csv"), index=False, float_format="%.9f")
np.savetxt(os.path.join(out, "anchors_A.txt"), np.array([A[ii] for ii, jj in slots]), fmt="%.6f", header="anchor A of every TDOA slot")
np.savetxt(os.path.join(out, "anchors_B.txt"), np.array([A[jj] for ii, jj in slots]), fmt="%.6f", header="anchor B of every TDOA slot")
P = d[["t_pose", "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"]].dropna().values
pd.DataFrame({"timestamp": P[:, 0], "x": P[:, 1], "y": P[:, 2], "z": P[:, 3]}).to_csv(os.path.join(out, "gt.csv"), index=False, float_format="%.6f")
pd.DataFrame(P, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]).to_csv(os.path.join(out, "gt_pose.csv"), index=False, float_format="%.6f")
m = t < t[0] + a.still
zbar = np.array([z[m & (s == k)].mean() if np.any(m & (s == k)) else np.nan for k in range(len(slots))])
use = np.isfinite(zbar); AA = np.array([A[ii] for ii, jj in slots])[use]; BB = np.array([A[jj] for ii, jj in slots])[use]; zz = zbar[use]
p = np.array([*np.mean(list(A.values()), 0)[:2], 0.5])
for _ in range(50):
    dA = np.linalg.norm(p - AA, axis=1); dB = np.linalg.norm(p - BB, axis=1)
    r = (dB - dA) - zz; J = (p - BB) / dB[:, None] - (p - AA) / dA[:, None]
    w = 1.0 / np.maximum(1.0, np.abs(r) / 0.2)
    dp = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]; p = p + dp
    if np.linalg.norm(dp) < 1e-6: break
body = p - LEVER
meta = dict(seq=seq, const=const, mode="tdoa3" if "tdoa3" in seq else "tdoa2", bin=a.bin, num_slots=len(slots), slots=slots,
            n_tdoa=int(len(t)), n_epochs=int(agg["timestamp"].nunique()), dropped_stray=n_stray, dropped_impossible=n_phys,
            init=[float(x) for x in body], init_yaw=0.0, still_fit_rms=float(np.sqrt(np.mean(r ** 2))), gt_first=[float(x) for x in P[0, 1:4]])
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(f"{seq}: {len(slots)} slots, {meta['n_tdoa']} tdoa -> {meta['n_epochs']} epochs ({meta['n_epochs'] / (t[-1] - t[0]):.0f} Hz), "
      f"dropped stray {n_stray} impossible {n_phys}; init {np.round(body, 3)} (GT {np.round(P[0, 1:4], 3)}, fit rms {meta['still_fit_rms']:.3f})")
