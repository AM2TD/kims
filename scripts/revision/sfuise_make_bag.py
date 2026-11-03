#!/usr/bin/env python3
"""SFUISE input for MILUV / NTU VIRAL from the same csv files KIMS reads (ToA, several tags; SFUISE patched for per-tag
lever arms and an initial nav->UWB transform).
    python3 sfuise_make_bag.py <miluv|ntu|iului> <seq>   (ROS noetic and the sfuise workspace sourced)
Writes $SFUISE_BAGS/<dataset>_<seq>.bag (default ~/util_raw/kims_bags) (/imu_data, /rtls_flares one RTLSStick per range with id = tag,
/anchor_list 40 times at the start, /pose_data = GT stamps) and <...>.yaml: SFUISE's published ToA configuration
(config_test_isas-walk1.yaml: weights, IMU variances, 10 Hz knots, 100-knot window, 50 calibration windows, 0.3 m gate)
with the dataset's rates, tags, anchors, zero per-anchor offsets, output at the body origin (offset 0) and the start
pose KIMS uses: NTU the rest-phase fit (position, yaw) with the VN-100 z-down flip, MILUV the origin with yaw 0; attitude
tilt from the mean accelerometer over the first 3 s (KIMS align_still 3). IULUI (AC1, tags 8/9, anchors 0-3 from the meta
json): the KIMS start position (meta init) with yaw 0, z-up IMU. A second config <...>_g15.yaml has the 1.5 m gate."""
import sys, os, json, csv, numpy as np, pandas as pd, rospy, rosbag
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from isas_msgs.msg import RTLSStick, RTLSRange, Anchorlist, AnchorPosition
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, PKG + "/scripts"); from datasets import CONFIG, anchor_path
ds, seq = sys.argv[1], sys.argv[2]; OUT = os.environ.get("SFUISE_BAGS", os.path.expanduser("~/util_raw/kims_bags")); os.makedirs(OUT, exist_ok=True)
if ds == "ntu":
    d = f"{PKG}/data/ntu/{seq}"; imu = pd.read_csv(f"{d}/imu.csv"); uwb = pd.read_csv(f"{d}/uwb_range.csv")
    A = np.loadtxt(f"{PKG}/data/ntu/" + ("anchors_eee.txt" if seq.startswith("eee") else f"anchors_{seq}.txt"), comments="#")
    aid = [100, 101, 102]; tags = [2000, 2001, 2010, 2011]; off = CONFIG["ntu"]["tag_offsets"]
    G = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"{d}/leica.csv")))[1:]]); gq = None; z_down = True
    if seq.startswith("eee"): ini = json.load(open(f"{PKG}/result/runlogs/ntu_init_args.json"))[seq]
    else: ini = [x for x in open(f"{PKG}/result/runlogs/ntu_new_{seq}.log") if " init:" in x][0].split("init:")[1]
    iv = {k: float(v) for k, v in (t.split(":=") for t in ini.split())}; p0 = [iv["init_x"], iv["init_y"], iv["init_z"]]; yaw0 = iv["init_yaw"]
elif ds == "iului":
    d = f"{PKG}/data/iului/{seq}"; imu = pd.read_csv(f"{d}/imu.csv"); uwb = pd.read_csv(f"{d}/uwb_range.csv", usecols=["timestamp", "range", "from_id", "to_id"])
    mp = f"{PKG}/data/iului/AC1_{seq}.json"; mp = mp if os.path.exists(mp) else f"{PKG}/data/iului/{seq}.json"   # AC3_125000 etc.
    meta = json.load(open(mp)); A = np.array(meta["anchors"]); aid = [0, 1, 2, 3]; tags = [meta["tag0_id"], meta["tag1_id"]]
    off = [meta["tag0_offset"], meta["tag1_offset"]]
    M = pd.read_csv(f"{d}/mocap.csv").values; G = M[:, :4]; gq = M[:, 4:8]; z_down = False; p0 = list(meta["init"]); yaw0 = 0.0
else:
    d = f"{PKG}/data/miluv/{seq}/ifo001"; imu = pd.read_csv(f"{d}/imu_px4.csv"); uwb = pd.read_csv(f"{d}/uwb_range.csv", usecols=["timestamp", "range", "from_id", "to_id"])
    A = np.loadtxt(f"{PKG}/{anchor_path('miluv', seq)}", comments="#"); aid = list(range(6)); tags = CONFIG["miluv"]["tag_ids"]; off = CONFIG["miluv"]["tag_offsets"]
    M = pd.read_csv(f"{d}/mocap.csv").values; G = M[:, :4]; gq = M[:, 4:8]; z_down = False; p0 = [0.0, 0.0, 0.0]; yaw0 = 0.0
uwb = uwb[uwb.from_id.isin(tags) & uwb.to_id.isin(aid)].sort_values("timestamp")
ti = imu["timestamp"].values; acc = imu[["linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]].values
gyr = imu[["angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]].values
# initial attitude exactly as kims_node (align_still 3 s, z-down flip, then yaw about world z)
am = acc[ti <= ti[0] + 3.0].mean(0); Rb = np.diag([1.0, -1.0, -1.0]) if z_down else np.eye(3)
ub = Rb @ am; ub /= np.linalg.norm(ub); v = np.cross(ub, [0, 0, 1]); s = np.linalg.norm(v); c = ub[2]
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); Rq = np.eye(3) + K + K @ K * ((1 - c) / s ** 2) if s > 1e-9 else np.eye(3)
cy, sy = np.cos(np.radians(yaw0)), np.sin(np.radians(yaw0)); R0 = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ (Rq @ Rb)
def r2q(R):   # w, x, y, z
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6: return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; q = [0.0] * 4
    q[i + 1] = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) / 2; q[0] = (R[k, j] - R[j, k]) / (4 * q[i + 1])
    q[j + 1] = (R[j, i] + R[i, j]) / (4 * q[i + 1]); q[k + 1] = (R[k, i] + R[i, k]) / (4 * q[i + 1]); return q
q0 = r2q(R0)
T = lambda t: rospy.Time.from_sec(float(t))
with rosbag.Bag(f"{OUT}/{ds}_{seq}.bag", "w") as b:
    t0 = min(ti[0], uwb.timestamp.iloc[0])
    al = Anchorlist(); al.anchor = [AnchorPosition(id=int(i), position=__import__("geometry_msgs.msg", fromlist=["Point"]).Point(*map(float, A[k]))) for k, i in enumerate(aid)]
    for k in range(40): b.write("/anchor_list", al, T(t0 + 0.005 * k + 1e-3))
    for k in range(len(ti)):
        m = Imu(); m.header.stamp = T(ti[k]); m.header.frame_id = "imu"
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, acc[k])
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyr[k]); m.orientation.w = 1.0
        b.write("/imu_data", m, m.header.stamp)
    for t, r, tg, an in uwb[["timestamp", "range", "from_id", "to_id"]].itertuples(index=False):
        st = RTLSStick(); st.header.stamp = T(t); st.id = int(tg); st.ranges = [RTLSRange(id=int(an), range=float(r), ra=1)]
        b.write("/rtls_flares", st, st.header.stamp)
    for k in range(len(G)):
        p = PoseWithCovarianceStamped(); p.header.stamp = T(G[k, 0]); p.header.frame_id = "map"
        p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z = map(float, G[k, 1:4])
        if gq is not None: p.pose.pose.orientation.x, p.pose.pose.orientation.y, p.pose.pose.orientation.z, p.pose.pose.orientation.w = map(float, gq[k])
        else: p.pose.pose.orientation.w = 1.0
        b.write("/pose_data", p, p.header.stamp)
fu = len(uwb) / (uwb.timestamp.iloc[-1] - uwb.timestamp.iloc[0]); fi = len(ti) / (ti[-1] - ti[0])
cfg = f"""topic_imu: /imu_data
topic_uwb: /rtls_flares
topic_anchor_list: /anchor_list
topic_ground_truth: /pose_data
if_tdoa: false
uwb_frequency: {fu:.0f}
imu_frequency: {fi:.0f}
acc_ratio: false
gyro_unit: false
window_size: 100
uwb_sample_coeff: 1
imu_sample_coeff: {min(1.0, 100.0 / fi):.4f}
control_point_fps: 10
max_iter: 5
n_window_calib: 50
offset: [0.0, 0.0, 0.0]
tag_ids: {list(map(int, tags))}
tag_offsets: {[float(x) for o in off for x in o]}
t_nav_uwb_init: {[float(x) for x in p0]}
q_nav_uwb_init: {[float(x) for x in q0]}
accel_var_inv: [7.8483, 7.8483, 7.8483]
gyro_var_inv: [27.5646, 27.5646, 27.5646]
bias_accel_var_inv: [514.9887, 514.9887, 514.9887]
bias_gyro_var_inv: [2950.6680, 2950.6680, 2950.6680]
w_uwb: 7
w_accel: 1
w_gyro: 1
w_bias_accel: 1
w_bias_gyro: 1
if_reject_uwb: true
reject_uwb_thresh: 0.3
reject_uwb_window_width: 0.5
toa_offset: {[0.0] * len(aid)}
"""
open(f"{OUT}/{ds}_{seq}.yaml", "w").write(cfg)
open(f"{OUT}/{ds}_{seq}_g15.yaml", "w").write(cfg.replace("reject_uwb_thresh: 0.3", "reject_uwb_thresh: 1.5"))
print(f"{ds} {seq}: {len(ti)} IMU ({fi:.0f} Hz), {len(uwb)} ranges ({fu:.0f} Hz), {len(G)} GT stamps, start {np.round(p0, 3)} yaw {yaw0}")
