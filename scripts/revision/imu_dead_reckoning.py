#!/usr/bin/env python3
"""E12: how far the IMU alone goes. Dead reckoning with the same IMU stream KIMS uses (rest-phase biases subtracted as in the
final runs, g 9.81), started every 1 s from the ground-truth position and velocity and from KIMS's own estimated attitude at
that time (the datasets have no attitude ground truth that is in the IMU frame; NTU has none at all), propagated with no UWB
for 1, 2, 3, 5 and 10 s. Drift = |p_imu(t0+h) - p_gt(t0+h)|. Also the plain dead reckoning from the first committed epoch
(drift after 10, 30 and 60 s). NTU: Leica prism point (eee: dataset transform; nya/sbs: 4-DoF fit of the Leica track to the
KIMS run, as in the tables). usage: python3 scripts/revision/imu_dead_reckoning.py > result/final/imu_dead_reckoning.txt"""
import os, sys, re, csv, json, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es, eval_flip as ef
os.chdir(es.PKG)
G = 9.81; T_BP = np.array([-0.293656, -0.012288, -0.273095]); H = [1.0, 2.0, 3.0, 5.0, 10.0]
MIL = ["default_1_remoteControlHighPace_1", "default_1_random3_1", "default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2",
       "default_1_remoteControlLowPace_0", "default_1_circular2D_0", "default_1_remoteControlHighPace_0", "obstacles_1_random3_0b", "default_1_random3_2",
       "default_1_remoteControlHighPace_2", "noAprilTags_1_random3_2b", "noAprilTags_1_remoteControlHighPace_2b"]
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]; IU = ["SquareTurn", "CircleSpin", "AC3_125000"]
UT = sorted(os.listdir("data/util"))
def imu_path(ds, s): return {"miluv": f"data/miluv/{s}/ifo001/imu_px4.csv", "ntu": f"data/ntu/{s}/imu.csv", "util": f"data/util/{s}/imu.csv", "iului": f"data/iului/{s}/imu.csv"}[ds]
def log_bias(lab, s):
    txt = open(f"result/runlogs/{lab}{s}.log").read(); m = re.search(r"acc bias \[([^\]]+)\] and gyro bias \[([^\]]+)\]", txt)
    return (np.array(m.group(1).split(), float), np.array(m.group(2).split(), float)) if m else (np.zeros(3), np.zeros(3))
def expm(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
def integrate(ds, s, lab):
    D = np.loadtxt(imu_path(ds, s), delimiter=",", skiprows=1); t = D[:, 0]; w = D[:, 1:4]; a = D[:, 4:7]
    ba, bg = log_bias(lab, s); a = a - ba; w = w - bg
    n = len(t); Q = np.zeros((n, 3, 3)); Q[0] = np.eye(3)
    dt = np.diff(t)
    for k in range(n - 1): Q[k + 1] = Q[k] @ expm(w[k] * dt[k])
    qa = np.einsum("nij,nj->ni", Q, a)                       # specific force in the frame of Q (gyro-integrated from identity)
    C1 = np.vstack([np.zeros(3), np.cumsum(qa[:-1] * dt[:, None], 0)]); C2 = np.vstack([np.zeros(3), np.cumsum(C1[:-1] * dt[:, None], 0)])
    return t, Q, C1, C2
def gt(ds, s, kimsf, lag):
    if ds == "util": g = np.loadtxt(f"data/util/{s}/gt.csv", delimiter=",", skiprows=1); return g[:, 0], g[:, 1:4]
    if ds == "ntu":
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"data/ntu/{s}/leica.csv")))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
        if s.startswith("eee"): return ef.load_gt("ntu", s)
        P = es.rows(kimsf); th = P[:-lag, 0]; X = P[lag:, 1:4]; Qs = P[lag:, 4:8]       # 4-DoF fit of the Leica track to the KIMS prism estimate
        E = np.array([X[i] + es.Rof(Qs[i]) @ T_BP for i in range(len(X))]); k = (th > lt[0] + 10) & (th < lt[-1] - 10)
        Gm = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); Ek = E[k]; gc, ec = Gm.mean(0), Ek.mean(0)
        g2, e2 = Gm[:, :2] - gc[:2], Ek[:, :2] - ec[:2]; yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
        c, sn = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]); return lt, (Rz @ lp.T).T + (ec - Rz @ gc)
    return ef.load_gt(ds, s)
def run(ds, s, lab, lag):
    t, Q, C1, C2 = integrate(ds, s, lab); gt_t, gp = gt(ds, s, f"result/{ds}/diag/{lab}{s}_pose.txt", lag)
    P = es.rows(f"result/{ds}/diag/{lab}{s}_pose.txt"); pt = P[:-lag, 0]; pq = P[lag:, 4:8]
    ip = lambda tt: np.stack([np.interp(tt, gt_t, gp[:, j]) for j in range(3)], 1)
    iq = lambda tt: pq[np.clip(np.searchsorted(pt, tt), 0, len(pt) - 1)]
    t0s = np.arange(max(gt_t[0] + 0.2, pt[0]), gt_t[-1] - 10.2, 1.0); out = []
    for t0 in t0s:
        i0 = np.searchsorted(t, t0); i0 = min(i0, len(t) - 2)
        p0 = ip(np.array([t0]))[0]; v0 = (ip(np.array([t0 + 0.05]))[0] - ip(np.array([t0 - 0.05]))[0]) / 0.1
        R0 = es.Rof(iq(t0)) @ Q[i0].T; d = []
        for h in H:
            i1 = np.searchsorted(t, t0 + h); i1 = min(i1, len(t) - 1); tau = t[i1] - t[i0]
            p = p0 + v0 * tau + R0 @ (C2[i1] - C2[i0] - C1[i0] * tau) - 0.5 * G * tau ** 2 * np.array([0, 0, 1.0])
            if ds == "ntu": p = p + (R0 @ Q[i1]) @ T_BP                # the Leica track is the prism point
            d.append(np.linalg.norm(p - ip(np.array([t[i1]]))[0]))
        out.append(d)
    out = np.array(out)
    # plain dead reckoning from the first committed epoch
    t0 = max(gt_t[0] + 0.2, pt[0]); i0 = np.searchsorted(t, t0); p0 = ip(np.array([t0]))[0]; v0 = (ip(np.array([t0 + 0.05]))[0] - ip(np.array([t0 - 0.05]))[0]) / 0.1; R0 = es.Rof(iq(t0)) @ Q[i0].T; dr = []
    for h in [10.0, 30.0, 60.0]:
        if t0 + h > gt_t[-1]: dr.append(np.nan); continue
        i1 = min(np.searchsorted(t, t0 + h), len(t) - 1); tau = t[i1] - t[i0]
        p = p0 + v0 * tau + R0 @ (C2[i1] - C2[i0] - C1[i0] * tau) - 0.5 * G * tau ** 2 * np.array([0, 0, 1.0])
        if ds == "ntu": p = p + (R0 @ Q[i1]) @ T_BP
        dr.append(np.linalg.norm(p - ip(np.array([t[i1]]))[0]))
    return out, np.array(dr)
print("IMU dead reckoning (no UWB), same IMU stream and rest-phase bias correction as KIMS, started every 1 s from the ground-truth")
print("position/velocity and KIMS's attitude estimate. Drift [m] = |p_imu - p_gt| after h seconds; median / p90 over all starts of a dataset.\n")
print(f"{'dataset':8s} {'seq':>4s} {'starts':>6s} | " + " ".join(f"{'h=' + str(int(h)) + ' s':>13s}" for h in H) + " | from the first epoch: 10 s   30 s   60 s (median over sequences)")
summary = {}
for ds, seqs, lab, lag in [("miluv", MIL, "f32a_", 228), ("ntu", NTU, "f32a_", 117), ("util", UT, "uf1_", 99), ("iului", IU, "f32g_", 199)]:
    allv = []; drs = []; per = []
    for s in seqs:
        try: out, dr = run(ds, s, lab, lag)
        except Exception as e: print(f"# {ds} {s}: {type(e).__name__} {e}"); continue
        allv.append(out); drs.append(dr); per.append((s, np.median(out[:, 1]), np.median(out[:, 4])))
    A = np.vstack(allv); D = np.array(drs); summary[ds] = (A, D, per)
    print(f"{ds:8s} {len(allv):4d} {len(A):6d} | " + " ".join(f"{np.median(A[:, j]):6.2f}/{np.percentile(A[:, j], 90):6.2f}" for j in range(len(H))) + " | " + " ".join(f"{np.nanmedian(D[:, j]):6.1f}" for j in range(3)))
print("\nPer sequence, median drift at 2 s and 10 s [m]:")
for ds in summary:
    for s, d2, d10 in summary[ds][2]: print(f"  {ds:6s} {s:40s} {d2:6.2f} {d10:7.2f}")
print("\nMILUV/IULUI/UTIL ground truth = motion capture body origin; NTU = Leica prism (estimates moved to the prism with the propagated attitude).")
print("Attitude at the start is KIMS's estimate at that time (the IMU-frame attitude the smoother actually carries), so the drift measures IMU propagation error only.")
