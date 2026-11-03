#!/usr/bin/env python3
"""Orientation RMSE (deg) of the final runs against the motion-capture attitude: MILUV, IULUI and UTIL (NTU VIRAL's Leica
total station gives position only). UTIL has one UWB tag, so attitude comes only through the IMU kinematics (gravity for tilt,
horizontal acceleration for heading); its methods are KIMS, SFUISE (body output) and our run of the devkit ESKF (the UTIL paper's
Table 4 reports position only). Error = rotation angle of R_gt^T R_est (geodesic), RMS over the ground-truth span covered by each
method's committed output, no alignment, GT attitude by slerp. KIMS: lag T-1 output only (its first/last window flush carries
no attitude). Split: tilt = angle between the estimated and true body z axes (roll/pitch), heading = yaw of R_gt^T R_est.
"aligned": RMS after removing one constant world-frame rotation per run (R_est ~ W R_gt, chordal mean), applied to every
method alike. It is needed for SFUISE, which reports attitude in its own navigation frame: that frame is levelled by SFUISE's
estimated gravity vector, not by the published pose, so its raw attitude carries a constant 4-18 deg tilt (0.1-1 deg for KIMS).
"from 20 s": total RMS from 20 s after the GT start, i.e. after the heading has converged from the cold start (KIMS and SFUISE
start at yaw 0; the devkit EKF starts from the GT attitude), reported with the GT initial yaw.
Methods: KIMS (final runs), SFUISE (authors' published configuration), MILUV filter = devkit EKF + height (our run of the MILUV
paper's Table 10 filter, which reports position only; it starts from the ground-truth state, attitude included).
usage: python3 scripts/revision/orient_final.py > result/runlogs/orient_final.txt   (run from the package root)"""
import os, sys, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es
from datasets import CONFIG
os.chdir(es.PKG)
DEVKIT = os.environ.get("MILUV_DEVKIT", os.path.expanduser("~/miluv_devkit"))
MIL = ["default_1_remoteControlHighPace_1", "default_1_random3_1", "default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2",
       "default_1_remoteControlLowPace_0", "default_1_circular2D_0", "default_1_remoteControlHighPace_0", "obstacles_1_random3_0b", "default_1_random3_2",
       "default_1_remoteControlHighPace_2", "noAprilTags_1_random3_2b", "noAprilTags_1_remoteControlHighPace_2b"]
IU = ["SquareTurn", "CircleSpin", "AC3_125000"]
UT = sorted(os.listdir(os.path.join(es.PKG, "data/util")))
def gt_file(ds, seq):
    return os.path.join(es.PKG, f"data/util/{seq}/gt_pose.csv") if ds == "util" else os.path.join(es.PKG, CONFIG[ds]["data_dir"].format(seq=seq), "mocap.csv")
def gt(ds, seq):
    g = np.loadtxt(gt_file(ds, seq), delimiter=",", skiprows=1)
    t, k = np.unique(g[:, 0], return_index=True); return t, g[k, 1:4], Slerp(t, Rot.from_quat(g[k, 4:8]))
def score(t, X, Q, G, skip=0.0):
    gt_t, gp, sl = G
    k = (t > gt_t[0] + skip) & (t < gt_t[-1]) & np.all(np.isfinite(Q), 1) & (np.linalg.norm(Q, axis=1) > 0.5)
    if k.sum() < 50: return None
    t, X, Q = t[k], X[k], Q[k]
    pos = np.sqrt(np.mean(np.sum((X - np.stack([np.interp(t, gt_t, gp[:, j]) for j in range(3)], 1)) ** 2, 1)))
    if not np.isfinite(pos) or pos > 100: return {"div": True}
    Rg = sl(t); Re = Rot.from_quat(Q); E = (Rg.inv() * Re).as_matrix()
    ang = np.degrees(np.arccos(np.clip((np.trace(E, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    zg = Rg.as_matrix()[:, :, 2]; ze = Re.as_matrix()[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(np.sum(zg * ze, 1), -1, 1)))
    yaw = np.degrees(np.arctan2(E[:, 1, 0], E[:, 0, 0]))
    Mg, Me = Rg.as_matrix(), Re.as_matrix()
    U, _, Vt = np.linalg.svd(np.mean(np.einsum("nij,nkj->nik", Me, Mg), 0)); W = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    Ea = np.einsum("nji,njk->nik", Mg, np.einsum("ji,njk->nik", W, Me))   # R_gt^T W^T R_est
    angc = np.degrees(np.arccos(np.clip((np.trace(Ea, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    return {"div": False, "rmse": rms(ang), "tilt": rms(tilt), "yaw": rms(yaw), "off": rms(angc), "cover": (t[-1] - t[0]) / (gt_t[-1] - gt_t[0]), "dur": t[-1] - t[0]}
def pose_file(f, lag):
    if not os.path.exists(f): return None
    P = es.rows(f); th = P[:-lag, 0] if lag else P[:, 0]; return th, P[lag:, 1:4], P[lag:, 4:8]
def devkit(seq):
    f = os.path.join(DEVKIT, f"results/ekf_{seq}_h.csv")
    if not os.path.exists(f): return None
    D = pd.read_csv(f)
    if "qw" not in D: return None
    return D["t"].values, D[["x", "y", "z"]].values, D[["qx", "qy", "qz", "qw"]].values
def fmt(r, r20=None):
    if r is None: return f"{'-':>34s}"
    if r["div"]: return f"{'diverged':>34s}"
    return f"{r['rmse']:6.2f} ({r['tilt']:5.2f} {r['yaw']:5.2f}) {r['off']:5.2f} {r20['rmse'] if r20 and not r20['div'] else np.nan:6.2f}"
def yaw0(ds, seq):
    g = np.loadtxt(gt_file(ds, seq), delimiter=",", skiprows=1)
    return Rot.from_quat(g[0, 4:8]).as_euler("zyx", degrees=True)[0]
print("Orientation RMSE [deg] vs motion capture: raw total (tilt, heading), aligned (one constant world rotation per run removed, all")
print("methods), and raw total from 20 s after the GT start (heading converged from the cold start).")
print("NTU VIRAL: no attitude ground truth (Leica prism position only). KIMS lag T-1 output; SFUISE published config; MILUV filter =")
print("devkit EKF + height (our run of the paper's Table 10 filter; starts from the GT state).\n")
res = {}
COLS = {"miluv": ["KIMS", "SFUISE", "EKF + height"], "iului": ["KIMS", "SFUISE"], "util": ["KIMS", "SFUISE", "ESKF (devkit)"]}
def source(ds, c, s, lab, lag):
    if c == "KIMS": return pose_file(f"result/{ds}/diag/{lab}{s}_pose.txt", lag)
    if c == "SFUISE": return pose_file(f"result/{ds}/diag/{'sfuisebody' if ds == 'util' else 'sfuise'}_{s}_pose.txt", 0)
    return devkit(s) if ds == "miluv" else pose_file(f"result/util/diag/eskf_{s}_pose.txt", 0)
for ds, seqs, lab, lag in [("miluv", MIL, "f32a_", 228), ("iului", IU, "f32g_", 199), ("util", UT, "uf1_", 99)]:
    cols = COLS[ds]
    print(f"{ds.upper():6s} {'sequence (GT initial yaw)':48s} | " + " | ".join(f"{c + ': raw (tilt yaw) aligned 20s':>34s}" for c in cols))
    for s in seqs:
        G = gt(ds, s); row = []
        for c in cols:
            src = source(ds, c, s, lab, lag)
            r = score(*src, G) if src is not None else None; r20 = score(*src, G, 20.0) if src is not None else None
            row.append((r, r20)); res.setdefault((ds, c), []).append(r); res.setdefault((ds, c + "20"), []).append(r20)
        print(f"{'':6s} {s + f' ({yaw0(ds, s):+.0f})':48s} | " + " | ".join(fmt(r, r20) for r, r20 in row))
    for c in cols:
        v = [r for r in res[(ds, c)] if r is not None and not r["div"]]; nd = sum(1 for r in res[(ds, c)] if r is not None and r["div"])
        v20 = [r for r in res[(ds, c + "20")] if r is not None and not r["div"]]
        if v: print(f"{'':6s} {'median ' + c:48s} | from 20 s {np.median([r['rmse'] for r in v20]):5.2f}, total {np.median([r['rmse'] for r in v]):5.2f}, tilt {np.median([r['tilt'] for r in v]):5.2f}, heading {np.median([r['yaw'] for r in v]):5.2f}, aligned {np.median([r['off'] for r in v]):5.2f} deg over {len(v)} runs" + (f", {nd} diverged" if nd else ""))
    for c in cols[1:]:
        a = [(k, o) for k, o in zip(res[(ds, "KIMS")], res[(ds, c)]) if k and o and not k["div"] and not o["div"]]
        a20 = [(k, o) for k, o in zip(res[(ds, "KIMS20")], res[(ds, c + "20")]) if k and o and not k["div"] and not o["div"]]
        if a: print(f"{'':6s} {'KIMS lower than ' + c:48s} | raw {sum(1 for k, o in a if k['rmse'] < o['rmse'])}/{len(a)}, aligned {sum(1 for k, o in a if k['off'] < o['off'])}/{len(a)}, raw from 20 s {sum(1 for k, o in a20 if k['rmse'] < o['rmse'])}/{len(a20)}")
    print()
# ---- markdown page: result/final/orientation_rmse.md
def cell(r, key):
    return "-" if r is None else ("diverged" if r["div"] else f"{r[key]:.1f}")
md = ["# Orientation RMSE (deg)\n",
      "Rotation angle of R_gt^T R_est against motion capture or Vicon. NTU VIRAL has no attitude ground truth (Leica prism position only).",
      "**raw** = no alignment (as for position). **aligned** = one constant world-frame rotation per run removed, for every method alike;",
      "SFUISE needs it because it reports attitude in its own navigation frame, which is levelled by its estimated gravity vector, so its raw",
      "attitude carries a constant 4-18 deg tilt (KIMS 0.1-1 deg). KIMS and SFUISE start at yaw 0; the MILUV filter (devkit EKF + height,",
      "our run of the MILUV paper's Table 10 filter, which reports position only) starts from the ground-truth attitude. GT start yaw in",
      "brackets: on remoteControlHighPace_2 (-103 deg) and SquareTurn (-60 deg) KIMS needs about 15 s to turn its heading, which",
      "dominates its error there (from 20 s on: 5.8 and 3.3 deg). KIMS: committed lag T-1 output (the window flush has no attitude).\n"]
for ds, seqs, cols in [("miluv", MIL, ["KIMS", "SFUISE", "EKF + height"]), ("iului", IU, ["KIMS", "SFUISE"])]:
    md += [f"## {'MILUV (13)' if ds == 'miluv' else 'IULUI (3)'}\n", "| Sequence (GT start yaw) | " + " | ".join(f"{c} raw | {c} aligned" for c in cols) + " |",
           "|---|" + "---|" * (2 * len(cols))]
    for i, s in enumerate(seqs):
        md.append(f"| {s} ({yaw0(ds, s):+.0f}) | " + " | ".join(f"{cell(res[(ds, c)][i], 'rmse')} | {cell(res[(ds, c)][i], 'off')}" for c in cols) + " |")
    meds = []
    for c in cols:
        v = [r for r in res[(ds, c)] if r is not None and not r["div"]]
        meds.append(f"**{np.median([r['rmse'] for r in v]):.1f}** | **{np.median([r['off'] for r in v]):.1f}**" if c == "KIMS" else f"{np.median([r['rmse'] for r in v]):.1f} | {np.median([r['off'] for r in v]):.1f}")
    md.append("| **median** | " + " | ".join(meds) + " |\n")
    for c in cols[1:]:
        a = [(k, o) for k, o in zip(res[(ds, "KIMS")], res[(ds, c)]) if k and o and not k["div"] and not o["div"]]
        md.append(f"- KIMS lower than {c}: raw {sum(1 for k, o in a if k['rmse'] < o['rmse'])}/{len(a)}, aligned {sum(1 for k, o in a if k['off'] < o['off'])}/{len(a)}" + (f" ({sum(1 for r in res[(ds, c)] if r and r['div'])} diverged runs left out)" if any(r and r["div"] for r in res[(ds, c)]) else ""))
    md.append("")
# UTIL: one tag, so heading reaches the estimate only through the IMU (horizontal acceleration); medians per group
md += ["## UTIL (79, one UWB tag)\n",
       "With one tag the ranges carry no direct heading information; tilt is fixed by gravity, heading only weakly through the IMU",
       "(horizontal acceleration). The devkit ESKF keeps most of its start heading (its mean heading error follows the GT start yaw,",
       "r = -0.79); KIMS removes the start offset (r = 0.01) but its heading wanders on 5 of 79 flights (36-102 deg RMS:",
       "const2-trial3-tdoa3, const4-trial2-tdoa2-traj2, const4-trial2-tdoa3-traj2/traj3, const4-trial3-tdoa2-traj1) while its position",
       "there stays at or below the group median. The UTIL paper reports position only; ESKF here = our run of the devkit ESKF.\n",
       "| Group (flights) | KIMS raw | KIMS aligned | SFUISE raw | SFUISE aligned | ESKF raw | ESKF aligned |", "|---|---|---|---|---|---|---|"]
def gmed(c, idx, key):
    v = [res[("util", c)][i] for i in idx]; v = [r for r in v if r is not None and not r["div"]]
    return f"{np.median([r[key] for r in v]):.1f}" if v else "-"
groups = [(f"{c} {m}", [i for i, q in enumerate(UT) if q.startswith(c + "-") and m in q]) for c in ["const1", "const2", "const3", "const4"] for m in ["tdoa2", "tdoa3"]]
for g, idx in groups + [("**all**", list(range(len(UT))))]:
    md.append(f"| {g} ({len(idx)}) | " + " | ".join(f"{gmed(c, idx, 'rmse')} | {gmed(c, idx, 'off')}" for c in COLS["util"]) + " |")
for c in COLS["util"][1:]:
    a = [(k, o) for k, o in zip(res[("util", "KIMS")], res[("util", c)]) if k and o and not k["div"] and not o["div"]]
    md.append(f"\n- KIMS lower than {c}: raw {sum(1 for k, o in a if k['rmse'] < o['rmse'])}/{len(a)}, aligned {sum(1 for k, o in a if k['off'] < o['off'])}/{len(a)}" + (f" ({sum(1 for r in res[('util', c)] if r and r['div'])} diverged runs left out)" if any(r and r["div"] for r in res[("util", c)]) else ""))
# summaries across runs: median and mean of the per-run RMSE, and the time-weighted pooled RMSE
md += ["\n## Summary over runs\n", "Every value above is an RMSE over time: sqrt(mean of theta_k^2), theta_k = rotation angle of R_gt^T R_est",
       "(the length of the rotation-error vector, so the per-axis RMSEs of that vector combine as for 3D position). Tables give the",
       "median over runs, as the position tables do; mean and time-weighted pooled RMSE for comparison (diverged runs left out):\n",
       "| Dataset | Method | median | mean | pooled |", "|---|---|---|---|---|"]
for ds, seqs, lab, lag in [("miluv", MIL, "f32a_", 228), ("iului", IU, "f32g_", 199), ("util", UT, "uf1_", 99)]:
    for c in COLS[ds]:
        per, mse_t, tt = [], 0.0, 0.0
        for s_, r in zip(seqs, res[(ds, c)]):
            if r is None or r["div"]: continue
            per.append(r["rmse"]); mse_t += r["rmse"] ** 2 * r["dur"]; tt += r["dur"]
        md.append(f"| {ds.upper()} | {c} | {np.median(per):.1f} | {np.mean(per):.1f} | {np.sqrt(mse_t / tt):.1f} |")
md.append("\nOn UTIL the median hides the five KIMS heading-drift flights; the mean and pooled values show them.")
open(os.path.join(es.PKG, "result/final/orientation_rmse.md"), "w").write("\n".join(md) + "\n")
