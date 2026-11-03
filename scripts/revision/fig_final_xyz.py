#!/usr/bin/env python3
"""Final comparison X/Y/Z figures, three sequences per figure (estimate vs GT over time).
  miluv: KIMS (f32a_, lag 228 + window flush), SFUISE (published 0.3 m gate, the table's variant), MILUV devkit EKF with the height sensor (the paper's Table 10 filter);
         raw mocap GT for every method.
  ntu:   KIMS (f32a_, lag 117 + window flush), SFUISE (published config); prism point in the UWB anchor frame, Leica GT through the
         dataset transform (eee) or the 4-DoF fit of the run with the lowest RMSE (nya, sbs); z panels show the reflected GT.
  util:  KIMS (uf1_, lag 99 + window flush), SFUISE (body origin), UTIL devkit ESKF; Vicon GT.
  iului: KIMS (f32g_, lag 199 + window flush), SFUISE (0.3 m gate); Qualisys GT.
Legend: 3D RMSE as in the tables.   usage: fig_final_xyz.py <miluv|ntu|util|iului> <out_prefix> [seq ...]"""
import sys, os, csv, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])            # rows, Rof, T_BP, std_score, ntu_score, ef
from util_score import score as uscore, track as utrack
ds, outp = sys.argv[1], sys.argv[2]
MIL = ['default_1_remoteControlHighPace_1', 'default_1_random3_1', 'default_1_random3_0', 'default_1_circular3D_0', 'default_1_remoteControlLowPace_0_v2', 'default_1_remoteControlLowPace_0', 'default_1_circular2D_0', 'default_1_remoteControlHighPace_0', 'obstacles_1_random3_0b', 'default_1_random3_2', 'default_1_remoteControlHighPace_2', 'noAprilTags_1_random3_2b', 'noAprilTags_1_remoteControlHighPace_2b']
DEF = {"miluv": MIL, "ntu": ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"],
       "util": ["const1-trial1-tdoa2", "const1-trial1-tdoa3", "const2-trial1-tdoa2", "const2-trial1-tdoa3", "const3-trial1-tdoa2", "const3-trial1-tdoa3", "const4-trial1-tdoa2-traj1", "const4-trial1-tdoa3-traj1", "const4-trial7-tdoa2-manual1"],
       "iului": ["SquareTurn", "CircleSpin", "AC3_125000"]}
seqs = sys.argv[3:] or DEF[ds]
COL = {"KIMS": "#1f4e9c", "SFUISE": "#e07b00", "devkit": "#2a9d4b"}
def with_flush(f, lag, th, X):   # append the last-window flush (positions) after the lag-T output; first window before it
    ff = f.replace("_pose.txt", "_flush_pose.txt")
    if not os.path.exists(ff): return th, X
    F = rows(ff); n = len(F) // 2
    if n == 0: return th, X
    first, last = F[:n], F[n:]; mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
    return np.concatenate([first[mf, 0], th, last[ml, 0]]), np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
def kims_track(f, lag, prism=False):
    R = rows(f); th = R[:-lag, 0]; X = R[lag:, 1:4]; Q = R[lag:, 4:8]
    if prism:   # prism point; the flush has no attitude: its first/last windows reuse the nearest committed attitude
        off_first = Rof(Q[0]) @ T_BP; off_last = Rof(Q[-1]) @ T_BP; X = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
        th2, X2 = with_flush(f, lag, th, X); pre = th2 < th[0]; post = th2 > th[-1]
        X2[pre] = X2[pre] + off_first; X2[post] = X2[post] + off_last; return th2, X2
    return with_flush(f, lag, th, X)
def series(s):
    if ds == "miluv":
        k = f"result/miluv/diag/f32a_{s}_pose.txt"; sf = f"result/miluv/diag/sfuise_{s}_pose.txt"; dk = os.path.join(os.environ.get("MILUV_DEVKIT", os.path.expanduser("~/miluv_devkit")), f"results/ekf_{s}_h.csv")
        out = [("KIMS", *kims_track(k, 228), std_score("miluv", s, k, 228))]
        if os.path.exists(sf): P = rows(sf); out.append(("SFUISE", P[:, 0], P[:, 1:4], std_score("miluv", s, sf, 0)))
        if os.path.exists(dk):
            D = pd.read_csv(dk).values; out.append(("devkit", D[:, 0], D[:, 1:4], np.sqrt(np.mean(np.sum((D[:, 1:4] - D[:, 4:7]) ** 2, 1)))))
        gt_t, gp = ef.load_gt("miluv", s); return gt_t, gp, out, None
    if ds == "util":
        k = f"result/util/diag/uf1_{s}_pose.txt"; out = [("KIMS", *utrack(k, 99), uscore(s, k, 99)["rmse"])]
        for name, f in [("SFUISE", f"result/util/diag/sfuisebody_{s}_pose.txt"), ("devkit", f"result/util/diag/eskf_{s}_pose.txt")]:
            if os.path.exists(f): th, X = utrack(f, 0); out.append((name, th, X, uscore(s, f, 0)["rmse"]))
        G = pd.read_csv(f"data/util/{s}/gt.csv").values; return G[:, 0], G[:, 1:4], out, None
    if ds == "iului":
        k = f"result/iului/diag/f32g_{s}_pose.txt"; sf = f"result/iului/diag/sfuise_{s}_pose.txt"
        out = [("KIMS", *kims_track(k, 199), std_score("iului", s, k, 199))]
        if os.path.exists(sf): P = rows(sf); out.append(("SFUISE", P[:, 0], P[:, 1:4], std_score("iului", s, sf, 0)))
        gt_t, gp = ef.load_gt("iului", s); return gt_t, gp, out, None
    # ntu: prism, UWB anchor frame
    k = f"result/ntu/diag/f32a_{s}_pose.txt"; sf = f"result/ntu/diag/sfuise_{s}_pose.txt"
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"data/ntu/{s}/leica.csv")))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
    out = [("KIMS", *kims_track(k, 117, prism=True), ntu_score(s, k, 117))]
    P = rows(sf); Xs = np.array([P[i, 1:4] + Rof(P[i, 4:8]) @ T_BP for i in range(len(P))]); out.append(("SFUISE", P[:, 0], Xs, ntu_score(s, sf, 0)))
    if s.startswith("eee"): gt_t, gp = ef.load_gt("ntu", s)
    else:
        best = None
        for name, th, X, r in out:
            kk = (th > lt[0] + 10) & (th < lt[-1] - 10)
            if not np.all(np.isfinite(X[kk])) or np.abs(X[kk]).max() > 1e3: continue
            G = np.stack([np.interp(th[kk], lt, lp[:, j]) for j in range(3)], 1); E = X[kk]; gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
            yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, si = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
            if best is None or r < best[0]: best = (r, (Rz @ lp.T).T + t)
        gt_t, gp = lt, best[1]
    return gt_t, gp, out, 1.5
for page in range(0, len(seqs), 3):
    ss = seqs[page:page + 3]; fig, ax = plt.subplots(3 * len(ss), 1, figsize=(13, 2.4 * 3 * len(ss)))
    ax = np.atleast_1d(ax)
    for si, s in enumerate(ss):
        gt_t, gp, out, plane = series(s); t0 = gt_t[0]
        for j, nm in enumerate("xyz"):
            a = ax[3 * si + j]; a.plot(gt_t - t0, gp[:, j], color="black", lw=2.2, label="GT")
            if j == 2 and plane is not None:
                a.plot(gt_t - t0, 2 * plane - gp[:, 2], color="0.45", lw=1.1, ls="--", label="GT reflected about the anchor plane"); a.axhline(plane, color="0.6", lw=0.8, ls=":")
            for name, th, X, r in out:
                k = (th >= gt_t[0]) & (th <= gt_t[-1]); lab = f"{'devkit EKF + height' if (name == 'devkit' and ds == 'miluv') else ('devkit ESKF' if name == 'devkit' else name)} ({'diverged' if not np.isfinite(r) or r > 100 else f'{r:.3f} m'})"
                a.plot(th[k] - t0, X[k, j], color=COL[name], lw=1.1, label=lab)
            lo, hi = np.nanmin(gp[:, j]), np.nanmax(gp[:, j])
            if j == 2 and plane is not None: lo, hi = min(lo, 2 * plane - hi), max(hi, 2 * plane - lo)
            a.set_ylim(lo - 0.7, hi + 0.7); a.set_ylabel(f"{nm} [m]"); a.grid(alpha=0.3)
            if j == 0: a.set_title(f"{ds.upper()} {s}: 3D RMSE in legend (same scoring as the tables)")
        ax[3 * si + 2].legend(loc="upper right", fontsize=7.5, ncol=4)
    ax[-1].set_xlabel("time since GT start [s]"); fig.tight_layout(); fn = f"{outp}_{page // 3 + 1}.png"; fig.savefig(fn, dpi=95); plt.close(fig); print("wrote", fn)
