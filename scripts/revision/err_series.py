#!/usr/bin/env python3
"""Per-epoch position errors with exactly the scoring of the tables (final_table_v2.py, util_score.py), for the
distribution figures: MILUV / IULUI whole GT span, raw GT, KIMS lag T-1 plus the window flush; NTU VIRAL Leica prism,
4-DoF fit, GT start+10 s .. end-10 s; UTIL Vicon, whole span. Every function returns (t, e) with e the N x 3 error
(estimate - GT), or None when the file is missing; sqrt(mean(|e|^2)) equals the table RMSE."""
import os, sys, csv, numpy as np, pandas as pd
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(PKG, "scripts"), os.path.join(PKG, "scripts/revision")]
_cwd = os.getcwd(); os.chdir(PKG)
exec(open(os.path.join(PKG, "scripts/revision/final_table_v2.py")).read().split("MIL = [")[0])   # rows, Rof, T_BP, ef, std_score, ntu_score
os.chdir(_cwd)
import util_score
_rows = rows
def rows(f):
    """final_table_v2.rows, but an empty or header-only file gives an empty (0, 8) array instead of an IndexError."""
    try: return _rows(f)
    except (IndexError, ValueError): return np.zeros((0, 8))
def _short(P, lag): return len(P) < lag + 10

def err_std(ds, seq, f, lag):
    """MILUV / IULUI (std_score)."""
    f = os.path.join(PKG, f)
    if not os.path.exists(f): return None
    gt_t, gt = ef.load_gt(ds, seq); P = rows(f)
    if _short(P, lag): return None
    th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]
    ff = f.replace('_pose.txt', '_flush_pose.txt')
    if lag > 0 and os.path.exists(ff):
        F = rows(ff); n = len(F) // 2; first = F[:n]; last = F[n:]
        mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
    k = (th >= gt_t[0]) & (th <= gt_t[-1])
    return th[k], X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)

def err_ntu(seq, f, lag):
    """NTU VIRAL (ntu_score): prism point, 4-DoF (yaw + translation) fit to the Leica track."""
    f = os.path.join(PKG, f)
    if not os.path.exists(f): return None
    P = rows(f)
    if _short(P, lag): return None
    th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    prism = np.array([X[i] + Rof(Q[i]) @ T_BP for i in range(len(X))])
    L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(os.path.join(PKG, f'data/ntu/{seq}/leica.csv'))))[1:]]); lt = L[:, 0]; lp = L[:, 1:4]
    k = (th > lt[0] + 10) & (th < lt[-1] - 10)
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
    if not np.all(np.isfinite(E)): return th[k], np.full_like(E, 1e6)
    gc = G.mean(0); ec = E.mean(0); g2 = G[:, :2] - gc[:2]; e2 = E[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    return th[k], E - ((Rz @ G.T).T + t)

def err_util(seq, f, lag):
    """UTIL (util_score.score)."""
    f = os.path.join(PKG, f)
    if not os.path.exists(f): return None
    G = pd.read_csv(os.path.join(PKG, 'data/util', seq, 'gt.csv')).values; gt_t, gt = G[:, 0], G[:, 1:4]
    if _short(rows(f), lag): return None
    th, X = util_score.track(f, lag); k = (th >= gt_t[0]) & (th <= gt_t[-1])
    return th[k], X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)

def err_devkit_miluv(seq, variant="h"):
    """MILUV devkit EKF dump (run_ekf_dump.py): estimate and GT at the filter's own times. variant 'h' = with the laser height
    sensor (the MILUV paper's Table 10 filter, reproduced to 0.2-5.3 mm per axis), 'nh' = IMU + UWB only."""
    f = os.path.join(os.environ.get("MILUV_DEVKIT", os.path.expanduser("~/miluv_devkit")), f"results/ekf_{seq}_{variant}.csv")
    if not os.path.exists(f): return None
    D = pd.read_csv(f).values; return D[:, 0], D[:, 1:4] - D[:, 4:7]
