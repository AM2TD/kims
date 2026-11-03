#!/usr/bin/env python3
"""Score UTIL runs against Vicon in the anchor (Vicon) frame, no alignment: 3D / xy / z position RMSE over the whole GT
span (the KIMS output with lag T, plus its first/last window flush, as for MILUV/IULUI in final_table.py).
usage: util_score.py <seq> <pose_file> <lag> [--quiet]   (lag 0 for a filter output without flush)"""
import sys, os, argparse, numpy as np, pandas as pd
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def rows(f):
    R = [[float(x) for x in l.split()[:8]] for l in open(f) if not l.startswith('#') and len(l.split()) >= 8]
    return np.array(R) if R else np.zeros((0, 8))
def track(f, lag):
    P = rows(f); th = P[:-lag, 0] if lag > 0 else P[:, 0]; X = P[lag:, 1:4]
    ff = f.replace('_pose.txt', '_flush_pose.txt')
    F = rows(ff) if lag > 0 and os.path.exists(ff) else np.zeros((0, 8))
    if len(F) >= 2:
        n = len(F) // 2; first, last = F[:n], F[n:]
        mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
    return th, X
def score(seq, f, lag):
    G = pd.read_csv(os.path.join(PKG, 'data/util', seq, 'gt.csv')).values; gt_t, gt = G[:, 0], G[:, 1:4]
    th, X = track(f, lag)
    k = (th >= gt_t[0]) & (th <= gt_t[-1])
    if k.sum() < 10 or not np.all(np.isfinite(X[k])): return dict(rmse=np.inf, xy=np.inf, z=np.inf, zbias=np.nan, zstd=np.inf, cover=0.0)
    e = X[k] - np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1)
    return dict(rmse=float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), xy=float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1)))),
                z=float(np.sqrt(np.mean(e[:, 2] ** 2))), zbias=float(e[:, 2].mean()), zstd=float(e[:, 2].std()), cover=float((th[k][-1] - th[k][0]) / (gt_t[-1] - gt_t[0])))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('seq'); ap.add_argument('pose'); ap.add_argument('lag', type=int); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(); r = score(a.seq, a.pose, a.lag)
    if a.quiet: print(f"{r['rmse']:.3f} {r['xy']:.3f} {r['z']:.3f} {r['cover']:.3f}")
    else: print(f"{a.seq} {os.path.basename(a.pose)}: rmse {r['rmse']:.3f} xy {r['xy']:.3f} z {r['z']:.3f} (GT span covered {100 * r['cover']:.0f} %)")
