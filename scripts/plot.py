#!/usr/bin/env python3
"""python3 scripts/plot.py <dataset> [sequence ...] [--labels kims swpgo ...] [--root DIR] [--out DIR]"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "scripts"))
from run import WINDOW, sequences, miluv_constellation, iului_meta
from evaluate import track, gt_pose, position_rmse, rot, NTU_PRISM, rows

COLOR = {"kims": "#c0392b", "swpgo": "#2e8b57", "pf": "#d4a017", "sfuise": "#2a7fbf"}

def anchors(ds, seq):
    if ds == "iului": return np.array(iului_meta(seq)["anchors"])
    if ds == "miluv": f = f"{PKG}/data/miluv/anchors_constellation{miluv_constellation(seq)}.txt"
    elif ds == "ntu": f = f"{PKG}/" + json.load(open(f"{PKG}/data/ntu/init_poses.json"))[seq]["anchor_file"]
    else: f = f"{PKG}/data/util/{seq}/anchors_A.txt"
    A = np.loadtxt(f, comments="#")
    return A[:, -3:] if A.ndim == 2 else A[None, -3:]

def estimate(ds, seq, f, lag):
    if ds != "ntu":
        tr = track(f, lag); return (tr[0], tr[1]) if tr else None
    P = rows(f); th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    E = np.array([X[i] + rot(Q[i]) @ NTU_PRISM for i in range(len(X))])
    lt, lp, _ = gt_pose(ds, seq); k = (th > lt[0] + 10) & (th < lt[-1] - 10) & np.all(np.isfinite(E), 1)
    if k.sum() < 10: return None
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); Ek = E[k]
    gc, ec = G.mean(0), Ek.mean(0); g2, e2 = G[:, :2] - gc[:2], Ek[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return th[k], (Rz.T @ (Ek - (ec - Rz @ gc)).T).T

def plot(ds, seq, labels, root, out):
    gt_t, gp, _ = gt_pose(ds, seq); runs = []
    for l in labels:
        f = f"{root}/{ds}/{l}_{seq}_pose.txt"
        if not os.path.exists(f): continue
        lag = WINDOW[ds] - 1 if os.path.exists(f.replace("_pose.txt", "_now_pose.txt")) else 0
        e = estimate(ds, seq, f, lag)
        if e is not None: runs.append((l, e, position_rmse(ds, seq, f, lag)))
    if not runs: return False
    fig = plt.figure(figsize=(12, 6.5))
    lo, hi = gp.min(0) - 1.0, gp.max(0) + 1.0
    for j, ax_name in enumerate("xyz"):
        ax = fig.add_subplot(3, 2, 2 * j + 1)
        ax.plot(gt_t - gt_t[0], gp[:, j], color="0.4", lw=1.6, label="ground truth")
        for l, (t, X), r in runs:
            ok = np.all((X >= lo) & (X <= hi), 1)
            ax.plot(t - gt_t[0], np.where(ok, X[:, j], np.nan), color=COLOR.get(l), lw=0.8,
                    label=f"{l} {'div.' if r > 100 else f'{r:.3f} m'}")
        ax.set_ylabel(f"{ax_name} [m]"); ax.set_ylim(lo[j], hi[j]); ax.grid(alpha=0.3)
        if j == 0: ax.legend(fontsize=7, ncol=3, loc="upper left")
        if j == 2: ax.set_xlabel("time [s]")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot(*gp.T, color="0.4", lw=1.6)
    for l, (t, X), r in runs:
        Xc = np.where(np.all((X >= lo) & (X <= hi), 1)[:, None], X, np.nan); ax.plot(*Xc.T, color=COLOR.get(l), lw=0.6)
    if ds != "ntu":
        A = anchors(ds, seq); ax.scatter(*A.T, marker="^", color="crimson", s=30)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    fig.suptitle(f"{ds.upper()} {seq}"); fig.tight_layout()
    os.makedirs(out, exist_ok=True); fig.savefig(f"{out}/{ds}_{seq}.png", dpi=130); plt.close(fig)
    return True

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["miluv", "ntu", "util", "iului"])
    ap.add_argument("sequences", nargs="*")
    ap.add_argument("--labels", nargs="+", default=["kims", "swpgo", "pf", "sfuise"])
    ap.add_argument("--root", default=os.path.join(PKG, "result"))
    ap.add_argument("--out", default=os.path.join(PKG, "result/plots"))
    a = ap.parse_args()
    for s in a.sequences or sequences(a.dataset):
        print(f"{a.dataset} {s}: {'written' if plot(a.dataset, s, a.labels, a.root, a.out) else 'no runs'}")

if __name__ == "__main__":
    main()
