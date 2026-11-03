#!/usr/bin/env python3
"""3D trajectory overlay, one flight per dataset: ground truth, KIMS, SFUISE, the anchors and the best-fit anchor plane
(reviewer R5-2 / R2-3). NTU VIRAL: Leica prism point in the UWB anchor frame (dataset transform), estimates moved to the prism.
usage: python3 scripts/revision/fig_traj3d_final.py [out_png]   (package root; default result/figs/fig_traj3d.png)"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es, eval_flip as ef
from datasets import anchor_path
os.chdir(es.PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/fig_traj3d.png"
T_BP = np.array([-0.293656, -0.012288, -0.273095])
CASES = [("miluv", "default_1_random3_0", "f32a_", 228, "sfuise_"), ("ntu", "eee_01", "f32a_", 117, "sfuise_"),
         ("util", "const4-trial1-tdoa2-traj1", "uf1_", 99, "sfuisebody_"), ("iului", "CircleSpin", "f32g_", 199, "sfuise_")]
def est(ds, f, lag):
    P = es.rows(f); th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]
    if ds == "ntu": X = np.array([X[i] + es.Rof(P[lag + i, 4:8]) @ T_BP for i in range(len(X))])
    k = np.all(np.isfinite(X), 1) & (np.linalg.norm(X, axis=1) < 200); return th[k], X[k]
def gt(ds, s):
    if ds == "util": g = np.loadtxt(f"data/util/{s}/gt.csv", delimiter=",", skiprows=1); return g[:, 0], g[:, 1:4]
    return ef.load_gt(ds, s)
def anchors(ds, s):
    if ds == "util": return np.unique(np.vstack([np.loadtxt(f"data/util/{s}/anchors_A.txt"), np.loadtxt(f"data/util/{s}/anchors_B.txt")]), axis=0)
    if ds == "iului": return np.array(json.load(open(f"data/iului/AC3_125000.json" if s.startswith("AC3") else f"data/iului/AC1_{s}.json"))["anchors"])
    return np.loadtxt(anchor_path(ds, s))
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig = plt.figure(figsize=(11, 8.6))
for i, (ds, s, kl, lag, sl) in enumerate(CASES):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    gt_t, G = gt(ds, s); tk, K = est(ds, f"result/{ds}/diag/{kl}{s}_pose.txt", lag); ts, S = est(ds, f"result/{ds}/diag/{sl}{s}_pose.txt", 0)
    m = (tk >= gt_t[0]) & (tk <= gt_t[-1]); K = K[m]; m = (ts >= gt_t[0]) & (ts <= gt_t[-1]); S = S[m]
    A = anchors(ds, s); c = A.mean(0); _, _, Vt = np.linalg.svd(A - c); n = Vt[-1]
    allp = np.vstack([G, K, S, A]); lo, hi = allp.min(0), allp.max(0); pad = 0.05 * (hi - lo).max()
    xs = np.array([lo[0] - pad, hi[0] + pad]); ys = np.array([lo[1] - pad, hi[1] + pad])
    corners = np.array([[x, y, c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]] for x in xs for y in ys])[[0, 1, 3, 2]]
    ax.add_collection3d(Poly3DCollection([corners], facecolor="#888888", alpha=0.18, edgecolor="#666666", linewidth=0.6))
    ax.plot(G[:, 0], G[:, 1], G[:, 2], color="black", lw=1.0, label="ground truth")
    ax.plot(K[:, 0], K[:, 1], K[:, 2], color="#1f4e9c", lw=0.9, label="KIMS")
    ax.plot(S[:, 0], S[:, 1], S[:, 2], color="#e08a1e", lw=0.9, label="SFUISE")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], color="#c0392b", marker="^", s=40, depthshade=False, label="UWB anchors", zorder=5)
    rng = np.maximum(hi - lo, 1e-3); ax.set_box_aspect(rng / rng.max())
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_xlabel("x (m)", labelpad=2); ax.set_ylabel("y (m)", labelpad=2); ax.set_zlabel("z (m)", labelpad=2)
    rk = np.sqrt(np.mean(np.sum((K - np.stack([np.interp(tk[(tk >= gt_t[0]) & (tk <= gt_t[-1])], gt_t, G[:, j]) for j in range(3)], 1)) ** 2, 1)))
    ax.set_title(f"{ds.upper()} {s}  (anchor z {A[:, 2].min():.2f}-{A[:, 2].max():.2f} m)", fontsize=9)
    ax.view_init(elev=22, azim=-55); ax.locator_params(nbins=5)
    if i == 0: ax.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.suptitle("Ground truth, KIMS and SFUISE with the UWB anchors and the best-fit anchor plane (grey)", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
