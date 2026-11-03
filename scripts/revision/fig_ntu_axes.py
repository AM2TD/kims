#!/usr/bin/env python3
"""x, y, z over time on one NTU VIRAL sequence for KIMS and SFUISE, with the table scoring: Leica prism point, each method moved
into the Leica frame with its own 4-DoF (yaw + translation) fit over the scored span (GT start + 10 s to end - 10 s, shaded
outside), per-axis and 3D RMSE over that span. Bottom panel: 3D error over time.
usage: fig_ntu_axes.py <seq> [out_png] [label=pose_file:lag:color ...]   (default result/figs/ntu_axes_<seq>.png; extra
       methods are drawn after KIMS and SFUISE, e.g. "KIMS 10 s=result/ntu/diag/sweep/w10_eee_01_pose.txt:589:#7b2cbf")"""
import os, sys, csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import err_series as es
os.chdir(es.PKG)
seq = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"result/figs/ntu_axes_{seq}.png"
L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"data/ntu/{seq}/leica.csv")))[1:]]); lt, lp = L[:, 0], L[:, 1:4]
def aligned(f, lag):
    P = es.rows(f); th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    E = np.array([X[i] + es.Rof(Q[i]) @ es.T_BP for i in range(len(X))])
    k = (th > lt[0] + 10) & (th < lt[-1] - 10) & np.all(np.isfinite(E), 1)
    G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); Ek = E[k]
    gc, ec = G.mean(0), Ek.mean(0); g2, e2 = G[:, :2] - gc[:2], Ek[:, :2] - ec[:2]
    yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1])); c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); t = ec - Rz @ gc
    Ghat = (Rz.T @ (E - t).T).T                     # the estimate in the Leica frame
    m = (th >= lt[0]) & (th <= lt[-1]); ks = k[m]
    e = Ghat[m] - np.stack([np.interp(th[m], lt, lp[:, j]) for j in range(3)], 1)
    return th[m], Ghat[m], e, ks
M = [("KIMS", f"result/ntu/diag/f32a_{seq}_pose.txt", 117, "#c0392b"), ("SFUISE", f"result/ntu/diag/sfuise_{seq}_pose.txt", 0, "#e08a1e")]
for spec in sys.argv[3:]:
    lab, rest = spec.split("=", 1); f, lag, col = rest.rsplit(":", 2); M.append((lab, f, int(lag), col))
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]}); t0 = lt[0]
for a in ax: a.axvspan(0, 10, color="0.92", zorder=0); a.axvspan(lt[-1] - 10 - t0, lt[-1] - t0, color="0.92", zorder=0)
for j, n in enumerate("xyz"): ax[j].plot(lt - t0, lp[:, j], "k--", lw=1.3, label="Leica GT (prism)", zorder=4)
summ = []; etop = 0.0
for name, f, lag, col in M:
    th, Gh, e, k = aligned(f, lag)
    r = np.sqrt(np.mean(e[k] ** 2, 0)); r3 = np.sqrt(np.mean(np.sum(e[k] ** 2, 1)))
    summ.append(f"{name}: x {r[0]:.3f}, y {r[1]:.3f}, z {r[2]:.3f}, 3D {r3:.3f} m")
    if r3 < 10: etop = max(etop, np.percentile(np.linalg.norm(e[k], axis=1), 99.5))   # a diverged method does not set the error scale
    for j in range(3): ax[j].plot(th - t0, Gh[:, j], color=col, lw=1.0, label=f"{name} (RMSE {r[j]:.3f} m)", zorder=3)
    ax[3].plot(th - t0, np.linalg.norm(e, axis=1), color=col, lw=0.9, label=f"{name} (3D RMSE {r3:.3f} m)")
for j, n in enumerate("xyz"):
    ax[j].set_ylabel(f"{n} (m)"); ax[j].grid(ls="--", lw=0.4, alpha=0.5); ax[j].legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    lo, hi = lp[:, j].min(), lp[:, j].max(); pad = 0.15 * (hi - lo) + 0.3; ax[j].set_ylim(lo - pad, hi + pad)
ax[3].set_ylabel("3D error (m)"); ax[3].set_ylim(0, 1.3 * etop if etop > 0 else None); ax[3].grid(ls="--", lw=0.4, alpha=0.5); ax[3].legend(loc="upper right", fontsize=8, frameon=False)
ax[3].set_xlabel("time since Leica start (s); grey = outside the scored span")
ax[0].set_title(f"NTU VIRAL {seq}: each method in the Leica frame after its own 4-DoF fit (table scoring)")
fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out); print("\n".join(summ))
