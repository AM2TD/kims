import os
"""Window-length sweep figures: metrics versus window length with the derived
bounds, and the z trajectories of three windows against ground truth."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, f"{PKG}/scripts")
import eval_flip as ef
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False})
SEQ = {"hp": "default_1_remoteControlHighPace_1", "r3": "default_1_random3_1"}
LAB = {"hp": "HighPace_1", "r3": "random3_1"}
WIN = [(153, "hfw153"), (191, "hfw191"), (229, "hfw229"), (267, "hfw267"), (305, "hfw305")]


def rows(f):
    return np.array([[float(x) for x in l.split()] for l in open(f) if not l.startswith('#') and len(l.split()) == 8])


def metrics(sfx, T, lab):
    seq = SEQ[sfx]; gt_t, gt = ef.load_gt("miluv", seq); c, n = ef.anchor_plane("miluv", seq)
    P = rows(f"{PKG}/result/miluv/diag/{lab}_{sfx}_pose.txt"); th = P[:-(T - 1), 0]; X = P[T - 1:, 1:4]
    k = (th - gt_t[0] > 10) & (th < gt_t[-1] - 15)
    g = np.stack([np.interp(th[k], gt_t, gt[:, j]) for j in range(3)], 1); e = X[k] - g
    hg = (g - c) @ n; he = (X[k] - c) @ n; w = np.abs(hg) > 0.1
    B = [l.split() for l in open(f"{PKG}/result/miluv/diag/{lab}_{sfx}_bias.txt") if not l.startswith('#')]
    B = np.array([[float(x) for x in b[:16]] for b in B if len(b) >= 16]); hb = len(B) // 4
    return dict(rmse=np.sqrt(np.mean(np.sum(e ** 2, 1))), z=np.sqrt(np.mean(e[:, 2] ** 2)), wrong=np.mean(np.sign(he[w]) != np.sign(hg[w])),
                zc=np.corrcoef(X[k, 2], g[:, 2])[0, 1], ess=np.median(B[hb:, 13]), passes=B[hb:, 15].mean())


# ---- figure 1: metrics vs window length
Tw = np.array([T * 0.0131 for T, _ in WIN])
M = {sfx: [metrics(sfx, T, lab) for T, lab in WIN] for sfx in ("hp", "r3")}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
cols = {"hp": "#e34948", "r3": "#2a78d6"}
for sfx in ("hp", "r3"):
    for a, key, name in zip(ax, ["rmse", "wrong", "zc", "passes"], ["3-D RMSE (m)", "wrong-side fraction", "z correlation", "rollout passes per solve"]):
        a.plot(Tw, [m[key] for m in M[sfx]], "o-", color=cols[sfx], lw=1.4, ms=5, label=LAB[sfx]); a.set_title(name); a.set_xlabel("window (s)")
for a in ax:
    a.axvspan(2.6, 3.7, color="#1baf7a", alpha=0.12, lw=0)
    a.axvline(2.6, color="#1baf7a", lw=1.0, ls="--"); a.axvline(3.7, color="#1baf7a", lw=1.0, ls="--"); a.axvline(4.7, color="#eb7a3a", lw=1.0, ls=":")
ax[3].axhline(14.5 / 4.3, color="#8a8985", lw=0.8, ls=":")
ax[3].text(2.05, 14.5 / 4.3 + 0.4, "real-time budget (passes)", fontsize=8, color="#8a8985")
ax[0].text(2.65, ax[0].get_ylim()[1] * 0.98, "T_mask,z 2.6 s", fontsize=8, color="#1baf7a", va="top")
ax[0].text(3.75, ax[0].get_ylim()[1] * 0.98, "T_rt 3.7 s", fontsize=8, color="#1baf7a", va="top")
ax[0].text(4.75, ax[0].get_ylim()[1] * 0.90, "T_max 4.7 s", fontsize=8, color="#eb7a3a", va="top")
ax[0].legend(frameon=False, loc="lower left")
fig.suptitle("MILUV AC1, pure sampling + filtered head (N 8192), seed 1234: window-length sweep; green band = admissible interval [T_mask,z, T_rt]", y=1.02)
fig.tight_layout(); out = f"{PKG}/result/miluv/diag/win_sweep_hf_metrics.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)

