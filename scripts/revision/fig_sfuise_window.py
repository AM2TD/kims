#!/usr/bin/env python3
"""3D error over time (log scale) of KIMS and SFUISE with its published 10 s window, the window matched to KIMS (3 s MILUV /
2 s NTU) and, where run, the matched window with the wide 1.5 m gate. Shows how the short-window SFUISE estimate drifts away.
usage: python3 scripts/revision/fig_sfuise_window.py [out_png]   (package root; default result/figs/sfuise_window_err.png)"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["scripts", "scripts/revision"]
import err_series as es
os.chdir(es.PKG)
out = sys.argv[1] if len(sys.argv) > 1 else "result/figs/sfuise_window_err.png"
CASES = [("miluv", "default_1_circular3D_0", "f32a_", 228, "w30"), ("miluv", "default_1_remoteControlLowPace_0_v2", "f32a_", 228, "w30"),
         ("miluv", "noAprilTags_1_random3_2b", "f32a_", 228, "w30"), ("ntu", "eee_01", "f32a_", 117, "w20"), ("ntu", "nya_02", "f32a_", 117, "w20"), ("ntu", "sbs_01", "f32a_", 117, "w20")]
SCR = os.environ.get("KIMS_SCRATCH", "result/scratch")   # directory with the window-matched SFUISE outputs
def series(ds, seq, f, lag):
    if not os.path.exists(f): return None
    P = es.rows(f); fin = np.all(np.isfinite(P), 1)
    if not fin.all():   # SFUISE wrote non-finite poses after diverging: score the finite part (the scorer would mark everything)
        g = os.path.join(SCR, "fin_" + os.path.basename(f)); np.savetxt(g, P[fin], fmt="%.9f", header="finite rows of " + f); f = g
    r = es.err_std(ds, seq, f, lag) if ds == "miluv" else es.err_ntu(seq, f, lag)
    if r is None: return None
    t, e = r; return t - t[0], np.linalg.norm(e, axis=1)
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(2, 3, figsize=(12, 5.6)); ax = ax.ravel()
for i, (ds, seq, kl, lag, w) in enumerate(CASES):
    win = "3 s" if w == "w30" else "2 s"
    M = [("KIMS (" + win + ")", f"result/{ds}/diag/{kl}{seq}_pose.txt", lag, "#1f4e9c"), ("SFUISE 10 s (published)", f"result/{ds}/diag/sfuise_{seq}_pose.txt", 0, "#e08a1e"),
         (f"SFUISE {win}", f"result/{ds}/diag/sfuise_{w}_{seq}_pose.txt", 0, "#c0392b"), (f"SFUISE {win}, gate 1.5 m", f"result/{ds}/diag/sfuise_{w}_g15_{seq}_pose.txt", 0, "#7b2cbf")]
    for name, f, lg, col in M:
        s = series(ds, seq, f, lg)
        if s is None: continue
        t, n = s; rm = np.sqrt(np.mean(n ** 2)); ax[i].plot(t, np.maximum(n, 1e-3), color=col, lw=0.9, label=f"{name}: {'diverged' if rm > 100 else f'{rm:.3f} m'}")
    ax[i].set_yscale("log"); ax[i].set_ylim(1e-2, 1e6); ax[i].grid(ls="--", lw=0.4, alpha=0.5); ax[i].set_title(f"{ds.upper()} {seq}", fontsize=9)
    ax[i].legend(fontsize=6.5, loc="upper left", frameon=False); ax[i].set_xlabel("time since GT start (s)")
    if i % 3 == 0: ax[i].set_ylabel("3D error (m)")
fig.suptitle("SFUISE window length: 3D error over time (legend: RMSE over the scored span)", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
