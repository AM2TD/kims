#!/usr/bin/env python3
"""Flip-specific metrics for KIMS-MH outputs.

    python3 scripts/eval_flip.py --dataset iului --sequence SquareBob --pose result/iului/SquareBob/kims_mh_pose.txt

Metrics: flip_episodes, wrong_mode_ratio (|h_gt| < h_min excluded), time_to_resolve per crossing,
resolve_correct, ambiguity calibration (reliability table); RMSE as an auxiliary number.
"""
import argparse, csv, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CONFIG, anchor_path, gt_z_bias
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rot(deg, ax):
    r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
    if ax == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_gt(dataset, seq):
    ds = CONFIG[dataset]
    if dataset == "ntu":
        rows = [[float(v) for v in r[:4]] for r in list(csv.reader(open(os.path.join(PKG, ds["data_dir"].format(seq=seq), "leica.csv"))))[1:]]
        d = np.array(rows); al = ds["align"][seq]
        R = rot(al["rz"], "z") @ rot(al["ry"], "y") @ rot(al["rx"], "x")
        return d[:, 0], (R @ d[:, 1:4].T).T + np.array([al["tx"], al["ty"], al["tz"]])
    g = np.loadtxt(os.path.join(PKG, ds["data_dir"].format(seq=seq), "mocap.csv"), delimiter=",", skiprows=1)
    xyz = g[:, 1:4].copy()
    if dataset == "miluv": xyz[:, 2] += gt_z_bias(dataset, seq)
    return g[:, 0], xyz


def anchor_plane(dataset, seq):
    ds = CONFIG[dataset]
    if dataset == "iului":
        import json
        A = np.array(json.load(open(os.path.join(PKG, ds["meta"].format(seq=seq))))["anchors"])
    else:
        A = np.loadtxt(os.path.join(PKG, anchor_path(dataset, seq)))
    c = A.mean(0); _, _, Vt = np.linalg.svd(A - c); n = Vt[-1]
    if n[2] < 0: n = -n
    return c, n


def episodes(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return int((d == 1).sum())


def range_residuals(dataset, seq, sigma_r=0.1):
    """Measured minus predicted range from the ground-truth pose (tag lever arms applied when the GT
    carries orientation; body centre otherwise). Returns (t, residual, anchor index, outlier mask)."""
    ds = CONFIG[dataset]
    ddir = os.path.join(PKG, ds["data_dir"].format(seq=seq))
    u = np.genfromtxt(os.path.join(ddir, "uwb_range.csv"), delimiter=",", names=True)
    if dataset == "iului":
        import json
        meta = json.load(open(os.path.join(PKG, ds["meta"].format(seq=seq))))
        A = np.array(meta["anchors"]); tag_ids = list(ds["tag_ids"]); anc_ids = list(range(len(A)))
        offs = [np.array(meta["tag0_offset"]), np.array(meta["tag1_offset"])]
    else:
        A = np.loadtxt(os.path.join(PKG, anchor_path(dataset, seq))); tag_ids = list(ds["tag_ids"]); anc_ids = list(ds["anchor_ids"])
        offs = [np.array(o) for o in ds["tag_offsets"]]
    gt_t, gxyz = load_gt(dataset, seq)
    has_q = dataset != "ntu"
    if has_q:
        g = np.loadtxt(os.path.join(ddir, "mocap.csv"), delimiter=",", skiprows=1); gq = g[:, 4:8]
    t = u["timestamp"]; m = (t > gt_t[0]) & (t < gt_t[-1]); u = u[m]; t = t[m]
    gp = np.stack([np.interp(t, gt_t, gxyz[:, i]) for i in range(3)], 1)
    if has_q:
        q = np.stack([np.interp(t, g[:, 0], gq[:, i]) for i in range(4)], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
    res = np.full(len(t), np.nan); anc = np.full(len(t), -1)
    for k in range(len(t)):
        try:
            ti = tag_ids.index(int(u["from_id"][k])); ai = anc_ids.index(int(u["to_id"][k]))
        except ValueError:
            continue
        tw = gp[k]
        if has_q:
            x, y, z, w = q[k]
            R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            tw = gp[k] + R @ offs[ti]
        res[k] = u["range"][k] - np.linalg.norm(tw - A[ai]); anc[k] = ai
    ok = ~np.isnan(res)
    return t[ok], res[ok], anc[ok], np.abs(res[ok]) > 3.0 * sigma_r


def plot_xyz(a, t, xyz, gt_t, gxyz, gi, wrong, c, n, d):
    """X/Y/Z versus time: estimate against ground truth, anchor-plane height on z,
    wrong-mode intervals shaded, and the number of active hypotheses when a diag file exists."""
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = gt_t[0]
    tt = t - t0; gtt = gt_t - t0
    try:
        rt, rr, ra, rout = range_residuals(a.dataset, a.sequence, a.sigma_r)
    except Exception as e:   # no range file or metadata: skip the residual row
        print(f"  (range residuals unavailable: {e})"); rt = None
    nrows = 3 + (1 if d is not None else 0) + (1 if rt is not None else 0)
    fig, ax = plt.subplots(nrows, 1, figsize=(11, 2.3 * nrows), sharex=True)
    label = a.label or os.path.basename(a.pose).replace("_pose.txt", "")
    for i, name in enumerate("xyz"):
        ax[i].plot(gtt, gxyz[:, i], "k--", lw=1.0, label="ground truth")
        ax[i].plot(tt, xyz[:, i], color="#B03A2E", lw=1.0, label=label)
        ax[i].set_ylabel(f"{name} [m]")
        for lo, hi in _intervals(tt, wrong):
            ax[i].axvspan(lo, hi, color="#E67E22", alpha=0.18, lw=0)
    if abs(n[2]) > 0.9:   # horizontal anchor plane: draw its height on the z axis
        ax[2].axhline(c[2], color="#2272B4", lw=0.9, ls=":", label="anchor plane")
    err = np.linalg.norm(xyz - gi, axis=1)
    ax[0].set_title(f"{a.dataset}/{a.sequence}  RMSE {np.sqrt(np.mean(err**2)):.3f} m  wrong-mode {wrong.mean():.3f}  (shaded = wrong side)")
    ax[0].legend(loc="upper right", fontsize=8, ncol=3)
    ax[2].legend(loc="upper right", fontsize=8, ncol=3)
    row = 3
    if rt is not None:
        rtt = rt - t0
        for j in range(int(ra.max()) + 1):
            mj = (ra == j) & ~rout
            ax[row].plot(rtt[mj], rr[mj], ".", ms=1.5, alpha=0.5, label=f"anchor {j}")
        ax[row].plot(rtt[rout], rr[rout], "o", ms=3, color="red", mfc="none", label=f"outlier |r|>3σ ({rout.mean()*100:.1f}%)")
        ax[row].axhline(0, color="k", lw=0.5); ax[row].set_ylabel("range res. [m]")
        ax[row].legend(loc="upper right", fontsize=7, ncol=6)
        # mark outlier epochs on the z axis as ticks along the top
        zt = ax[2].get_ylim()
        ax[2].plot(rtt[rout], np.full(rout.sum(), zt[1]), "|", color="red", ms=6, alpha=0.6, label="UWB outlier")
        ax[2].legend(loc="upper right", fontsize=8, ncol=4)
        row += 1
    if d is not None:
        td = d[:, 0] + a.lag - t0
        ax[row].step(td, d[:, 2], where="post", color="#8E44AD", lw=1.0, label="active hypotheses")
        ax[row].plot(td, d[:, 3], color="#6F8A1D", lw=0.8, label="ambiguity")
        ax[row].set_ylim(-0.05, 2.2); ax[row].set_ylabel("n_hyp / amb"); ax[row].legend(loc="upper right", fontsize=8, ncol=2)
    ax[-1].set_xlabel("time [s]")
    fig.tight_layout()
    out = a.plot if a.plot != "auto" else a.pose.replace("_pose.txt", "") + "_xyz.png"
    fig.savefig(out, dpi=130)
    print(f"  plot: {out}")
    if a.show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def _intervals(t, mask):
    out = []; start = None
    for i, m in enumerate(mask):
        if m and start is None: start = t[i]
        if not m and start is not None: out.append((start, t[i])); start = None
    if start is not None: out.append((start, t[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--sequence", required=True)
    ap.add_argument("--pose", required=True); ap.add_argument("--diag", default=None)
    ap.add_argument("--h_min", type=float, default=0.10, help="exclude |h_gt| below this (m)")
    ap.add_argument("--lag", type=float, default=0.0, help="timestamp shift added to the pose file (s)")
    ap.add_argument("--plot", default="auto", help="'auto' (next to the pose file), a png path, or 'none'")
    ap.add_argument("--show", action="store_true", help="also plt.show() when a display is available")
    ap.add_argument("--label", default=None, help="legend label of the estimate")
    ap.add_argument("--sigma_r", type=float, default=0.1, help="UWB range std for the outlier threshold (3 sigma)")
    a = ap.parse_args()
    gt_t, gxyz = load_gt(a.dataset, a.sequence)
    c, n = anchor_plane(a.dataset, a.sequence)
    P = np.loadtxt(a.pose); t = P[:, 0] + a.lag; xyz = P[:, 1:4].copy()
    if a.dataset == "ntu":   # xy origin alignment as in plot.py
        t_ref = max(t[0], gt_t[0])
        xyz[:, :2] -= np.array([np.interp(t_ref, t, xyz[:, i]) for i in range(2)])
        gxyz = gxyz.copy(); gxyz[:, :2] -= np.array([np.interp(t_ref, gt_t, gxyz[:, i]) for i in range(2)])
    m = (t >= gt_t[0]) & (t <= gt_t[-1]); t, xyz = t[m], xyz[m]
    gi = np.stack([np.interp(t, gt_t, gxyz[:, i]) for i in range(3)], 1)
    h_est = (xyz - c) @ n; h_gt = (gi - c) @ n
    valid = np.abs(h_gt) > a.h_min
    wrong = (np.sign(h_est) != np.sign(h_gt)) & valid
    err = np.linalg.norm(xyz - gi, axis=1)
    out = dict(flip_episodes=episodes(wrong), wrong_mode_ratio=float(wrong[valid].mean()) if valid.any() else float("nan"),
               rmse=float(np.sqrt(np.mean(err ** 2))), median_err=float(np.median(err)))
    # onset classification: a jump in z at the first wrong epoch means an output switch, otherwise a drift
    n_switch_onset = n_drift_onset = 0
    for lo, hi in _intervals(t - gt_t[0], wrong):
        i = int(np.searchsorted(t - gt_t[0], lo))
        dz = abs(xyz[i, 2] - xyz[max(i - 1, 0), 2])
        if dz > 0.5: n_switch_onset += 1
        else: n_drift_onset += 1
    print(f"{a.dataset}/{a.sequence}: flip_episodes {out['flip_episodes']} (switch-onset {n_switch_onset}, drift-onset {n_drift_onset})  "
          f"wrong_mode_ratio {out['wrong_mode_ratio']:.3f}  RMSE {out['rmse']:.3f}  median {out['median_err']:.3f}")

    diag_path = a.diag or a.pose.replace("_pose.txt", "_diag.txt")
    d = np.loadtxt(diag_path) if os.path.isfile(diag_path) else None
    if a.plot != "none":
        plot_xyz(a, t, xyz, gt_t, gxyz, gi, wrong, c, n, d)
    if d is not None:
        td = d[:, 0] + a.lag; nh = d[:, 2]; amb = d[:, 3]; cross = d[:, 16]
        # time_to_resolve: for each crossing (from the leader's height), time until n_active returns to 1
        ttr, correct = [], []
        cr_idx = np.where(np.diff(cross) > 0)[0]
        for i in cr_idx:
            j = i
            while j < len(nh) and (nh[j] == 2 or j == i): j += 1
            if j < len(nh):
                ttr.append(td[j] - td[i])
                hg = np.interp(td[j], gt_t, (gxyz - c) @ n); he = np.interp(td[j], t, h_est)
                correct.append(np.sign(hg) == np.sign(he))
        if ttr:
            print(f"  crossings {len(cr_idx)}  time_to_resolve median {np.median(ttr):.2f} s (min {np.min(ttr):.2f}, max {np.max(ttr):.2f})  "
                  f"resolve_correct {np.mean(correct):.2f}  two-hypothesis time {np.mean(nh == 2):.2f}")
        # ambiguity calibration: bin ambiguity, compare with actual wrong-mode rate
        w_d = np.interp(td, t, wrong.astype(float)) > 0.5
        print("  ambiguity calibration (bin: wrong-mode rate, n):", end="")
        for lo, hi in [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01)]:
            mm = (amb >= lo) & (amb < hi)
            if mm.sum(): print(f"  [{lo:.2f},{hi:.2f}) {w_d[mm].mean():.2f} n={mm.sum()}", end="")
        print()


if __name__ == "__main__":
    main()
