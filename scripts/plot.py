#!/usr/bin/env python3
import argparse, csv, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CONFIG

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    "kims":               "#B03A2E",
    "ukf_rts":            "#6F8A1D",
    "ekf_rts":            "#0D2B6E",
    "particle_smoothing": "#2272B4",
    "sw_map":             "#41B3BA",
}
LABELS = {
    "kims": "KIMS", "ukf_rts": "UKF-RTS", "ekf_rts": "EKF-RTS",
    "particle_smoothing": "Particle Smoothing", "sw_map": "SW-MAP",
}
DRAW_ORDER = ["sw_map", "particle_smoothing", "ukf_rts", "ekf_rts", "kims"]
GT_KW = dict(color="black", linewidth=1.1, linestyle="--", alpha=0.9, zorder=5)

PDF_METADATA = {
    "Creator": None, "Producer": None, "CreationDate": None,
    "Title": None, "Author": None, "Subject": None,
}
PNG_METADATA = {"Software": None, "Author": None, "Title": None,
                "Description": None}

matplotlib.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "grid.alpha": 0.5, "figure.dpi": 130, "savefig.bbox": "tight",
})


def save_figure(fig, out):
    fig.savefig(f"{out}.png", metadata=PNG_METADATA)
    fig.savefig(f"{out}.pdf", metadata=PDF_METADATA)


def read_tum(path):
    """TUM file: t x y z qx qy qz qw. Returns (t, xyz, quat) or None."""
    if not os.path.isfile(path):
        return None
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            rows.append([float(v) for v in p[:8]])
    if len(rows) < 2:
        return None
    d = np.array(rows)
    d = d[np.isfinite(d).all(axis=1)]
    return d[:, 0], d[:, 1:4], d[:, 4:8]


def read_gt_csv(path):
    """mocap.csv: timestamp,px,py,pz,qx,qy,qz,qw (header row)."""
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            rows.append([float(v) for v in r[:8]])
    d = np.array(rows)
    return d[:, 0], d[:, 1:4], d[:, 4:8] if d.shape[1] >= 8 else None


def read_leica(path):
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            rows.append([float(v) for v in r[:4]])
    d = np.array(rows)
    return d[:, 0], d[:, 1:4]


def quat_to_rpy(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def quat_angle(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=1))
    return 2 * np.arccos(np.clip(dot, -1, 1))


def rot(deg, ax):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_sequence(dataset, seq):
    res = os.path.join(PKG, "result", dataset, seq)
    ds_cfg = CONFIG.get(dataset, {})
    if dataset == "ntu":
        gt_t, L = read_leica(os.path.join(res, "leica.csv"))
        al = ds_cfg["align"][seq]
        R = rot(al["rz"], "z") @ rot(al["ry"], "y") @ rot(al["rx"], "x")
        gt_xyz = (R @ L.T).T + np.array([al["tx"], al["ty"], al["tz"]])
        gt_q = None
    else:
        gt_t, gt_xyz, gt_q = read_gt_csv(os.path.join(res, "mocap.csv"))
        if dataset == "miluv":
            gt_xyz = gt_xyz.copy()
            gt_xyz[:, 2] += ds_cfg.get("gt_z_bias", 0.0)

    methods = {}
    for m in DRAW_ORDER:
        d = read_tum(os.path.join(res, f"{m}_pose.txt"))
        if d is None:
            continue
        t, xyz, q = d
        if m == "kims":
            t = t + ds_cfg.get("lag_shift", 0.0)
        methods[m] = (t, xyz, q)
    if not methods:
        return None

    if dataset == "ntu":
        t_ref = max(max(t[0] for t, _, _ in methods.values()), gt_t[0])
        for m, (t, xyz, q) in methods.items():
            xy0 = np.array([np.interp(t_ref, t, xyz[:, i]) for i in range(2)])
            xyz = xyz.copy()
            xyz[:, :2] -= xy0
            keep = t >= t_ref
            methods[m] = (t[keep], xyz[keep], q[keep])
        gxy0 = np.array([np.interp(t_ref, gt_t, gt_xyz[:, i]) for i in range(2)])
        gt_xyz = gt_xyz.copy()
        gt_xyz[:, :2] -= gxy0
        keep = gt_t >= t_ref
        gt_t, gt_xyz = gt_t[keep], gt_xyz[keep]

    t0 = gt_t[0]
    gt_t = gt_t - t0
    methods = {m: (t - t0, xyz, q) for m, (t, xyz, q) in methods.items()}
    return dict(gt=(gt_t, gt_xyz, gt_q), methods=methods)


def errors(data):
    gt_t, gt_xyz, gt_q = data["gt"]
    out = {}
    for m, (t, xyz, q) in data["methods"].items():
        msk = (t >= gt_t[0]) & (t <= gt_t[-1])
        if msk.sum() < 2:
            continue
        gi = np.stack([np.interp(t[msk], gt_t, gt_xyz[:, i]) for i in range(3)], 1)
        pe = np.linalg.norm(xyz[msk] - gi, axis=1)
        oe = None
        if gt_q is not None and q is not None:
            gq = np.stack([np.interp(t[msk], gt_t, gt_q[:, i]) for i in range(4)], 1)
            gq /= np.linalg.norm(gq, axis=1, keepdims=True)
            oe = quat_angle(q[msk], gq)
        out[m] = (pe, oe)
    return out


def fig_boxplot(data, err, title, out):
    have_ori = any(oe is not None for _, oe in err.values())
    nrows = 2 if have_ori else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(5.2, 2.2 * nrows), squeeze=False)
    ms = [m for m in DRAW_ORDER if m in err][::-1]
    for ri, key in enumerate(["pos", "ori"][:nrows]):
        ax = axes[ri][0]
        vals, cols, labs = [], [], []
        for m in ms:
            v = err[m][0] if key == "pos" else err[m][1]
            if v is None:
                continue
            vals.append(v)
            cols.append(COLORS[m])
            labs.append(LABELS[m])
        bp = ax.boxplot(vals, vert=True, showfliers=False, whis=(10, 90),
                        widths=0.5, patch_artist=True, labels=labs)
        for patch, med, col in zip(bp["boxes"], bp["medians"], cols):
            patch.set(facecolor="white", edgecolor=col, linewidth=1.4)
            med.set(color=col, linewidth=1.8)
        for part in ("whiskers", "caps"):
            for i, artist in enumerate(bp[part]):
                artist.set(color=cols[i // 2], linewidth=0.9)
        ax.set_ylabel("(m)" if key == "pos" else "(rad)")
        ax.tick_params(axis="x", rotation=12)
    axes[0][0].set_title(title)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def fig_traj3d(data, title, out):
    gt_t, gt_xyz, _ = data["gt"]
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[::5, 0], gt_xyz[::5, 1], gt_xyz[::5, 2],
            label="Ground truth", **GT_KW)
    for m in DRAW_ORDER:
        if m not in data["methods"]:
            continue
        t, xyz, _ = data["methods"][m]
        zo = 10 if m == "kims" else 3
        ax.plot(xyz[::5, 0], xyz[::5, 1], xyz[::5, 2], color=COLORS[m],
                linewidth=0.9, alpha=0.9, zorder=zo, label=LABELS[m])
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, out)
    plt.close(fig)


def fig_states(data, title, out):
    gt_t, gt_xyz, gt_q = data["gt"]
    have_ori = gt_q is not None
    rows = [("x (m)", 0), ("y (m)", 1), ("z (m)", 2)]
    if have_ori:
        rows += [("roll (rad)", 3), ("pitch (rad)", 4), ("yaw (rad)", 5)]
        gt_rpy = quat_to_rpy(gt_q)
    ncol = 2 if have_ori else 1
    nrow = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.6), squeeze=False)
    for k, (lab, dim) in enumerate(rows):
        ax = axes[k % 3][k // 3]
        if dim < 3:
            ax.plot(gt_t[::5], gt_xyz[::5, dim], **GT_KW)
        else:
            ax.plot(gt_t[::5], gt_rpy[::5, dim - 3], **GT_KW)
        for m in DRAW_ORDER:
            if m not in data["methods"]:
                continue
            t, xyz, q = data["methods"][m]
            zo = 10 if m == "kims" else 3
            if dim < 3:
                ax.plot(t[::5], xyz[::5, dim], color=COLORS[m], lw=0.8,
                        alpha=0.9, zorder=zo)
            elif q is not None:
                ax.plot(t[::5], quat_to_rpy(q)[::5, dim - 3], color=COLORS[m],
                        lw=0.8, alpha=0.9, zorder=zo)
        ax.set_ylabel(lab)
        if k % 3 == 2:
            ax.set_xlabel("t (s)")
    handles = [plt.Line2D([], [], label="Ground truth", **GT_KW)]
    handles += [plt.Line2D([], [], color=COLORS[m], lw=1.6, label=LABELS[m])
                for m in DRAW_ORDER if m in data["methods"]]
    fig.suptitle(title, y=1.0)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--sequence", default="all")
    args, _ = ap.parse_known_args()

    res_root = os.path.join(PKG, "result")
    datasets = sorted(os.listdir(res_root)) if args.dataset == "all" else [args.dataset]
    unknown = [d for d in datasets if d not in CONFIG]
    datasets = [d for d in datasets if d in CONFIG]
    for d in unknown:
        print(f"[plot] result/{d}: not a known dataset, skipped")
    if not datasets:
        sys.exit(f"[plot] no results found under {res_root} — run scripts/run.py first")

    summary = []
    for dataset in datasets:
        droot = os.path.join(res_root, dataset)
        if not os.path.isdir(droot):
            continue
        known = CONFIG[dataset]["sequences"]
        seqs = ([s for s in sorted(os.listdir(droot)) if s in known]
                if args.sequence == "all" else [args.sequence])
        for seq in seqs:
            if not os.path.isdir(os.path.join(droot, seq)):
                continue
            if seq not in known:   
                continue
            data = load_sequence(dataset, seq)
            if data is None:
                print(f"[plot] {dataset}/{seq}: no pose files, skipped")
                continue
            err = errors(data)
            diverged = [m for m, (pe, _) in err.items()
                        if math.sqrt(float(np.mean(pe ** 2))) > 50.0]
            for m in diverged:
                del data["methods"][m]
                del err[m]
                summary.append((dataset, seq, LABELS[m], float("nan"), None))
                print(f"[plot] {dataset}/{seq}: {LABELS[m]} diverged "
                      f"(> 50 m RMSE) — excluded from figures")
            pdir = os.path.join(droot, seq, "plots")
            os.makedirs(pdir, exist_ok=True)
            title = f"{dataset.upper()} / {seq}"
            fig_boxplot(data, err, title, os.path.join(pdir, "boxplot"))
            fig_traj3d(data, title, os.path.join(pdir, "traj3d"))
            fig_states(data, title, os.path.join(pdir, "states"))
            for m in DRAW_ORDER:
                if m in err:
                    pe, oe = err[m]
                    summary.append((dataset, seq, LABELS[m],
                                    math.sqrt(float(np.mean(pe ** 2))),
                                    math.sqrt(float(np.mean(oe ** 2))) if oe is not None else None))
            print(f"[plot] {dataset}/{seq}: plots -> {pdir}")

    if summary:
        print("\n  {:6s} {:38s} {:20s} {:>9s} {:>9s}".format(
            "data", "sequence", "method", "pos[m]", "ori[rad]"))
        for d, s, m, p, o in summary:
            ps = "diverged" if math.isnan(p) else f"{p:9.3f}"
            print("  {:6s} {:38s} {:20s} {:>9s} {:>9s}".format(
                d, s, m, ps, f"{o:.3f}" if o is not None else "-"))


if __name__ == "__main__":
    main()
