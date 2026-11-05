#!/usr/bin/env python3
"""python3 scripts/evaluate.py [labels ...] [--root DIR] [--out FILE]"""
import argparse, csv, json, os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "scripts"))
from run import SEQUENCES, WINDOW, sequences

NTU_PRISM = np.array([-0.293656, -0.012288, -0.273095])
T10 = {'default_1_remoteControlHighPace_1': 0.76667, 'default_1_random3_1': 0.14737, 'default_1_random3_0': 0.12541,
       'default_1_circular3D_0': 0.17514, 'default_1_remoteControlLowPace_0_v2': 0.12665, 'default_1_remoteControlLowPace_0': 0.14266,
       'default_1_circular2D_0': 0.19352, 'default_1_remoteControlHighPace_0': 0.78438, 'obstacles_1_random3_0b': 0.14545,
       'default_1_random3_2': 0.14877, 'default_1_remoteControlHighPace_2': 0.81323, 'noAprilTags_1_random3_2b': 0.13901,
       'noAprilTags_1_remoteControlHighPace_2b': 0.15775}
_T4 = """c1 1 0.108 0.144|c1 2 0.122 0.144|c1 3 0.109 0.123|c1 4 0.111 0.127|c1 5 0.115 0.149|c1 6 0.123 0.135
c2 1 0.103 0.114|c2 2 0.114 0.116|c2 3 0.108 0.117|c2 4 0.094 0.121|c2 5 0.122 0.125|c2 6 0.106 0.117
c3 1 0.245 0.232|c3 2 0.194 0.225|c3 3 0.189 0.214|c3 4 0.215 0.204|c3 5 0.220 0.241|c3 6 0.193 0.211
c3 m1 0.265 -|c3 m2 0.335 -|c3 m3 - 0.340|c3 m4 - 0.355|c4 1.1 0.201 0.207|c4 1.2 0.168 0.163
c4 1.3 0.194 0.190|c4 2.1 0.231 0.221|c4 2.2 0.200 0.219|c4 2.3 0.222 0.238|c4 3.1 0.919 0.742|c4 3.2 0.718 0.558
c4 3.3 0.651 0.674|c4 4.1 0.818 0.759|c4 4.2 0.766 0.845|c4 4.3 0.735 0.838|c4 5.1 0.473 0.432|c4 5.2 0.470 0.435
c4 5.3 0.395 0.443|c4 6.1 0.509 0.623|c4 6.2 0.459 0.618|c4 6.3 0.517 0.578|c4 m1 0.529 -|c4 m2 0.587 -
c4 m3 0.333 -"""
T4 = {}
for e in _T4.replace("\n", "|").split("|"):
    c, q, e2, e3 = e.split(); c = int(c[1])
    for mode, ev in (("tdoa2", e2), ("tdoa3", e3)):
        if ev == "-": continue
        if q.startswith("m"): T4[f"const{c}-trial7-{mode}-manual{q[1:]}"] = float(ev)
        elif c < 4: T4[f"const{c}-trial{q}-{mode}"] = float(ev)
        else: t, j = q.split("."); T4[f"const4-trial{t}-{mode}-traj{j}"] = float(ev)

def rows(f):
    R = [[float(x) for x in l.split()[:8]] for l in open(f) if not l.startswith("#") and len(l.split()) >= 8]
    return np.array(R) if R else np.zeros((0, 8))

def gt_pose(ds, seq):
    if ds == "ntu":
        L = np.array([[float(v) for v in r[:4]] for r in list(csv.reader(open(f"{PKG}/data/ntu/{seq}/leica.csv")))[1:]])
        return L[:, 0], L[:, 1:4], None
    f = {"miluv": f"{PKG}/data/miluv/{seq}/ifo001/mocap.csv", "iului": f"{PKG}/data/iului/{seq}/mocap.csv",
         "util": f"{PKG}/data/util/{seq}/gt_pose.csv"}[ds]
    g = np.loadtxt(f, delimiter=",", skiprows=1)
    return g[:, 0], g[:, 1:4], g[:, 4:8]

def track(f, lag, flush=True):
    P = rows(f)
    if len(P) <= lag: return None
    th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
    ff = f.replace("_pose.txt", "_flush_pose.txt")
    if flush and lag and os.path.exists(ff):
        F = rows(ff); n = len(F) // 2; first, last = F[:n], F[n:]
        mf = first[:, 0] < th[0]; ml = last[:, 0] > th[-1]
        th = np.concatenate([first[mf, 0], th, last[ml, 0]]); X = np.concatenate([first[mf, 1:4], X, last[ml, 1:4]])
        Q = None
    return th, X, Q

def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def position_rmse(ds, seq, f, lag):
    if not os.path.exists(f): return np.nan
    try:
        if ds == "ntu":
            P = rows(f); th = P[:-lag, 0] if lag else P[:, 0]; X = P[lag:, 1:4]; Q = P[lag:, 4:8]
            prism = np.array([X[i] + rot(Q[i]) @ NTU_PRISM for i in range(len(X))])
            lt, lp, _ = gt_pose(ds, seq)
            k = (th > lt[0] + 10) & (th < lt[-1] - 10)
            if k.sum() < 10: return np.nan
            G = np.stack([np.interp(th[k], lt, lp[:, j]) for j in range(3)], 1); E = prism[k]
            if not np.all(np.isfinite(E)): return np.nan
            gc, ec = G.mean(0), E.mean(0); g2, e2 = G[:, :2] - gc[:2], E[:, :2] - ec[:2]
            yaw = np.arctan2(np.sum(g2[:, 0] * e2[:, 1] - g2[:, 1] * e2[:, 0]), np.sum(g2[:, 0] * e2[:, 0] + g2[:, 1] * e2[:, 1]))
            c, s = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            e = E - ((Rz @ G.T).T + ec - Rz @ gc)
            return float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
        tr = track(f, lag)
        if tr is None: return np.inf
        th, X, _ = tr; gt_t, gp, _ = gt_pose(ds, seq)
        k = (th >= gt_t[0]) & (th <= gt_t[-1])
        if k.sum() < 10: return np.nan if ds != "util" else np.inf
        if ds == "util" and not np.all(np.isfinite(X[k])): return np.inf
        e = X[k] - np.stack([np.interp(th[k], gt_t, gp[:, j]) for j in range(3)], 1)
        return float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
    except Exception:
        return np.nan

def orientation_rmse(ds, seq, f, lag):
    if ds == "ntu" or not os.path.exists(f): return np.nan
    from scipy.spatial.transform import Rotation as Rot, Slerp
    tr = track(f, lag, flush=False)
    if tr is None: return np.nan
    t, X, Q = tr; gt_t, gp, gq = gt_pose(ds, seq)
    gt_t, u = np.unique(gt_t, return_index=True); gp, gq = gp[u], gq[u]
    k = (t > gt_t[0]) & (t < gt_t[-1]) & np.all(np.isfinite(Q), 1) & (np.linalg.norm(Q, axis=1) > 0.5)
    if k.sum() < 50: return np.nan
    t, X, Q = t[k], X[k], Q[k]
    pos = np.sqrt(np.mean(np.sum((X - np.stack([np.interp(t, gt_t, gp[:, j]) for j in range(3)], 1)) ** 2, 1)))
    if not np.isfinite(pos) or pos > 100: return np.inf
    R = Slerp(gt_t, Rot.from_quat(gq))(t).inv() * Rot.from_quat(Q)
    E = R.as_matrix() if hasattr(R, "as_matrix") else R.as_dcm()
    ang = np.degrees(np.arccos(np.clip((np.trace(E, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return float(np.sqrt(np.mean(ang ** 2)))

def fmt(x, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "-"
    return "div." if x > 100 else f"{x:.{p}f}"

def summary(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if not len(v): return "-"
    return f"{fmt(float(np.median(v)))} ({int(np.sum(v > 100))}/{int(np.sum(v > 1))})"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--root", default=os.path.join(PKG, "result"))
    ap.add_argument("--out", default=os.path.join(PKG, "result/tables.txt"))
    a = ap.parse_args()
    labels = a.labels or [l for l in ["kims", "swpgo", "pf", "sfuise"]
                          if any(os.path.exists(f"{a.root}/{ds}/{l}_{s}_pose.txt") for ds in SEQUENCES for s in sequences(ds))]
    out = []; P = out.append
    path = lambda ds, l, s: f"{a.root}/{ds}/{l}_{s}_pose.txt"
    lag = lambda ds, l, s: WINDOW[ds] - 1 if os.path.exists(path(ds, l, s).replace("_pose.txt", "_now_pose.txt")) else 0
    pos, ori = {}, {}
    for ds in SEQUENCES:
        for s in sequences(ds):
            for l in labels:
                f = path(ds, l, s); pos[ds, l, s] = position_rmse(ds, s, f, lag(ds, l, s)); ori[ds, l, s] = orientation_rmse(ds, s, f, lag(ds, l, s))
    P(f"Position RMSE [m] (orientation RMSE [deg]) per sequence; 'div.' = above 100 m, '-' = no output. Labels: {', '.join(labels)}")
    for ds in SEQUENCES:
        seqs = sequences(ds)
        if not seqs or all(np.isnan(pos[ds, l, s]) for l in labels for s in seqs): continue
        ref = {"miluv": "EKF+height (paper)", "util": "ESKF (paper)"}.get(ds)
        P(""); P(f"{ds.upper():44s} " + " ".join(f"{l:>16s}" for l in labels) + (f" {ref:>20s}" if ref else ""))
        for s in seqs:
            if ds == "util": continue
            cells = [fmt(pos[ds, l, s]) + ("" if ds == "ntu" else f" ({fmt(ori[ds, l, s], 1)})") for l in labels]
            r = {"miluv": np.sqrt(3) * T10.get(s, np.nan)}.get(ds)
            P(f"  {s:42s} " + " ".join(f"{c:>16s}" for c in cells) + (f" {fmt(r):>20s}" if r is not None else ""))
        if ds == "util":
            for c in range(1, 5):
                g = [s for s in seqs if s.startswith(f"const{c}-")]
                med = lambda d: fmt(float(np.median([x for x in d if not np.isnan(x)]))) if any(not np.isnan(x) for x in d) else "-"
                omed = lambda l: med([ori[ds, l, s] for s in g])
                cells = [med([pos[ds, l, s] for s in g]) + (f" ({float(omed(l)):.1f})" if omed(l) not in ("-", "div.") else "") for l in labels]
                P(f"  {f'const. {c} ({len(g)} flights, median)':42s} " + " ".join(f"{x:>16s}" for x in cells) + f" {med([T4.get(s, np.nan) for s in g]):>20s}")
        meds = [summary([pos[ds, l, s] for s in seqs]) for l in labels]
        r = [np.sqrt(3) * T10[s] for s in seqs] if ds == "miluv" else ([T4[s] for s in seqs if s in T4] if ds == "util" else None)
        P(f"  {'median (diverged/failed)':42s} " + " ".join(f"{m:>16s}" for m in meds) + (f" {summary(r):>20s}" if r else ""))
        if "kims" in labels and len(labels) > 1:
            def lower(l):
                if l == "kims": return ""
                n = sum(1 for s in seqs if not np.isnan(pos[ds, l, s]))
                return f"{sum(1 for s in seqs if pos[ds, 'kims', s] < pos[ds, l, s])}/{n}"
            rl = f"{sum(1 for s, x in zip(seqs, r) if pos[ds, 'kims', s] < x)}/{len(r)}" if r and len(r) == len(seqs) else ""
            P(f"  {'KIMS lower than':42s} " + " ".join(f"{lower(l):>16s}" for l in labels) + (f" {rl:>20s}" if rl else ""))
    P(""); P("Summary (Table III): median position RMSE [m] (diverged > 100 m / failed > 1 m) over the sequences with output")
    P("(UTIL common: the flights on which every listed label has an output, e.g. the 8 flights of Table III)")
    common = [s for s in sequences("util") if all(not np.isnan(pos["util", l, s]) for l in labels)]
    P(f"  {'':12s} " + " ".join(f"{ds.upper():>22s}" for ds in SEQUENCES) + f" {f'UTIL common ({len(common)})':>22s}")
    for l in labels:
        cells = []
        for ds in SEQUENCES:
            v = [pos[ds, l, s] for s in sequences(ds)]; n = sum(1 for x in v if not np.isnan(x))
            cells.append(f"{summary(v)}" + (f" n{n}" if n and n != len(v) else ""))
        cells.append(summary([pos["util", l, s] for s in common]) if common else "-")
        P(f"  {l:12s} " + " ".join(f"{c:>22s}" for c in cells))
    text = "\n".join(out) + "\n"; print(text, end="")
    if a.out: os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w").write(text)

if __name__ == "__main__":
    main()
