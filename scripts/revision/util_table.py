#!/usr/bin/env python3
"""UTIL comparison table: KIMS (u1_*, lag T-1 with window flush), SFUISE (sfuise_*, its published UTIL configuration),
the UTIL paper's ESKF (Table 4, 3D only, as published; util_table4.py) and our run of the released devkit ESKF (eskf_*, used only
in the figures). 3D position RMSE against Vicon in the anchor frame, no alignment, whole GT span.
usage: util_table.py [kims_prefix] > result/runlogs/util_table.txt"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from util_score import score; from util_table4 import T4
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
kp = sys.argv[1] if len(sys.argv) > 1 else "u1_"
D = os.path.join(PKG, "result/util/diag")
seqs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PKG, "data/util/*")))
M = [("KIMS", kp, 99), ("LM", "l1_", 99), ("SFUISE", "sfuisebody_", 0), ("devkit", "eskf_", 0)]   # SFUISE outputs the tag; util_sfuise_body.py moves it to the body origin
R = {}
for s in seqs:
    for name, pre, lag in M:
        f = os.path.join(D, f"{pre}{s}_pose.txt")
        done = name not in ("KIMS", "LM") or (os.path.exists(os.path.join(PKG, "result/runlogs", f"{pre}{s}.log")) and "solves in" in open(os.path.join(PKG, "result/runlogs", f"{pre}{s}.log")).read())
        R[(s, name)] = score(s, f, lag) if os.path.exists(f) and done else None
print("UTIL flight dataset, 3D position RMSE [m] vs Vicon (anchor frame, no alignment, whole GT span).")
print("KIMS: lag 2 s fixed-lag smoother (50 Hz epochs, N 8192). SFUISE: 10 s sliding-window spline (published UTIL config,")
print("Release build). ESKF T4: the UTIL paper's ESKF (Table 4, as published, 3D only). devkit: our run of the released devkit")
print("ESKF (causal), used only in the figures. best: lowest 3D of KIMS, LM, SFUISE, ESKF T4. '-' = no output; coverage < 95 % marked with *.\n")
print(f"{'sequence':34s} | " + " | ".join(f"{n:>7s} {'xy':>6s} {'z':>6s}" for n, _, _ in M) + f" | {'ESKF T4':>7s} | best")
def fmt(r):
    if r is None: return f"{'-':>7s} {'':6s} {'':6s}"
    return f"{r['rmse']:7.3f} {r['xy']:6.3f} {r['z']:6.3f}" + ("*" if r["cover"] < 0.95 else "")
for s in seqs:
    k = R[(s, "KIMS")]; v = {n: R[(s, n)]["rmse"] for n, _, _ in M if n != "devkit" and R[(s, n)] is not None}; v["ESKF T4"] = T4[s]
    best = min(v, key=v.get) if v else "-"
    print(f"{s:34s} | " + " | ".join(fmt(R[(s, n)]) for n, _, _ in M) + f" | {T4[s]:7.3f} | {best}")
print("\nSummary by constellation and mode (sequences where every method has output): mean / median 3D RMSE [m] (mean xy RMSE), KIMS best count")
groups = {}
for s in seqs:
    c = s.split("-")[0]; m = "tdoa3" if "tdoa3" in s else "tdoa2"
    if all(R[(s, n)] is not None for n, _, _ in M): groups.setdefault((c, m), []).append(s)
allk = []
for key in sorted(groups):
    ss = groups[key]; line = f"{key[0]} {key[1]} (n={len(ss):2d}) |"
    for n, _, _ in M:
        x = np.array([R[(s, n)]["rmse"] for s in ss]); xy = np.array([R[(s, n)]["xy"] for s in ss]); line += f" {n} {x.mean():6.3f} / {np.median(x):6.3f} (xy {xy.mean():5.3f}) |"
    t4 = np.array([T4[s] for s in ss]); line += f" ESKF T4 {t4.mean():6.3f} / {np.median(t4):6.3f} |"
    wins = sum(1 for s in ss if R[(s, "KIMS")]["rmse"] == min([R[(s, q)]["rmse"] for q in ("KIMS", "LM", "SFUISE")] + [T4[s]])); line += f" KIMS best {wins}/{len(ss)}"
    print(line); allk += ss
line = f"all        (n={len(allk):2d}) |"
for n, _, _ in M:
    x = np.array([R[(s, n)]["rmse"] for s in allk]); xy = np.array([R[(s, n)]["xy"] for s in allk]); line += f" {n} {x.mean():6.3f} / {np.median(x):6.3f} (xy {xy.mean():5.3f}) |"
t4 = np.array([T4[s] for s in allk]); line += f" ESKF T4 {t4.mean():6.3f} / {np.median(t4):6.3f} |"
wins = sum(1 for s in allk if R[(s, "KIMS")]["rmse"] == min([R[(s, q)]["rmse"] for q in ("KIMS", "LM", "SFUISE")] + [T4[s]]))
print(line + f" KIMS best {wins}/{len(allk)}")
for n, _, _ in M:
    x = np.array([R[(s, n)]["rmse"] for s in allk]); zb = np.array([R[(s, n)]["zbias"] for s in allk]); zs = np.array([R[(s, n)]["zstd"] for s in allk])
    print(f"{n}: RMSE > 0.5 m in {int((x > 0.5).sum())} sequences, > 1 m in {int((x > 1).sum())}; mean z offset {zb.mean():+.3f} m "
          f"(range {zb.min():+.3f} .. {zb.max():+.3f}), z error std about that offset {zs.mean():.3f} m")
