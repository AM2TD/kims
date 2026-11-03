#!/usr/bin/env python3
"""UTIL summary: medians per constellation/mode of 3D and xy RMSE for KIMS, SFUISE (body origin, util_sfuise_body.py) and the
devkit ESKF, win counts, divergence counts, and the 18-flight subset of WRAP (arXiv 2608.09807, tdoa2 trials 4-6).
usage: util_summary.py > result/runlogs/util_summary.txt"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from util_score import score
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seqs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PKG, "data/util/*")))
kp = sys.argv[1] if len(sys.argv) > 1 else "u1_"   # KIMS run prefix (uf1_: FP32 rollout + 8 host threads)
M = [("KIMS", kp, 99), ("LM", "l1_", 99), ("SFUISE", "sfuisebody_", 0), ("ESKF", "eskf_", 0)]
R = {(s, n): score(s, os.path.join(PKG, f"result/util/diag/{p}{s}_pose.txt"), l) for s in seqs for n, p, l in M}
print("UTIL, 79 flights. Medians (robust to SFUISE divergences). 3D / xy position RMSE [m] vs Vicon, no alignment.")
print("SFUISE converted from its tag output to the body origin (util_sfuise_body.py).")
print(f"{'group':14s} {'n':>3s} | {'KIMS 3D':>8s} {'xy':>6s} | {'LM 3D':>8s} {'xy':>6s} | {'SFUISE 3D':>9s} {'xy':>6s} | {'ESKF 3D':>8s} {'xy':>6s} | KIMS<SFUISE 3D, xy | SFUISE >10 m")
for c, m in [(c, m) for c in ["const1", "const2", "const3", "const4"] for m in ["tdoa2", "tdoa3"]] + [("all", "")]:
    ss = [s for s in seqs if (c == "all" or (s.startswith(c + "-") and m in s))]
    med = {n: (np.median([R[(s, n)]["rmse"] for s in ss]), np.median([R[(s, n)]["xy"] for s in ss])) for n, _, _ in M}
    w3 = sum(R[(s, "KIMS")]["rmse"] < R[(s, "SFUISE")]["rmse"] for s in ss); wxy = sum(R[(s, "KIMS")]["xy"] < R[(s, "SFUISE")]["xy"] for s in ss)
    print(f"{(c + ' ' + m):14s} {len(ss):3d} | {med['KIMS'][0]:8.3f} {med['KIMS'][1]:6.3f} | {med['LM'][0]:8.3f} {med['LM'][1]:6.3f} | {med['SFUISE'][0]:9.3f} {med['SFUISE'][1]:6.3f} | "
          f"{med['ESKF'][0]:8.3f} {med['ESKF'][1]:6.3f} | {w3:2d}/{len(ss)}, {wxy:2d}/{len(ss)} | {sum(R[(s, 'SFUISE')]['rmse'] > 10 for s in ss)}")
ok = [s for s in seqs if R[(s, "SFUISE")]["rmse"] < 10]
print(f"\nThe {len(ok)} flights where SFUISE did not diverge (medians):")
for n, _, _ in M:
    g = lambda k: np.median([R[(s, n)][k] for s in ok])
    print(f"  {n:6s} 3D {g('rmse'):.3f} | xy {g('xy'):.3f} | z {g('z'):.3f} | z offset {g('zbias'):+.3f} | z std about the offset {g('zstd'):.3f}")
for n, _, _ in M:
    x = np.array([R[(s, n)]["rmse"] for s in seqs]); print(f"{n}: max 3D RMSE {x.max():.3f} m; > 0.5 m in {int((x > 0.5).sum())}, > 10 m in {int((x > 10).sum())}")
print("\nWRAP subset (tdoa2 trials 4-6, 18 flights), mean 3D RMSE; paper: ESKF 0.116/0.107/0.211/0.487, WRAP 0.065/0.067/0.209/0.327")
for c in ["const1", "const2", "const3", "const4"]:
    ss = [s for s in seqs if s.startswith(c + "-trial4-tdoa2") or s.startswith(c + "-trial5-tdoa2") or s.startswith(c + "-trial6-tdoa2")]
    line = f"  {c} n={len(ss)}"
    for n, _, _ in M:
        v = np.array([R[(s, n)]["rmse"] for s in ss]); good = v < 10
        line += f" | {n} {v[good].mean():.3f}" + (f" ({int((~good).sum())} diverged)" if (~good).any() else "")
    print(line)
