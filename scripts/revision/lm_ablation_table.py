#!/usr/bin/env python3
"""Sampling (KIMS) vs the matched LM (same objective, window, prior and rollout model; mode + Laplace instead of weighted
samples) on all four datasets, final runs, table scoring. MILUV: lag 228 + window flush, whole span; NTU VIRAL: Leica prism,
4-DoF, gt+10 s .. end-10 s; UTIL: whole span; IULUI: lag 199 + flush, whole span. LM runs: MILUV/NTU as in final_table_v2.py
(constellation 2 re-run with the corrected anchors: lmf_*), UTIL l1_*, IULUI lmf_* (final IULUI settings).
usage: python3 scripts/revision/lm_ablation_table.py > result/runlogs/lm_ablation.txt   (package root)"""
import os, sys, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
from util_score import score as us
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("MIL = ["); j = src.index("\n", src.index("lm_lab.update")); exec(src[i:j + 1])
lm_lab.update({"default_1_random3_2": "lmf_default_1_random3_2", "default_1_remoteControlHighPace_2": "lmf_default_1_remoteControlHighPace_2",
               "SquareTurn": "lmf_SquareTurn", "CircleSpin": "lmf_CircleSpin", "AC3_125000": "lmf_AC3_125000"})
def u(s, f, lag):
    try: return us(s, f, lag)["rmse"] if os.path.exists(f) else np.nan
    except Exception: return np.nan
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]; IU = ["SquareTurn", "CircleSpin", "AC3_125000"]
UT = sorted(os.listdir("data/util"))
rows = {"miluv": [(s, sc("miluv", s, f"result/miluv/diag/f32a_{s}_pose.txt", 228), sc("miluv", s, f"result/miluv/diag/{lm_lab[s]}_pose.txt", 228)) for s in MIL],
        "ntu": [(s, sc("ntu", s, f"result/ntu/diag/f32a_{s}_pose.txt", 117), sc("ntu", s, f"result/ntu/diag/{lm_lab[s]}_pose.txt", 117)) for s in NTU],
        "util": [(s, u(s, f"result/util/diag/uf1_{s}_pose.txt", 99), u(s, f"result/util/diag/l1_{s}_pose.txt", 99)) for s in UT],
        "iului": [(s, sc("iului", s, f"result/iului/diag/f32g_{s}_pose.txt", 199), sc("iului", s, f"result/iului/diag/{lm_lab[s]}_pose.txt", 199)) for s in IU]}
print("KIMS (sampling) vs matched LM, 3D position RMSE [m]. Same objective and rollout model; LM = mode + Laplace Gaussian.")
print("'same' = within 1 cm.\n")
for ds in ["miluv", "ntu", "iului"]:
    print(f"{ds.upper():6s} {'sequence':40s} |   KIMS     LM   LM-KIMS")
    for s, k, l in rows[ds]: print(f"{'':6s} {s:40s} | {k:6.3f} {l:6.3f}  {1e3 * (l - k):+7.1f} mm")
    print()
print(f"{'dataset':8s} {'n':>3s} | median KIMS   LM | KIMS lower  LM lower  same | LM worse by >5 cm")
for ds in ["miluv", "ntu", "util", "iului"]:
    a = np.array([(k, l) for _, k, l in rows[ds]], float); ok = np.all(np.isfinite(a), 1); a = a[ok]; d = a[:, 1] - a[:, 0]
    print(f"{ds:8s} {len(a):3d} | {np.median(a[:, 0]):11.3f} {np.median(a[:, 1]):6.3f} | {int(np.sum(d > 0.01)):10d} {int(np.sum(d < -0.01)):9d} {int(np.sum(np.abs(d) <= 0.01)):5d} | "
          + ", ".join(f"{s} ({k:.3f} vs {l:.3f})" for (s, k, l), dd in zip([r for r, o in zip(rows[ds], ok) if o], d) if dd > 0.05))
