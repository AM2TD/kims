#!/usr/bin/env python3
"""E9: the first/second-submission KIMS runs (result/<ds>/<seq>/kims_pose.txt, Aug 2026, old configuration, lag 15) scored
with the final protocol, next to the final runs, so that the change in the reported numbers splits into 'same run, new
scoring' and 'new configuration'. Final protocol: MILUV/IULUI whole GT span, raw mocap, no alignment; NTU Leica prism,
4-DoF, gt+10 s .. end-10 s.   usage: python3 scripts/revision/rescore_v2_runs.py > result/final/rescore_v2_runs.txt"""
import os, sys, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("MIL = ["); exec(src[i:src.index("\n", i) + 1])
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
f = lambda x: "-" if not np.isfinite(x) else ("diverged" if x > 100 else f"{x:.3f}")
print(f"{'dataset':6s} {'sequence':40s} | {'v2 run, final scoring':>22s} {'final run':>10s} | v2 file date")
for ds, seqs, lab, lag in [("miluv", MIL, "f32a_", 228), ("ntu", NTU, "f32a_", 117), ("iului", ["SquareTurn", "CircleSpin"], "f32g_", 199)]:
    for s in seqs:
        old = f"result/{ds}/{s}/kims_pose.txt"; new = f"result/{ds}/diag/{lab}{s}_pose.txt"
        if not os.path.exists(old): print(f"{ds:6s} {s:40s} | {'no v2 run':>22s} {f(sc(ds, s, new, lag)):>10s} |"); continue
        import datetime; d = datetime.date.fromtimestamp(os.path.getmtime(old)).isoformat()
        print(f"{ds:6s} {s:40s} | {f(sc(ds, s, old, 15)):>22s} {f(sc(ds, s, new, lag)):>10s} | {d}")
print("\nv2 runs: lag 15 epochs (the old window), no window flush. MILUV constellation 2 v2 runs used the devkit anchor order.")
