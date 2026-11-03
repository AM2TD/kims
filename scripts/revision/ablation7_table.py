#!/usr/bin/env python3
"""E7 ablations on the 33-sequence sweep set, final configuration vs (nt_) no tempering and (ga_) a Gaussian range likelihood
(no Tukey cut; MILUV: Gaussian with the effective std 0.081 m instead of the LOS/NLOS mixture). Table scoring as in
final_rmse_final.py. Also the NTU split of the transfer experiment (tq_: prior only, th_: head_filt only), when present.
usage: python3 scripts/revision/ablation7_table.py > result/final/ablation7_table.txt"""
import os, sys, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("MIL = ["); exec(src[i:src.index("\n", i) + 1])
from util_score import score as us
def u(s, f, lag):
    try: return us(s, f, lag)["rmse"] if os.path.exists(f) else np.nan
    except Exception: return np.nan
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]; IU = ["SquareTurn", "CircleSpin", "AC3_125000"]
UT8 = ["const1-trial1-tdoa2", "const1-trial1-tdoa3", "const2-trial1-tdoa2", "const2-trial1-tdoa3", "const3-trial1-tdoa2", "const3-trial1-tdoa3", "const4-trial1-tdoa2-traj1", "const4-trial1-tdoa3-traj1"]
fm = lambda x: "-" if not np.isfinite(x) else ("diverged" if x > 100 else f"{x:.3f}")
def done(lab, s):   # a run killed mid-way leaves a partial pose file: score only runs whose log reports the solve count
    f = f"result/runlogs/{lab}{s}.log"; return os.path.exists(f) and "solves in" in open(f).read()
def diverged_partial(ds, s, lab):   # a run the watcher killed after leaving the 100 m box: its output exists and left the box
    f = f"result/{ds}/diag/{lab}{s}_pose.txt"
    if not os.path.exists(f): return False
    rows = [l.split() for l in open(f) if not l.startswith("#")]; rows = [r for r in rows if len(r) == 8]
    if len(rows) < 10: return False
    P = np.array(rows, float); return bool(np.max(np.linalg.norm(P[:, 1:4], axis=1)) > 100)
def score(ds, s, lab):
    if not done(lab, s): return 1e5 if diverged_partial(ds, s, lab) else np.nan
    if ds == "util": return u(s, f"result/util/diag/{lab}{s}_pose.txt", 99)
    lag = {"miluv": 228, "ntu": 117, "iului": 199}[ds]; return sc(ds, s, f"result/{ds}/diag/{lab}{s}_pose.txt", lag)
FINAL = {"miluv": "f32a_", "ntu": "f32a_", "util": "uf1_", "iului": "f32g_"}
print("3D RMSE [m], final configuration vs ablations (33-sequence set). nt = no tempering (one sampling pass per epoch);")
print("no cut = the Tukey cut removed (uwb_delta 1e9; MILUV: Gaussian std 0.081 m instead of the LOS/NLOS mixture); MILUV also: no cut with std 0.3 m,")
print("and 'Tukey, no mixture' = the plain robust loss of the other datasets (std 0.081 m, cut 5 sigma) instead of the calibrated mixture.")
print("A diverged run (left the 100 m box) was killed by the batch watcher and is scored on its output up to that point.\n")
for ds, seqs in [("miluv", MIL), ("ntu", NTU), ("util", UT8), ("iului", IU)]:
    cols = ["nt_", "ga_"] + (["ga3_", "ga5_"] if ds == "miluv" else []) + (["tq_", "th_"] if ds == "ntu" else [])
    hdr = {"nt_": "no temper", "ga_": "no cut", "ga3_": "no cut, std 0.3", "ga5_": "Tukey, no mixture", "tq_": "NTU prior->common", "th_": "NTU head_filt 1"}
    print(f"{ds.upper():6s} {'sequence':40s} | {'final':>8s} " + " ".join(f"{hdr[c]:>18s}" for c in cols))
    A = []
    for s in seqs:
        v = [score(ds, s, FINAL[ds])] + [score(ds, s, c) for c in cols]; A.append(v)
        print(f"{'':6s} {s:40s} | {fm(v[0]):>8s} " + " ".join(f"{fm(x):>18s}" for x in v[1:]))
    A = np.array(A, float)
    med = lambda j: np.median(A[np.isfinite(A[:, j]), j]) if np.isfinite(A[:, j]).any() else np.nan
    print(f"{'':6s} {'median':40s} | {med(0):8.3f} " + " ".join(f"{med(j + 1):18.3f}" for j in range(len(cols))))
    for j, c in enumerate(cols):
        d = A[:, j + 1] - A[:, 0]; ok = np.isfinite(d)
        print(f"{'':6s} {hdr[c] + ' vs final':40s} | worse by >1 cm {int(np.sum(d[ok] > 0.01))}, better by >1 cm {int(np.sum(d[ok] < -0.01))}, within 1 cm {int(np.sum(np.abs(d[ok]) <= 0.01))}, diverged {int(np.sum(A[:, j + 1] > 100))} of {int(ok.sum())}")
    print()
