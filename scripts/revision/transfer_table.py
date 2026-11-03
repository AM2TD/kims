#!/usr/bin/env python3
"""E3 parameter transfer: the final per-dataset settings vs ONE common setting on every dataset (transfer_run.sh, labels tr_):
q_acc 0.040, q_gyr 0.0013, head_filt 1, window 2 s. IULUI already runs the common setting (f32g_). Table scoring as in
final_rmse_final.py.  usage: python3 scripts/revision/transfer_table.py > result/final/transfer_table.txt"""
import os, sys, numpy as np
sys.path[:0] = ["scripts", "scripts/revision"]
exec(open("scripts/revision/final_table_v2.py").read().split("MIL = [")[0])
src = open("scripts/revision/final_table_v2.py").read(); i = src.index("MIL = ["); exec(src[i:src.index("\n", i) + 1])
from util_score import score as us
def u(s, f, lag):
    try: return us(s, f, lag)["rmse"] if os.path.exists(f) else np.nan
    except Exception: return np.nan
NTU = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]; UT = sorted(os.listdir("data/util"))
f = lambda x: "-" if not np.isfinite(x) else ("diverged" if x > 100 else f"{x:.3f}")
CH = {"miluv": "window 3 s -> 2 s (T 229 -> 153)", "ntu": "q_acc 0.014 -> 0.040, q_gyr 0.00061 -> 0.0013, head_filt 0 -> 1", "util": "q_acc 0.179 -> 0.040, q_gyr 0.0089 -> 0.0013"}
print("3D RMSE [m]: final (tuned per dataset) vs the common setting (q_acc 0.040, q_gyr 0.0013, head_filt 1, 2 s window) on every dataset.")
print("Range-noise models stay per dataset. IULUI's final setting is the common one.\n")
summary = []
for ds, seqs in [("miluv", MIL), ("ntu", NTU), ("util", UT)]:
    recs = []
    for s in seqs:
        if ds == "util": a, b = u(s, f"result/util/diag/uf1_{s}_pose.txt", 99), u(s, f"result/util/diag/tr_{s}_pose.txt", 99)
        else: lag = 228 if ds == "miluv" else 117; a, b = sc(ds, s, f"result/{ds}/diag/f32a_{s}_pose.txt", lag), sc(ds, s, f"result/{ds}/diag/tr_{s}_pose.txt", 117 if ds == "ntu" else 152)
        recs.append((s, a, b))
    A = np.array([r[1:] for r in recs], float)
    print(f"{ds.upper()} ({len(recs)}): changed = {CH[ds]}")
    if ds != "util":
        for s, a, b in recs: print(f"   {s:40s} final {f(a):>8s}  common {f(b):>8s}  diff {1e3 * (b - a):+7.1f} mm")
    else:
        for c in ["const1", "const2", "const3", "const4"]:
            for m in ["tdoa2", "tdoa3"]:
                k = np.array([r[0].startswith(c + "-") and m in r[0] for r in recs]); print(f"   {c + ' ' + m + f' ({int(k.sum())})':40s} final {np.median(A[k, 0]):8.3f}  common {np.median(A[k, 1]):8.3f}  (medians)")
    d = A[:, 1] - A[:, 0]
    print(f"   median final {np.median(A[:, 0]):.3f}  common {np.median(A[:, 1]):.3f} | common worse by >1 cm on {int(np.sum(d > 0.01))}, better by >1 cm on {int(np.sum(d < -0.01))}, within 1 cm {int(np.sum(np.abs(d) <= 0.01))} | diverged {int(np.sum(A[:, 1] > 100))}\n")
