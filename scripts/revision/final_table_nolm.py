#!/usr/bin/env python3
"""All final 3D RMSE without the matched LM, per dataset and sequence (from final_table_v2.txt, util_table.txt, final_table.txt)
-> result/runlogs/final_rmse_nolm.txt.  MILUV: whole GT span, no alignment; NTU: gt+10..end-10 s, 4-DoF aligned, prism;
UTIL: whole GT span, no alignment (SFUISE converted from its tag output to the body origin, ~10 s output lag);
IULUI: whole GT span, no alignment, baselines are the published runs (not re-run)."""
import os, re, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def num(x): return float(">1e4" != x and x or 1e5)
out = []
def block(title, hdr, rows):
    out.append(title); out.append(f"{'sequence':40s} | " + " ".join(f"{h:>10s}" for h in hdr))
    for s, v in rows: out.append(f"{s:40s} | " + " ".join(("%10.3f" % x if x < 100 else f"{'diverged':>10s}") if np.isfinite(x) else f"{'-':>10s}" for x in v))
    a = np.array([v for _, v in rows], float)
    out.append(f"{'median':40s} | " + " ".join("%10.3f" % np.nanmedian(a[:, j]) for j in range(a.shape[1])))
    out.append(f"{'> 1 m (incl. diverged)':40s} | " + " ".join("%10d" % int(np.nansum(a[:, j] > 1)) for j in range(a.shape[1])))
    out.append(f"{'diverged (> 100 m)':40s} | " + " ".join("%10d" % int(np.nansum(a[:, j] > 100)) for j in range(a.shape[1])))
    out.append(f"{'KIMS lower than this method':40s} | " + " ".join(f"{'':>10s}" if j == 0 else "%7d/%-2d" % (int(np.nansum(a[:, 0] < a[:, j])), len(a)) for j in range(a.shape[1]))); out.append("")
V2 = [l.split() for l in open("result/runlogs/final_table_v2.txt") if re.match(r"^(miluv|ntu)\s", l)]
mil = [(r[1], [num(r[3]), num(r[5]), num(r[6]), num(r[7]), num(r[8])]) for r in V2 if r[0] == "miluv"]   # KIMS SFUISE RefFilt SW-PGO PS
ntu = [(r[1], [num(r[3]), num(r[5]), num(r[6]), num(r[7]), num(r[8])]) for r in V2 if r[0] == "ntu"]
block("MILUV (13 flights) 3D RMSE [m]", ["KIMS", "SFUISE", "devkitEKF", "SW-PGO", "PS"], mil)
block("NTU VIRAL (9 sequences) 3D RMSE [m]", ["KIMS", "SFUISE", "ESKF-lag2s", "SW-PGO", "PS"], ntu)
U = []
for l in open("result/runlogs/util_table.txt"):
    if re.match(r"^const\d", l) and "|" in l and "(n=" not in l:
        p = [x.strip() for x in l.split("|")]; c = [x.split() for x in p[1:5]]
        U.append((p[0], [num(c[0][0]) if c[0] else np.nan, num(c[2][0]) if c[2] else np.nan, num(c[3][0]) if c[3] else np.nan]))
block("UTIL (79 flights) 3D RMSE [m]", ["KIMS", "SFUISE", "devkitESKF"], U)
I = [l.split() for l in open("result/runlogs/final_table.txt") if l.startswith("iului ")]
iul = [(r[1], [num(r[4]), num(r[6]), num(r[7]), num(r[8]), num(r[9])]) for r in I]   # KIMS pubKIMS EKF-RTS SW-PGO PS
block("IULUI AC1 (3 sequences) 3D RMSE [m]  (pubKIMS, EKF-RTS, SW-PGO, PS = published runs, not re-run)", ["KIMS", "pubKIMS", "EKF-RTS", "SW-PGO", "PS"], iul)
open("result/runlogs/final_rmse_nolm.txt", "w").write("\n".join(out)); print("\n".join(out))
