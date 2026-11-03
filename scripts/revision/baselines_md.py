#!/usr/bin/env python3
"""Final baseline comparison page (result/final/baselines_rmse.md) built from result/runlogs/final_rmse_final.txt
(final_rmse_final.py) and result/runlogs/util_table.txt; published UTIL subsets (He & Chan TVT 2026, Li et al. Measurement 2026
STG-AKF) are entered as reported. Run from the package root."""
import re, numpy as np
L = [l.rstrip("\n") for l in open("result/runlogs/final_rmse_final.txt")]
def seqrows(ds):
    out = []
    for l in L:
        p = [x.strip() for x in l.split("|")]
        if p[0].startswith(ds + " ") and not re.search(r" (median|>1 m|KIMS lower|reference)", p[0]) and not p[0].startswith(f"{ds} const"):
            out.append((p[0].split(" ", 1)[1], p[1:]))
    return out
def stat(ds, key):
    for l in L:
        if l.startswith(f"{ds} {key}"): return [x.strip() for x in l.split("|")][1:]
def num(x): return np.inf if x == "diverged" else (np.nan if x == "-" else float(x))
fmt = lambda x: x if x in ("-", "diverged") else f"{float(x):.3f}"
def row(name, vals):
    v = [num(x) for x in vals]; b = int(np.nanargmin(v))
    return f"| {name} | " + " | ".join(f"**{fmt(x)}**" if i == b else fmt(x) for i, x in enumerate(vals)) + " |"
U = {}
for l in open("result/runlogs/util_table.txt"):
    m = re.match(r"^(const\S+)\s+\|\s+([\d.]+)\s", l)
    if m: U[m.group(1)] = float(m.group(2))
med = {ds: [x.split()[-1] for x in stat(ds, "median")] for ds in ("miluv", "ntu", "util", "iului")}
low = {ds: [x.split()[-1] for x in stat(ds, "KIMS lower than")] for ds in ("miluv", "ntu", "util", "iului")}
div = {ds: [int(re.search(r"\((\d+)\)", x).group(1)) for x in stat(ds, ">1 m")] for ds in ("miluv", "ntu", "util", "iului")}
ref = {ds: next(l for l in L if l.startswith(f"{ds} reference")) for ds in ("miluv", "util")}
o = ["# KIMS vs baselines: 3D position RMSE (m)\n",
     "KIMS uses IMU + UWB only (final configuration, fixed-lag output, seed 1234). Lower is better; **bold** = best per row.\n",
     "## Summary (median over sequences)\n",
     "| Dataset (sequences) | KIMS | SFUISE | Dataset filter(s), as published | KIMS lower than |", "|---|---|---|---|---|",
     f"| MILUV (13) | **{med['miluv'][0]}** | {med['miluv'][1]} ({div['miluv'][1]} diverged) | EKF + height {med['miluv'][2]} | SFUISE {low['miluv'][0]}; EKF + height {low['miluv'][1]} |",
     f"| NTU VIRAL (9) | **{med['ntu'][0]}** | {med['ntu'][1]} ({div['ntu'][1]} diverged) | none | SFUISE {low['ntu'][0]} |",
     f"| UTIL (79) | {med['util'][0]} | **{med['util'][1]}** ({div['util'][1]} diverged) | ESKF {med['util'][2]} | SFUISE {low['util'][0]}; ESKF {low['util'][1]} |",
     f"| IULUI (3) | **{med['iului'][0]}** | {med['iului'][1]} | none | SFUISE {low['iului'][0]} |\n",
     "Diverged = RMSE above 100 m (counted in the medians). KIMS diverges on none of the 104 sequences.\n",
     "## MILUV (raw mocap ground truth, whole span, no alignment)\n",
     "| Experiment | KIMS | SFUISE | EKF + height (MILUV paper Table 10 x sqrt3) |", "|---|---|---|---|"]
o += [row(s, v) for s, v in seqrows("miluv")] + [f"| **median** | **{med['miluv'][0]}** | {med['miluv'][1]} | {med['miluv'][2]} |\n"]
o += ["## NTU VIRAL (Leica prism, 4-DoF alignment, start+10 s to end-10 s)\n", "| Sequence | KIMS | SFUISE |", "|---|---|---|"]
o += [row(s, v) for s, v in seqrows("ntu")] + [f"| **median** | **{med['ntu'][0]}** | {med['ntu'][1]} |\n"]
o += ["## UTIL (Vicon, whole span, no alignment): medians per constellation and TDOA mode\n",
      "| Group (flights) | KIMS | SFUISE | ESKF (UTIL paper Table 4) |", "|---|---|---|---|"]
for l in L:
    m = re.match(r"^util (const\d) (tdoa\d) n=(\d+) \| (\S+) \| (\S+) \| (\S+)", l)
    if m: o.append(row(f"{m.group(1)} {m.group(2)} ({m.group(3)})", [m.group(4), m.group(5), m.group(6)]))
o += [f"| **all (79)** | {med['util'][0]} | **{med['util'][1]}** | {med['util'][2]} |\n"]
o += ["## IULUI (Qualisys, whole span, no alignment)\n", "| Flight | KIMS | SFUISE |", "|---|---|---|"]
o += [row(s, v) for s, v in seqrows("iului")] + [f"| **median** | **{med['iului'][0]}** | {med['iului'][1]} |\n"]
o += ["## Published results on UTIL subsets (as reported; tdoa2 flights)\n",
      "He & Chan, IEEE TVT 2026 (training-free RAF-ESKF / AF-ESKF; LAF-ESKF is trained on ground truth):\n",
      "| Flight | He & Chan | KIMS |", "|---|---|---|"]
for s, meth, v in [("const4-trial3-tdoa2-traj1", "RAF-ESKF", 0.290), ("const4-trial3-tdoa2-traj2", "RAF-ESKF", 0.294), ("const4-trial3-tdoa2-traj3", "RAF-ESKF", 0.308),
                   ("const4-trial5-tdoa2-traj1", "RAF-ESKF", 0.303), ("const4-trial5-tdoa2-traj2", "RAF-ESKF", 0.246), ("const4-trial5-tdoa2-traj3", "RAF-ESKF", 0.243),
                   ("const2-trial1-tdoa2", "AF-ESKF (LAF-ESKF, trained: 0.0445)", 0.0995), ("const2-trial4-tdoa2", "AF-ESKF (LAF-ESKF, trained: 0.0473)", 0.0918)]:
    k = U[s]; kb = k < v
    o.append(f"| {s} | {'' if kb else '**'}{v:.4g}{'' if kb else '**'} {meth} | {'**' if kb else ''}{k:.3f}{'**' if kb else ''} |")
o += ["\nLi et al., Measurement 2026 (STG-AKF, learned from ground-truth bias labels on the other const4 scenarios / trajectories; its ESKF baseline is the UTIL Table 4 ESKF):\n",
      "| const4 tdoa2 flights | STG-AKF cross-scenario | STG-AKF cross-trajectory | KIMS | ESKF (UTIL paper Table 4) |", "|---|---|---|---|---|",
      "| scenarios 1-6 = trials 1-6 (18), mean | **0.184** | 0.230 | 0.318 | 0.480 |",
      "| scenario 7 = trial 7 manual, held out (3), mean | 0.314 | - | **0.207** | 0.483 |\n",
      "## Conditions\n",
      "- KIMS: IMU + UWB; fixed-lag output (3 s MILUV, 2 s others); FP32 rollout, 8 host threads, N 8192.",
      "- SFUISE: the authors' published configuration (0.3 m gate, 10 s window), same start pose as KIMS; our runs.",
      "- MILUV filters: the MILUV paper's own EKF with the laser height sensor (its Table 10), started from the ground-truth state; the paper's per-axis RMS multiplied by sqrt(3). Constellation-2 values are as published (devkit anchor order); KIMS and SFUISE use the corrected order.",
      "- UTIL filter: the UTIL paper's ESKF (Table 4, per flight, as published). The figures need per-epoch errors, which the paper does not give, so they use our run of the released devkit ESKF; it matches Table 4 within 0.01 m on 45/79 flights and is 0.03-0.31 m lower (better) on the 27 const4 trial 3-7 flights.",
      f"- References (not in the table): {ref['miluv'].split('reference: ')[1]} (MILUV); {ref['util'].split('reference: ')[1]} (UTIL).",
      "- Per-flight UTIL values: result/final/util_table.txt."]
open("result/final/baselines_rmse.md", "w").write("\n".join(o) + "\n"); print("\n".join(o[:12]))
