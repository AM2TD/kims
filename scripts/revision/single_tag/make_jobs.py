#!/usr/bin/env python3
"""Single-tag NTU test: one KIMS run per (sequence, tag) with only that tag's ranges
(num_tags:=1), the window held at the 4-tag run's length in seconds (T = 118 * mean 4-tag spacing / mean 1-tag spacing),
everything else as the final NTU runs (w118_eee_*, n_*). Pairs whose tag has a range gap of more than a minute are skipped.
Writes plan.txt (seq tag n_ranges mean_dt T window_s) and jobs.txt next to this file."""
import os, csv, json, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(os.path.dirname(os.path.dirname(H)))
sys.path.insert(0, os.path.join(PKG, "scripts")); from datasets import CONFIG
os.chdir(PKG)
seqs = ["eee_01","eee_02","eee_03","nya_01","nya_02","nya_03","sbs_01","sbs_02","sbs_03"]
tags = [str(t) for t in CONFIG["ntu"]["tag_ids"]]; off = dict(zip(tags, CONFIG["ntu"]["tag_offsets"]))
init = json.load(open("result/runlogs/ntu_init_args.json"))
for s in seqs[3:]:
    init[s] = [x for x in open(f"result/runlogs/ntu_new_{s}.log") if " init:" in x][0].split("init:")[1].strip()
A = "still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
plan, jobs = [], []
for s in seqs:
    R = list(csv.reader(open(f"data/ntu/{s}/uwb_range.csv")))[1:]
    ta = np.array([float(r[0]) for r in R]); dt4 = (ta[-1] - ta[0]) / (len(ta) - 1)
    for tg in tags:
        t = np.array([float(r[0]) for r in R if r[2] == tg]); dt = (t[-1] - t[0]) / (len(t) - 1); T = int(round(118 * dt4 / dt))
        gap = max(np.diff(t).max(), ta[-1] - t[-1], t[0] - ta[0])
        plan.append(f"{s} {tg} {len(t)} {dt:.5f} {T} {T * dt:.3f}")
        if gap > 60: continue
        af = "" if s.startswith("eee") else f" anchor_file:={PKG}/data/ntu/anchors_{s}.txt"
        o = off[tg]
        jobs.append(f"scripts/revision/ntu_run.sh t1_{s}_{tg} {s} T:={T} dt:={dt:.5f} {init[s]} {A}{af} num_tags:=1 tag0_id:={tg} "
                    f"tag0_offset_x:={o[0]} tag0_offset_y:={o[1]} tag0_offset_z:={o[2]}")
open(os.path.join(H, "plan.txt"), "w").write("\n".join(plan) + "\n"); open(os.path.join(H, "jobs.txt"), "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
