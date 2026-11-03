#!/usr/bin/env python3
"""Anchor availability and geometry per evaluated sequence (reviewer R3-1): anchors, anchor height spread, tags, measurement
type, measurements per second, duration. usage: python3 scripts/revision/anchor_table.py > result/final/anchor_table.txt"""
import os, sys, json, csv, numpy as np
sys.path[:0] = ["scripts"]; from datasets import CONFIG, anchor_path
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG)
def rate(path, tcol=0):
    t = []
    with open(path) as f:
        r = csv.reader(f); hdr = next(r)
        ti = hdr.index("timestamp") if "timestamp" in hdr else tcol
        for row in r:
            try: t.append(float(row[ti]))
            except Exception: pass
    t = np.array(t); return len(t), t[-1] - t[0]
MIL = ["default_1_remoteControlHighPace_1", "default_1_random3_1", "default_1_random3_0", "default_1_circular3D_0", "default_1_remoteControlLowPace_0_v2",
       "default_1_remoteControlLowPace_0", "default_1_circular2D_0", "default_1_remoteControlHighPace_0", "obstacles_1_random3_0b", "default_1_random3_2",
       "default_1_remoteControlHighPace_2", "noAprilTags_1_random3_2b", "noAprilTags_1_remoteControlHighPace_2b"]
rows = []
for s in MIL:
    A = np.loadtxt(anchor_path("miluv", s)); n, d = rate(f"data/miluv/{s}/ifo001/uwb_range.csv")
    rows.append(("MILUV", s, len(A), A[:, 2].min(), A[:, 2].max(), 2, "ToA range", n / d, d))
for s in ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]:
    A = np.loadtxt(anchor_path("ntu", s)); n, d = rate(f"data/ntu/{s}/uwb_range.csv")
    rows.append(("NTU VIRAL", s, len(A), A[:, 2].min(), A[:, 2].max(), 4, "ToA range", n / d, d))
for s in ["SquareTurn", "CircleSpin", "AC3_125000"]:
    m = json.load(open(f"data/iului/AC3_125000.json" if s.startswith("AC3") else f"data/iului/AC1_{s}.json")); A = np.array(m["anchors"])
    n, d = rate(f"data/iului/{s}/uwb_range.csv"); rows.append(("IULUI", s, len(A), A[:, 2].min(), A[:, 2].max(), 2, "ToA range", n / d, d))
ut = sorted(os.listdir("data/util")); g = {}
for s in ut:
    A = np.loadtxt(f"data/util/{s}/anchors_A.txt"); B = np.loadtxt(f"data/util/{s}/anchors_B.txt"); U = np.unique(np.vstack([A, B]), axis=0)
    n, d = rate(f"data/util/{s}/uwb_range.csv"); m = json.load(open(f"data/util/{s}/meta.json"))
    key = (s.split("-")[0], "tdoa3" if "tdoa3" in s else "tdoa2"); g.setdefault(key, []).append((len(U), U[:, 2].min(), U[:, 2].max(), m["num_slots"], n / d, d))
print(f"{'dataset':10s} {'sequence':40s} | {'anchors':>7s} {'anchor z min-max [m]':>20s} {'tags':>4s} {'measurement':>12s} {'meas/s':>7s} {'dur [s]':>7s}")
for ds, s, na, zlo, zhi, nt, kind, r, d in rows:
    print(f"{ds:10s} {s:40s} | {na:7d} {f'{zlo:.2f} - {zhi:.2f}':>20s} {nt:4d} {kind:>12s} {r:7.0f} {d:7.0f}")
for (c, mode), v in sorted(g.items()):
    v = np.array(v)
    print(f"{'UTIL':10s} {f'{c} {mode} ({len(v)} flights)':40s} | {int(v[0,0]):7d} {f'{v[0,1]:.2f} - {v[0,2]:.2f}':>20s} {1:4d} {'TDOA ' + str(int(v[0,3])) + ' pairs':>12s} {v[:,4].mean():7.0f} {v[:,5].mean():7.0f}")
print("\nmeas/s = UWB events per second in the converted data (MILUV/NTU/IULUI: one range per event; UTIL: one TDOA per event, before the 50 Hz epoch binning).")
print("Anchor z from the surveyed anchor files; MILUV constellation 2 in the corrected order. Durations are the UWB spans.")
