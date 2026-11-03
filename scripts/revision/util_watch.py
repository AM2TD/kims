#!/usr/bin/env python3
"""Stop a UTIL batch as soon as one run diverges: the latest estimate leaves the anchor bounding box enlarged by 2 m
(floor -2 m) or is not finite. Kills every estimator whose output file starts with the batch prefix, and the runner.
usage: util_watch.py <prefix> <runner_pid>"""
import glob, json, os, signal, subprocess, sys, time, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pref, runner = sys.argv[1], int(sys.argv[2])
def last(f):
    try:
        with open(f, "rb") as h:
            h.seek(0, 2); n = h.tell(); h.seek(max(0, n - 4096)); L = h.read().decode(errors="ignore").splitlines()
    except OSError: return None
    for l in reversed(L):
        p = l.split()
        if len(p) >= 8 and not l.startswith("#"):
            try: return np.array([float(x) for x in p[1:4]])
            except ValueError: pass
    return None
def running(): return [p for p in subprocess.run(["pgrep", "-f", "bin/kims .*diag/" + pref], capture_output=True, text=True).stdout.split() if p != str(os.getpid())]
box = {}
while True:
    for f in glob.glob(os.path.join(PKG, "result/util/diag", pref + "*_pose.txt")):
        if f.endswith("_flush_pose.txt") or f.endswith("_now_pose.txt"): continue
        seq = os.path.basename(f)[len(pref):-len("_pose.txt")]
        if seq not in box:
            A = np.loadtxt(os.path.join(PKG, "data/util", seq, "anchors_A.txt")); B = np.loadtxt(os.path.join(PKG, "data/util", seq, "anchors_B.txt"))
            AB = np.vstack([A, B]); lo = AB.min(0) - 2.0; lo[2] = -2.0; box[seq] = (lo, AB.max(0) + 2.0)
        p = last(f)
        if p is None: continue
        if not np.all(np.isfinite(p)) or np.any(p < box[seq][0]) or np.any(p > box[seq][1]):
            for pid in running() + [str(runner)]:
                try: os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError): pass
            subprocess.run(["pkill", "-f", "util_run.sh " + pref])
            print(f"[watch] DIVERGED {os.path.basename(f)} at {np.round(p, 2)} -> batch stopped", flush=True); sys.exit(0)
    try: os.kill(runner, 0); alive = True
    except ProcessLookupError: alive = False
    if not alive and not running(): print("[watch] batch finished without divergence", flush=True); sys.exit(0)
    time.sleep(10)
