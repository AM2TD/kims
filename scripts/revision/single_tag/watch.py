#!/usr/bin/env python3
import os
"""Stop the single-tag NTU batch when any run diverges: last position farther than 60 m horizontally from the anchor
centroid, |z| > 40 m, or non-finite. Kills every estimator whose output file carries the t1_ prefix and the job runner."""
import glob, os, signal, subprocess, sys, time, numpy as np
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); runner_pid = int(sys.argv[1]); pref = "t" + "1_"
def last(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4096)); L = f.read().decode(errors="ignore").splitlines()
    except OSError: return None
    for l in reversed(L):
        p = l.split()
        if len(p) == 8 and not l.startswith("#"):
            try: return np.array([float(x) for x in p[1:4]])
            except ValueError: pass
    return None
def anchors(seq):
    f = os.path.join(PKG, "data/ntu", "anchors_eee.txt" if seq.startswith("eee") else f"anchors_{seq}.txt")
    return np.loadtxt(f, comments="#")
def running():
    out = subprocess.run(["pgrep", "-f", "bin/kims .*" + pref], capture_output=True, text=True).stdout.split()
    return [p for p in out if p != str(os.getpid())]
while True:
    for f in glob.glob(os.path.join(PKG, "result/ntu/diag", pref + "*_pose.txt")):
        b = os.path.basename(f); seq = b[len(pref):len(pref) + 6]; p = last(f)
        if p is None: continue
        c = anchors(seq).mean(0)
        if not np.all(np.isfinite(p)) or np.hypot(*(p[:2] - c[:2])) > 60 or abs(p[2]) > 40:
            for pid in running() + [str(runner_pid)]:
                try: os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError): pass
            subprocess.run(["pkill", "-f", "ntu_run.sh " + pref])
            print(f"[watch] DIVERGED {b} at {np.round(p, 2)} -> batch stopped", flush=True); sys.exit(0)
    try: os.kill(runner_pid, 0); alive = True
    except ProcessLookupError: alive = False
    if not alive and not running():
        print("[watch] batch finished without divergence", flush=True); sys.exit(0)
    time.sleep(10)
