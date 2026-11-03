#!/usr/bin/env python3
"""Stop a protocol batch as soon as any of its runs diverges.

A run counts as diverged when its latest estimated position leaves the anchor bounding box enlarged by
--margin metres (no ground truth needed). The watcher then kills the protocol drivers and the estimator
processes whose output file carries the tag, and exits.

    python3 scripts/watch_divergence.py --dataset miluv --tag b2w3 [--margin 2.0] [--period 10]
"""
import argparse, glob, os, signal, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CONFIG, anchor_path
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def anchors(ds, seq):
    if ds == "iului":
        import json
        return np.array(json.load(open(os.path.join(PKG, CONFIG[ds]["meta"].format(seq=seq))))["anchors"])
    return np.loadtxt(os.path.join(PKG, anchor_path(ds, seq)), comments="#")


def last_position(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
            lines = f.read().decode(errors="ignore").splitlines()
    except OSError:
        return None
    for l in reversed(lines):
        p = l.split()
        if len(p) == 8 and not l.startswith("#"):
            try: return np.array([float(x) for x in p[1:4]])
            except ValueError: continue
    return None


def kill_tag(tag):
    out = subprocess.run(["pgrep", "-f", "--", f"--tag {tag}( |$)"], capture_output=True, text=True).stdout.split()
    out += subprocess.run(["pgrep", "-f", f"protocol/{tag}_s[0-9]+_pose.txt"], capture_output=True, text=True).stdout.split()
    me = str(os.getpid())
    for pid in set(out):
        if pid == me: continue
        try: os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError): pass
    return len(set(out) - {me})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--margin", type=float, default=2.0); ap.add_argument("--period", type=float, default=10.0)
    a = ap.parse_args()
    boxes = {}
    while True:
        files = glob.glob(os.path.join(PKG, "result", a.dataset, "*", "protocol", f"{a.tag}_s*_pose.txt"))
        running = [q for q in subprocess.run(["pgrep", "-f", "--", f"--tag {a.tag}( |$)"], capture_output=True, text=True).stdout.split()
                   if q != str(os.getpid())]                      # the watcher's own command line matches too
        for f in files:
            seq = f.split(os.sep)[-3]
            if seq not in boxes:
                A = anchors(a.dataset, seq); boxes[seq] = (A.min(0) - a.margin, A.max(0) + a.margin)
                boxes[seq][0][2] = min(boxes[seq][0][2], -a.margin)          # the floor is below the anchors
            p = last_position(f)
            if p is None: continue
            lo, hi = boxes[seq]
            if np.any(p < lo) or np.any(p > hi) or not np.all(np.isfinite(p)):
                n = kill_tag(a.tag)
                print(f"[watch] DIVERGED {os.path.relpath(f, PKG)} at position {np.round(p, 2)} -> stopped {n} processes of tag {a.tag}", flush=True)
                return
        if not running and files:
            print(f"[watch] tag {a.tag} finished without divergence", flush=True)
            return
        time.sleep(a.period)


if __name__ == "__main__":
    main()
