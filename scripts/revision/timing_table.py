#!/usr/bin/env python3
"""Standalone per-update processing time of every method (runs of timing_all.sh) -> result/runlogs/timing_table.txt.
Per sequence: updates per second, mean update interval (the real-time budget), mean / median / p95 / max time per update,
load = mean time / mean interval (< 1 keeps up with the data), share of updates longer than the mean interval.
The first 20 updates (start-up) are dropped. Pooled rows concatenate the sequences of a dataset."""
import sys, os, re, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PKG); O = "result/timing"
MIL = ["default_1_remoteControlHighPace_1", "default_1_random3_0", "default_1_circular2D_0"]; NTU = ["eee_01", "sbs_01"]
UT = ["const1-trial1-tdoa2", "const1-trial1-tdoa3"]
SH = {"default_1_remoteControlHighPace_1": "hp1", "default_1_random3_0": "r30", "default_1_circular2D_0": "c2d", "eee_01": "eee01",
      "sbs_01": "sbs01", "const1-trial1-tdoa2": "u2", "const1-trial1-tdoa3": "u3"}
def col(f, j=0): return np.array([float(l.split()[j]) for l in open(f) if l.strip() and not l.startswith("#")])
def span(f):   # time span and line count of a pose file
    t = col(f, 0); return t[-1] - t[0], len(t)
KP = sys.argv[1] if len(sys.argv) > 1 else "tt_k"   # KIMS label prefix: tt_k (FP64, 1 thread) or f32 (FP32 rollout, 8 threads)
def kims(ds, m, s):
    f = f"result/{ds}/diag/{KP if m == 'k' else 'tt_' + m}_{SH[s]}"; ms = col(f + "_timing.txt"); d, n = span(f + "_pose.txt"); return ms, d / max(n - 1, 1) * 1e3
def sfuise(ds, s):
    ms = col(f"{O}/sfuise_{s}_ms.txt"); return ms, 100.0   # one window per 10 Hz knot
def cpp(pre, s):
    ms = col(f"{O}/{pre}_{s}_timing.txt"); d, n = span(f"{O}/{pre}_{s}_pose.txt"); return ms, d / max(len(ms) - 1, 1) * 1e3
def py(f):
    t = col(f, 0); ms = col(f, 1); return ms, (t[-1] - t[0]) / max(len(t) - 1, 1) * 1e3
ROWS = [("MILUV", "miluv", MIL, [("KIMS", lambda s: kims("miluv", "k", s)), ("LM", lambda s: kims("miluv", "l", s)),
                                  ("SFUISE", lambda s: sfuise("miluv", s)), ("devkit EKF (no height)", lambda s: py(f"{O}/devkit_{s}_ms.txt")),
                                  ("SW-PGO", lambda s: cpp("swpgo", s)), ("PS", lambda s: cpp("ps", s))]),
        ("NTU", "ntu", NTU, [("KIMS", lambda s: kims("ntu", "k", s)), ("LM", lambda s: kims("ntu", "l", s)), ("SFUISE", lambda s: sfuise("ntu", s)),
                             ("ESKF fixed-lag 2 s", lambda s: cpp("ekfl", s)), ("SW-PGO", lambda s: cpp("swpgo", s)), ("PS", lambda s: cpp("ps", s))]),
        ("UTIL", "util", UT, [("KIMS", lambda s: kims("util", "k", s)), ("LM", lambda s: kims("util", "l", s)), ("SFUISE", lambda s: sfuise("util", s)),
                              ("devkit ESKF", lambda s: py(f"{O}/utileskf_{s}_ms.txt"))])]
def stats(ms, bud):
    ms = ms[20:]; return dict(n=len(ms), bud=bud, mean=ms.mean(), med=np.median(ms), p95=np.percentile(ms, 95), mx=ms.max(),
                              load=ms.mean() / bud, over=100.0 * np.mean(ms > bud))
fmt = "{:6s} {:24s} {:36s} | {:>7s} {:>8s} | {:>8s} {:>8s} {:>8s} {:>8s} | {:>5s} {:>6s}"
lines = ["Standalone per-update time [ms] (one process at a time, machine otherwise idle; RTX 5070 Ti, i7-14700K). KIMS: "
         + ("FP32 rollout (rollout_fp32:=1) and 8 host threads (KIMS_OMP_THREADS=8); others single-threaded." if KP != "tt_k" else "FP64 rollout, 1 host thread (OMP/BLAS 1 thread)."),
         "budget = mean interval between updates in the data; load = mean / budget (< 1 keeps up); >bud = share of updates longer",
         "than the budget. KIMS/LM: one solve per UWB epoch; SFUISE: one window solve per 10 Hz knot; ESKF/SW-PGO/PS: per range",
         "event; devkit filters: per prediction/measurement step. First 20 updates dropped.",
         fmt.format("data", "method", "sequence", "updates", "budget", "mean", "median", "p95", "max", "load", ">bud%")]
for dn, ds, seqs, meths in ROWS:
    for name, fn in meths:
        pool_ms, pool_b = [], []
        for s in seqs:
            if name == "SFUISE" and s == "default_1_remoteControlHighPace_1":
                lines.append(f"{dn:6s} {name:24s} {s:36s} | not timed (SFUISE diverges on this flight)"); continue
            try: ms, b = fn(s)
            except Exception as e: lines.append(f"{dn:6s} {name:24s} {s:36s} | missing ({type(e).__name__})"); continue
            st = stats(ms, b); pool_ms.append(ms[20:]); pool_b.append((b, len(ms) - 20))
            lines.append(fmt.format(dn, name, s, str(st["n"]), f"{b:.1f}", f"{st['mean']:.3f}", f"{st['med']:.3f}", f"{st['p95']:.3f}", f"{st['mx']:.2f}", f"{st['load']:.2f}", f"{st['over']:.1f}"))
        if len(pool_ms) > 1:
            ms = np.concatenate(pool_ms); b = sum(x * n for x, n in pool_b) / sum(n for _, n in pool_b)
            lines.append(fmt.format(dn, name, "POOLED", str(len(ms)), f"{b:.1f}", f"{ms.mean():.3f}", f"{np.median(ms):.3f}", f"{np.percentile(ms, 95):.3f}", f"{ms.max():.2f}", f"{ms.mean() / b:.2f}", f"{100 * np.mean(ms > b):.1f}"))
    lines.append("")
for s in NTU:   # the fixed-lag ESKF timing file holds the per-step average; its log has the worst backward pass
    f = f"{O}/ekfl_{s}.log"
    if os.path.exists(f):
        m = re.search(r"max per-step backward pass ([0-9.]+) ms", open(f).read()); lines.append(f"ESKF fixed-lag {s}: worst backward (smoothing) pass {m.group(1) if m else '?'} ms")
open("result/runlogs/timing_table.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
