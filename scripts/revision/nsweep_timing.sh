#!/bin/bash
# Standalone solve-time measurement per N (GPU otherwise idle, one run at a time), MILUV HighPace_1 and IULUI SquareTurn.
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1; R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
for N in 4096 2048; do
  $R/hp_mix.sh tmn${N}_hp default_1_remoteControlHighPace_1 $A T:=229 N:=$N > /dev/null 2>&1
  $R/iu_run.sh tmn${N}_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 N:=$N > /dev/null 2>&1
done
python3 - <<'PY' | tee result/runlogs/nsweep_timing.txt
import numpy as np, os
print("Standalone per-epoch processing time [ms] (from after the host-side window assembly to after the solve and its diagnostic writes; pose output not included), RTX 5070 Ti, first 20 solves excluded.")
print(f"{'N':>5s} | {'HighPace_1 (3 s, epoch 13 ms)':>44s} | {'SquareTurn (2 s, epoch 10 ms)':>44s}")
print(f"{'':5s} | {'mean':>7s} {'median':>7s} {'p95':>7s} {'>13ms':>6s} {'passes':>7s} | {'mean':>7s} {'median':>7s} {'p95':>7s} {'>10ms':>6s} {'passes':>7s}")
def st(f, lim, b):
    if not os.path.exists(f): return '   -' * 5
    t = np.loadtxt(f)[20:]
    B = np.array([[float(x) for x in l.split()[:16]] for l in open(b) if not l.startswith('#') and len(l.split()) >= 16])
    return f"{t.mean():7.1f} {np.median(t):7.1f} {np.percentile(t,95):7.1f} {np.mean(t>lim)*100:5.0f}% {B[20:,15].mean():7.2f}"
for N in (8192, 4096, 2048, 1024, 512, 256):
    lh = 'tm_hp' if N == 8192 else f'tmn{N}_hp'; ls_ = 'tm_SquareTurn' if N == 8192 else f'tmn{N}_SquareTurn'
    print(f"{N:5d} | {st(f'result/miluv/diag/{lh}_timing.txt', 13, f'result/miluv/diag/{lh}_bias.txt')} | {st(f'result/iului/diag/{ls_}_timing.txt', 10, f'result/iului/diag/{ls_}_bias.txt')}")
PY
echo NTIMING_DONE
