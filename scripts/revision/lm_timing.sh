#!/bin/bash
# Standalone per-epoch time of the matched LM (GPU otherwise idle, one run at a time): MILUV HighPace_1, IULUI SquareTurn.
# Compare with the sampler's standalone runs tm_hp / tm_SquareTurn (nsweep_timing.txt, N 8192 row).
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1; R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
$R/iu_run.sh tlm_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 lm:=1 > /dev/null 2>&1
$R/hp_mix.sh tlm_hp default_1_remoteControlHighPace_1 $A T:=229 lm:=1 > /dev/null 2>&1
echo LMTIMING_DONE
