#!/bin/bash
# Window-length sweep of the current configuration (filtered head), MILUV, seed 1234.
cd "$(dirname "$0")/../.."
R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1"
for T in 153 191 229 267 305; do
  $R/hp_mix.sh hfw${T}_hp default_1_remoteControlHighPace_1 $A T:=$T > /dev/null 2>&1 &
  $R/hp_mix.sh hfw${T}_r3 default_1_random3_1 $A T:=$T > /dev/null 2>&1 &
done
wait
echo WIN_SWEEP_HF_DONE
