#!/bin/bash
# IULUI window sweep with the filtered head. 12 runs, 6 at a time.
cd "$(dirname "$0")/../.."
R=scripts/revision
for T in 100 150 200 300; do
  for s in SquareBob SquareTurn CircleSpin; do $R/iu_run.sh hfw${T}_$s $s T:=$T head_filt:=1 > /dev/null 2>&1 & done
  [ $T = 150 ] && wait
done
wait
echo IU_SWEEP_HF_DONE
