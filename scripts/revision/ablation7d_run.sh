#!/bin/bash
# E7 addition for MILUV: the calibrated LOS/NLOS mixture vs the plain robust loss the other datasets use (Tukey, std 0.081 m = the
# mixture's effective std, cut 5 sigma): ga5_. Isolates what the mixture adds, since removing the cut altogether diverges everywhere.
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/ablation7d.log; export KIMS_OMP_THREADS=8
J=$L/jobs_ab7d.txt; : > $J
grep "hp_mix.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# ga5_#; s# > /dev/null 2>&1##' | sed 's#$# uwb_model:=0 uwb_std:=0.081 uwb_delta:=5 > /dev/null 2>\&1#' >> $J
echo "$(date +%T) ABLATION7D START $(wc -l < $J) jobs" > $LOG; rm -f $L/ab7d_done; : > $L/ab7d_killed.txt
( while sleep 20; do
    for f in result/miluv/diag/ga5_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue; grep -qx "$f" $L/ab7d_killed.txt && continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$f" >> $L/ab7d_killed.txt; echo "$(date +%T) DIVERGED $f (run killed, batch continues)" >> $LOG
        lab=$(basename "$f" _pose.txt); pkill -f "bin/kims .*diag/${lab}_"; fi; done
    [ -f $L/ab7d_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/ab7d_done; wait $W
echo "$(date +%T) ABLATION7D_DONE" >> $LOG
