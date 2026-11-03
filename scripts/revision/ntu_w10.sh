#!/bin/bash
# KIMS with a 10 s window (T 590 at dt 0.0169) on the nine NTU VIRAL sequences, to compare with SFUISE's 10 s window.
# Final configuration otherwise (f32a NTU command lines); three runs at a time; stops if a pose leaves the 60 m box.
cd "$(dirname "$0")/../.."; L=result/runlogs/sweep; export KIMS_OMP_THREADS=8; J=$L/jobs_w10.txt; : > $J
for s in eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do
  echo "$(grep " f32a_$s $s " result/runlogs/f32a_jobs.txt | sed "s# > /dev/null 2>&1##; s#f32a_$s #sweep/w10_$s #") T:=590 > /dev/null 2>&1" >> $J; done
rm -f $L/w10_done; echo "$(date +%T) W10 START" >> $L/w10.log
( while sleep 15; do
    for f in result/ntu/diag/sweep/w10_*_pose.txt; do [ -f "$f" ] || continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 3600) bad=1 } END { exit !bad }'; then
        echo "$(date +%T) DIVERGED $f" >> $L/w10.log; pkill -f "xargs -P 3 -I"; pkill -f "diag/sweep/w10_"; exit 0; fi; done
    [ -f $L/w10_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/w10_done; wait $W
echo "$(date +%T) W10_DONE" >> $L/w10.log
