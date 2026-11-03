#!/bin/bash
# E7 addition: the Gaussian likelihood with the LOS width (ga_, std 0.081 m, no cut) diverges on every MILUV flight within ~30 s;
# a fairer Gaussian uses an inflated std. ga3_ = Gaussian std 0.3 m (the NTU value), no cut, MILUV 13 flights. Runs after the
# ablation continuation; the watcher kills only a diverging run. Log result/runlogs/ablation7c.log.
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/ablation7c.log; export KIMS_OMP_THREADS=8
until grep -q "ABLATION7B_DONE" $L/ablation7b.log 2>/dev/null; do sleep 30; done
J=$L/jobs_ab7c.txt; : > $J
grep "hp_mix.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# ga3_#; s# > /dev/null 2>&1##' | sed 's#$# uwb_model:=0 uwb_std:=0.3 uwb_delta:=1e9 > /dev/null 2>\&1#' >> $J
echo "$(date +%T) ABLATION7C START $(wc -l < $J) jobs" > $LOG; rm -f $L/ab7c_done; : > $L/ab7c_killed.txt
( while sleep 20; do
    for f in result/miluv/diag/ga3_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue; grep -qx "$f" $L/ab7c_killed.txt && continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$f" >> $L/ab7c_killed.txt; echo "$(date +%T) DIVERGED $f (run killed, batch continues)" >> $LOG
        lab=$(basename "$f" _pose.txt); pkill -f "bin/kims .*diag/${lab}_"; fi; done
    [ -f $L/ab7c_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/ab7c_done; wait $W
echo "$(date +%T) ABLATION7C_DONE" >> $LOG
