#!/bin/bash
# E3 follow-up: the common setting broke NTU (eee_01 5.86 m, eee_03 5.69 m) but changed two things at once. Split them:
#   tq_  final NTU setting with only the prior changed (q_acc 0.014 -> 0.040, q_gyr 0.00061 -> 0.0013), head_filt 0 kept
#   th_  final NTU setting with only head_filt 0 -> 1, prior kept
# Waits for the ablation batch (ABLATION7_DONE); three at a time; stops on a 100 m box exit. Log result/runlogs/transfer_ntu.log.
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/transfer_ntu.log; export KIMS_OMP_THREADS=8
until grep -q "ABLATION7_DONE\|STOPPED" $L/ablation7.log 2>/dev/null; do sleep 30; done
J=$L/jobs_trn.txt; : > $J
grep "ntu_run.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# tq_#; s# > /dev/null 2>&1##' | sed 's#$# q_acc:=0.040 q_gyr:=0.0013 > /dev/null 2>\&1#' >> $J
grep "ntu_run.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# th_#; s# > /dev/null 2>&1##' | sed 's#$# head_filt:=1 > /dev/null 2>\&1#' >> $J
echo "$(date +%T) TRN START $(wc -l < $J) jobs" > $LOG; rm -f $L/trn_done
( while sleep 20; do
    for f in result/ntu/diag/tq_*_pose.txt result/ntu/diag/th_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$(date +%T) DIVERGED $f" >> $LOG; pkill -f "xargs -P 3 -I"; pkill -f "bin/kims .*diag/t[qh]_"; exit 0; fi; done
    [ -f $L/trn_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/trn_done; wait $W
grep -q DIVERGED $LOG && { echo "$(date +%T) STOPPED" >> $LOG; exit 1; }
echo "$(date +%T) TRN_DONE" >> $LOG
