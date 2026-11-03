#!/bin/bash
# E3 parameter transfer: every dataset with ONE common setting instead of its tuned values. Common = the MILUV/IULUI values:
# q_acc 0.040, q_gyr 0.0013 (basis prior densities), head_filt 1, window 2 s. Changed per dataset: MILUV window 3 s -> 2 s
# (T 153); NTU q_acc 0.014 -> 0.040, q_gyr 0.00061 -> 0.0013, head_filt 0 -> 1; UTIL q_acc 0.179 -> 0.040, q_gyr 0.0089 ->
# 0.0013. IULUI already uses the common setting (f32g_). Range-noise models stay per dataset (sensor calibration).
# Labels tr_*; three at a time; stops the batch if a committed pose leaves a 100 m box. Log result/runlogs/transfer.log.
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/transfer.log; export KIMS_OMP_THREADS=8; J=$L/jobs_tr.txt; : > $J
C="q_acc:=0.040 q_gyr:=0.0013 head_filt:=1"
grep "hp_mix.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# tr_#; s# > /dev/null 2>&1##' | sed 's#$# T:=153 > /dev/null 2>\&1#' >> $J
grep "ntu_run.sh f32a_" $L/f32a_jobs.txt | sed 's# f32a_# tr_#; s# > /dev/null 2>&1##' | sed "s#\$# $C > /dev/null 2>\&1#" >> $J
for s in $(ls data/util); do echo "scripts/revision/util_run.sh tr_$s $s rollout_fp32:=1 $C > /dev/null 2>&1" >> $J; done
echo "$(date +%T) TRANSFER START $(wc -l < $J) jobs" > $LOG; rm -f $L/tr_done
( while sleep 20; do
    for f in result/miluv/diag/tr_*_pose.txt result/ntu/diag/tr_*_pose.txt result/util/diag/tr_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$(date +%T) DIVERGED $f" >> $LOG; pkill -f "xargs -P 3 -I"; pkill -f "bin/kims .*diag/tr_"; exit 0; fi; done
    [ -f $L/tr_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/tr_done; wait $W
grep -q DIVERGED $LOG && { echo "$(date +%T) STOPPED" >> $LOG; exit 1; }
echo "$(date +%T) TRANSFER_DONE" >> $LOG
