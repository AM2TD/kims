#!/bin/bash
# E7 ablations with the final configuration on the 33-sequence sweep set (MILUV 13, NTU 9, UTIL 8, IULUI 3):
#   nt_  no tempering (ess_min 0, seq_done 0: one importance-sampling pass per epoch)
#   ga_  Gaussian range likelihood instead of the robust one (MILUV: uwb_model 0 with the effective std 0.081 m instead of
#        the LOS/NLOS mixture; all: Tukey cut uwb_delta 1e9 = no cut)
# Waits for the transfer batch (TRANSFER_DONE); three at a time; stops if a committed pose leaves a 100 m box.
# Log result/runlogs/ablation7.log.
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/ablation7.log; export KIMS_OMP_THREADS=8; R=scripts/revision
until grep -q "TRANSFER_DONE\|STOPPED" $L/transfer.log 2>/dev/null; do sleep 30; done
J=$L/jobs_ab7.txt; : > $J
UT="const1-trial1-tdoa2 const1-trial1-tdoa3 const2-trial1-tdoa2 const2-trial1-tdoa3 const3-trial1-tdoa2 const3-trial1-tdoa3 const4-trial1-tdoa2-traj1 const4-trial1-tdoa3-traj1"
for v in nt ga; do
  if [ $v = nt ]; then X="ess_min:=0 seq_done:=0"; XM="$X"; else X="uwb_delta:=1e9"; XM="uwb_model:=0 uwb_std:=0.081 uwb_delta:=1e9"; fi
  grep "hp_mix.sh f32a_" $L/f32a_jobs.txt | sed "s# f32a_# ${v}_#; s# > /dev/null 2>&1##" | sed "s#\$# $XM > /dev/null 2>\&1#" >> $J
  grep "ntu_run.sh f32a_" $L/f32a_jobs.txt | sed "s# f32a_# ${v}_#; s# > /dev/null 2>&1##" | sed "s#\$# $X > /dev/null 2>\&1#" >> $J
  for s in $UT; do echo "$R/util_run.sh ${v}_$s $s rollout_fp32:=1 $X > /dev/null 2>&1" >> $J; done
  for s in SquareTurn CircleSpin AC3_125000; do echo "$R/iu_run.sh ${v}_$s $s T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1 $X > /dev/null 2>&1" >> $J; done
done
echo "$(date +%T) ABLATION7 START $(wc -l < $J) jobs" > $LOG; rm -f $L/ab7_done
( while sleep 20; do
    for f in result/*/diag/nt_*_pose.txt result/*/diag/ga_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$(date +%T) DIVERGED $f" >> $LOG; pkill -f "xargs -P 3 -I"; pkill -f "bin/kims .*diag/\(nt\|ga\)_"; exit 0; fi; done
    [ -f $L/ab7_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/ab7_done; wait $W
grep -q DIVERGED $LOG && { echo "$(date +%T) STOPPED" >> $LOG; exit 1; }
echo "$(date +%T) ABLATION7_DONE" >> $LOG
