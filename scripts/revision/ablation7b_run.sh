#!/bin/bash
# Continuation of ablation7_run.sh after its stop-on-divergence (ga_ on MILUV remoteControlHighPace_1 diverged, which is itself the
# ablation's result for that flight): re-runs the two nt_ jobs the stop killed and the remaining ga_ jobs. Divergence here is an
# outcome to record, so the watcher kills ONLY the diverging run and the batch goes on. Waits for the NTU split (TRN_DONE).
cd "$(dirname "$0")/../.."; L=result/runlogs; LOG=$L/ablation7b.log; export KIMS_OMP_THREADS=8
until grep -q "TRN_DONE\|STOPPED" $L/transfer_ntu.log 2>/dev/null; do sleep 30; done
J=$L/jobs_ab7b.txt; : > $J
for s in $(grep -L "solves in" $L/nt_*.log | sed 's#.*/nt_##; s#\.log##'); do grep " nt_$s " $L/jobs_ab7.txt >> $J; done
grep " ga_" $L/jobs_ab7.txt | grep -v "ga_default_1_remoteControlHighPace_1 " >> $J
echo "$(date +%T) ABLATION7B START $(wc -l < $J) jobs" > $LOG; rm -f $L/ab7b_done; : > $L/ab7b_killed.txt
( while sleep 20; do
    for f in result/*/diag/nt_*_pose.txt result/*/diag/ga_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue; grep -qx "$f" $L/ab7b_killed.txt && continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 10000) bad=1 } END { exit !bad }'; then
        echo "$f" >> $L/ab7b_killed.txt; echo "$(date +%T) DIVERGED $f (run killed, batch continues)" >> $LOG
        lab=$(basename "$f" _pose.txt); pkill -f "bin/kims .*diag/${lab}_" ; fi; done
    [ -f $L/ab7b_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/ab7b_done; wait $W
echo "$(date +%T) ABLATION7B_DONE" >> $LOG
