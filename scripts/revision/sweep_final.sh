#!/bin/bash
# Final-configuration sweeps for the revised paper figures (FP32 rollout + 8 host threads).
#  T: per-epoch solve time vs N, one run at a time with the GPU to itself (MILUV random3_0, NTU eee_01, UTIL
#     const1-trial1-tdoa2, IULUI SquareTurn), N 16384 .. 512; these runs are also the accuracy points of those 4 sequences.
#  N: accuracy vs N on the 33 sweep sequences (MILUV 13, NTU 9, UTIL trial-1 8, IULUI 3), three runs at a time, levels
#     16384, 4096, 2048, 1024, 512 (8192 = the table runs f32a_/uf1_/f32g_).
#  S: seeds 1..10 on the same 33 sequences (seed 1234 = the table runs), three at a time, seed-major order.
# Divergence (any pose of a running sweep leaves the 60 m box, feedback-stop-on-divergence): T skips the smaller N of that
# sequence, N stops that level and every smaller one, S stops the seed sweep. Outputs result/<ds>/diag/sweep/, logs
# result/runlogs/sweep/. usage: sweep_final.sh [T] [N] [S]   (default: all three, in this order)
cd "$(dirname "$0")/../.."; R=scripts/revision; L=result/runlogs/sweep; export KIMS_OMP_THREADS=8
mkdir -p $L result/miluv/diag/sweep result/ntu/diag/sweep result/util/diag/sweep result/iului/diag/sweep
LOG=$L/sweep.log; PH="${*:-T N S}"
MIL="default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
NTU="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
UT="const1-trial1-tdoa2 const1-trial1-tdoa3 const2-trial1-tdoa2 const2-trial1-tdoa3 const3-trial1-tdoa2 const3-trial1-tdoa3 const4-trial1-tdoa2-traj1 const4-trial1-tdoa3-traj1"
IU="SquareTurn CircleSpin AC3_125000"
TSEQ="default_1_random3_0 eee_01 const1-trial1-tdoa2 SquareTurn"
ALL=$(for s in $MIL; do echo "miluv $s"; done; for s in $NTU; do echo "ntu $s"; done; for s in $UT; do echo "util $s"; done; for s in $IU; do echo "iului $s"; done)
cmd() {   # cmd <dataset> <seq> <label> <extra args>: the table command of that sequence with a new label and extra args
  local ds=$1 s=$2 lab=$3 ex=$4
  case $ds in
    miluv) echo "$R/hp_mix.sh $lab $s dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1 T:=229 rollout_fp32:=1 $ex";;
    ntu)   echo "$(grep " f32a_$s $s " result/runlogs/f32a_jobs.txt | sed "s# > /dev/null 2>&1##; s#f32a_$s #$lab #") $ex";;
    util)  echo "$R/util_run.sh $lab $s rollout_fp32:=1 $ex";;
    iului) echo "$R/iu_run.sh $lab $s T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1 $ex";;
  esac
}
diverged() { for f in "$@"; do [ -f "$f" ] && tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 3600) bad=1 } END { exit !bad }' && { echo "$f"; return 0; }; done; return 1; }
watch() {   # watch <pattern of pose files> <stop file>: kill the sweep's kims runs and xargs on divergence
  while sleep 15; do
    [ -f "$2" ] && return 0
    f=$(diverged $1) && { echo "$(date +%T) DIVERGED $f" >> $LOG; touch $L/diverged_flag; pkill -f "xargs -P 3 -I"; pkill -f "diag/sweep/$3"; return 0; }
  done
}
echo "$(date +%T) START phases: $PH" >> $LOG
for ph in $PH; do case $ph in
T)  for ds_s in "miluv default_1_random3_0" "ntu eee_01" "util const1-trial1-tdoa2" "iului SquareTurn"; do
      set -- $ds_s; ds=$1; s=$2
      for n in 16384 8192 4096 2048 1024 512; do
        lab=sweep/tn${n}_$s; f=result/$ds/diag/${lab}_pose.txt; rm -f $L/run_done
        ( watch "$f" $L/run_done "tn${n}_" ) & W=$!
        eval "$(cmd $ds $s $lab "N:=$n")" > /dev/null 2>&1; touch $L/run_done; wait $W
        if [ -f $L/diverged_flag ]; then rm -f $L/diverged_flag; echo "$(date +%T) T: $s diverged at N $n -> smaller N skipped" >> $LOG; break; fi
        echo "$(date +%T) T done $lab $(grep -h 'solves in' result/runlogs/${lab}.log | cut -c8-90)" >> $LOG
      done
    done;;
N)  for n in 16384 4096 2048 1024 512; do
      J=$L/jobs_n$n.txt; : > $J
      echo "$ALL" | while read ds s; do case " $TSEQ " in *" $s "*) continue;; esac; echo "$(cmd $ds $s sweep/n${n}_$s "N:=$n") > /dev/null 2>&1" >> $J; done
      rm -f $L/level_done $L/diverged_flag
      ( watch "result/*/diag/sweep/n${n}_*_pose.txt" $L/level_done "n${n}_" ) & W=$!
      xargs -P 3 -I{} bash -c '{}' < $J; touch $L/level_done; wait $W
      if [ -f $L/diverged_flag ]; then echo "$(date +%T) N: level $n diverged -> levels below $n skipped" >> $LOG; rm -f $L/diverged_flag; break; fi
      echo "$(date +%T) N level $n done ($(wc -l < $J) runs)" >> $LOG
    done;;
S)  J=$L/jobs_seeds.txt; : > $J
    for sd in 1 2 3 4 5 6 7 8 9 10; do echo "$ALL" | while read ds s; do echo "$(cmd $ds $s sweep/sd${sd}_$s "seed:=$sd") > /dev/null 2>&1" >> $J; done; done
    rm -f $L/level_done $L/diverged_flag
    ( watch "result/*/diag/sweep/sd*_pose.txt" $L/level_done "sd" ) & W=$!
    xargs -P 3 -I{} bash -c '{}' < $J; touch $L/level_done; wait $W
    if [ -f $L/diverged_flag ]; then echo "$(date +%T) S: seed sweep stopped by a divergence" >> $LOG; rm -f $L/diverged_flag; else echo "$(date +%T) S done ($(wc -l < $J) runs)" >> $LOG; fi;;
esac; done
echo "$(date +%T) SWEEP_DONE" >> $LOG
