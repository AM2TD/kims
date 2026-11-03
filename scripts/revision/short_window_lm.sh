#!/bin/bash
# (1) Matched LM re-runs with the final configuration: MILUV constellation 2 (corrected anchor order) and IULUI (final f32g
#     settings), labels lmf_*; three at a time; the batch stops if a committed pose leaves a 30 m box (stop-on-divergence).
# (2) SFUISE with a window matched to KIMS (published configuration, only window_size changed: 30 knots = 3 s on MILUV,
#     20 knots = 2 s elsewhere), one run at a time with nothing else running; outputs sfuise_w30_* / sfuise_w20_* (UTIL:
#     sfuisebody_w20_* after util_sfuise_body.py). SFUISE divergence is a result here and is recorded, not a stop condition.
# Log: result/runlogs/short_lm.log.   usage: scripts/revision/short_window_lm.sh
cd "$(dirname "$0")/../.."; R=scripts/revision; L=result/runlogs; LOG=$L/short_lm.log; B=${SFUISE_BAGS:-$HOME/util_raw/kims_bags}
export KIMS_OMP_THREADS=8; J=$L/jobs_lmf.txt; : > $J
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1 T:=229 rollout_fp32:=1 lm:=1"
for s in default_1_random3_2 default_1_remoteControlHighPace_2; do echo "$R/hp_mix.sh lmf_$s $s $A > /dev/null 2>&1" >> $J; done
for s in SquareTurn CircleSpin AC3_125000; do echo "$R/iu_run.sh lmf_$s $s T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1 lm:=1 > /dev/null 2>&1" >> $J; done
echo "$(date +%T) LMF START" > $LOG; rm -f $L/lmf_done
( while sleep 15; do
    for f in result/miluv/diag/lmf_*_pose.txt result/iului/diag/lmf_*_pose.txt; do
      case "$f" in *_now_pose.txt|*_flush_pose.txt) continue;; esac; [ -f "$f" ] || continue
      if tail -n 50 "$f" | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 900) bad=1 } END { exit !bad }'; then
        echo "$(date +%T) DIVERGED $f" >> $LOG; pkill -f "xargs -P 3 -I"; pkill -f "bin/kims .*diag/lmf_"; exit 0; fi; done
    [ -f $L/lmf_done ] && exit 0
  done ) & W=$!
xargs -P 3 -I{} bash -c '{}' < $J; touch $L/lmf_done; wait $W
grep -q DIVERGED $LOG && { echo "$(date +%T) STOPPED (LM divergence)" >> $LOG; exit 1; }
echo "$(date +%T) LMF_DONE" >> $LOG
# ---- (2) SFUISE, matched window, sequential
MIL="default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
JM=$L/jobs_sfw_miluv.txt; : > $JM; for s in $MIL; do echo "miluv $s" >> $JM; done
JN=$L/jobs_sfw_ntu.txt; : > $JN; for s in eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do echo "ntu $s" >> $JN; done; for s in SquareTurn CircleSpin AC3_125000; do echo "iului $s" >> $JN; done
echo "$(date +%T) SFUISE MILUV w30 START" >> $LOG
SFUISE_PORT=13700 VARIANT=_w30 $R/sfuise_toa_batch.sh $JM >> $LOG 2>&1
echo "$(date +%T) SFUISE NTU+IULUI w20 START" >> $LOG
SFUISE_PORT=13760 VARIANT=_w20 $R/sfuise_toa_batch.sh $JN >> $LOG 2>&1
echo "$(date +%T) SFUISE UTIL w20 START" >> $LOG
port=13840
for s in const1-trial1-tdoa2 const1-trial1-tdoa3 const2-trial1-tdoa2 const2-trial1-tdoa3 const3-trial1-tdoa2 const3-trial1-tdoa3 const4-trial1-tdoa2-traj1 const4-trial1-tdoa3-traj1; do
  for att in 1 2 3; do
    port=$((port + 1)); out=result/util/diag/sfuise_w20_${s}_pose.txt; rm -f $out
    SFUISE_CFG=$B/util_w20.yaml SFUISE_PREFIX=sfuise_w20_ $R/util_sfuise.sh $s $port < /dev/null > /dev/null 2>&1
    n=$(grep -vc '^#' $out 2>/dev/null); n=${n:-0}; echo "util $s attempt $att: $n poses" >> $LOG; [ "$n" -gt 100 ] && break
  done
done
python3 $R/util_sfuise_body.py >> $LOG 2>&1
echo "$(date +%T) SHORT_LM_DONE" >> $LOG
