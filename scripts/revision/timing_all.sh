#!/bin/bash
# Standalone timing of every method in the final tables (one process at a time, machine otherwise idle), same settings as
# the accuracy runs. Sequences: MILUV HighPace_1 / random3_0 / circular2D_0, NTU eee_01 / sbs_01, UTIL const1-trial1-tdoa2/3
# (SFUISE skips HighPace_1, where it diverges). Order: KIMS, LM (GPU), SFUISE (1x playback), in-house C++ baselines,
# Python devkit filters. Labels tt_*; outputs result/{miluv,ntu,util}/diag/tt_* (KIMS, LM) and result/timing/ (others).
# A KIMS/LM run that leaves a 60 m box stops the campaign (feedback-stop-on-divergence). Summary: timing_table.py.
cd "$(dirname "$0")/../.."; K=$PWD; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; R=scripts/revision; L=result/runlogs; LOG=$L/timing_all.log; O=result/timing
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
NT="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
MIL="default_1_remoteControlHighPace_1 default_1_random3_0 default_1_circular2D_0"; NTU="eee_01 sbs_01"; UT="const1-trial1-tdoa2 const1-trial1-tdoa3"
ntu_init() { case $1 in eee_*) python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])";;
  *) grep " init:" $L/ntu_new_$1.log | head -1 | sed 's/.*init://';; esac; }
ntu_af() { case $1 in eee_*) echo anchor_file:=$PWD/data/ntu/anchors_eee.txt;; *) echo anchor_file:=$PWD/data/ntu/anchors_$1.txt;; esac; }
short() { case $1 in default_1_remoteControlHighPace_1) echo hp1;; default_1_random3_0) echo r30;; default_1_circular2D_0) echo c2d;;
  const1-trial1-tdoa2) echo u2;; const1-trial1-tdoa3) echo u3;; *) echo ${1/_/};; esac; }
log() { echo "$(date +%T) $*" >> $LOG; }
idle() { while pgrep -f '[b]uild_fp32/bin/' > /dev/null; do sleep 10; done; }
diverged() { tail -n 50 "$1" 2>/dev/null | awk '!/^#/ { if ($2*$2+$3*$3+$4*$4 > 3600) bad=1 } END { exit !bad }'; }
compose() { python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('$1','$2','$3','$4')))"; }
echo "$(date +%T) START" > $LOG
for m in k l; do X=""; [ $m = l ] && X="lm:=1"
  for s in $MIL; do idle; lab=tt_${m}_$(short $s); log "$lab"; $R/hp_mix.sh $lab $s $A T:=229 $X > /dev/null 2>&1
    diverged result/miluv/diag/${lab}_pose.txt && { log "DIVERGED $lab -> stop"; exit 1; }; done
  for s in $NTU; do idle; lab=tt_${m}_$(short $s); log "$lab"; $R/ntu_run.sh $lab $s T:=118 $(ntu_init $s) $NT $(ntu_af $s) $X > /dev/null 2>&1
    diverged result/ntu/diag/${lab}_pose.txt && { log "DIVERGED $lab -> stop"; exit 1; }; done
  for s in $UT; do idle; lab=tt_${m}_$(short $s); log "$lab"; $R/util_run.sh $lab $s $X > /dev/null 2>&1
    diverged result/util/diag/${lab}_pose.txt && { log "DIVERGED $lab -> stop"; exit 1; }; done
done
port=13800
for s in default_1_random3_0 default_1_circular2D_0; do port=$((port+1)); log "sfuise $s"; $R/timing_sfuise.sh miluv $s $port >> $LOG 2>&1 < /dev/null; done
for s in $NTU; do port=$((port+1)); log "sfuise $s"; $R/timing_sfuise.sh ntu $s $port >> $LOG 2>&1 < /dev/null; done
for s in $UT; do port=$((port+1)); log "sfuise $s"; $R/timing_sfuise.sh util $s $port >> $LOG 2>&1 < /dev/null; done
for s in $MIL; do
  log "swpgo/ps $s"
  ${KIMS_BUILD:-build}/bin/sw_pgo $(compose miluv sw_pgo $s $O/swpgo_${s}_pose.txt) window_size:=15 align_still:=3 > $O/swpgo_$s.log 2>&1
  ${KIMS_BUILD:-build}/bin/particle_smoothing $(compose miluv particle_smoothing $s $O/ps_${s}_pose.txt) fresh_only:=1 sigma_uwb:=0.3 still_bias:=1 align_still:=3 > $O/ps_$s.log 2>&1
done
for s in $NTU; do
  log "eskf/swpgo/ps $s"; I="$(ntu_init $s) align_still:=3 $(ntu_af $s)"
  ${KIMS_BUILD:-build}/bin/ekf_rts $(compose ntu ekf_rts $s $O/ekfl_${s}_pose.txt) $I fix_q:=1 fresh_only:=1 gate:=5 still_bias:=1 std_uwb:=0.3 lag:=117 > $O/ekfl_$s.log 2>&1
  ${KIMS_BUILD:-build}/bin/sw_pgo $(compose ntu sw_pgo $s $O/swpgo_${s}_pose.txt) window_size:=15 $I > $O/swpgo_$s.log 2>&1
  ${KIMS_BUILD:-build}/bin/particle_smoothing $(compose ntu particle_smoothing $s $O/ps_${s}_pose.txt) $I fresh_only:=1 sigma_uwb:=0.3 still_bias:=1 > $O/ps_$s.log 2>&1
done
for s in $MIL; do log "devkit ekf $s"; (cd ${MILUV_DEVKIT:-$HOME/miluv_devkit} && python3 $K/$R/timing_devkit_miluv.py $s $K/$O/devkit_${s}_ms.txt) >> $LOG 2>&1; done
for s in $UT; do log "util eskf $s"; python3 $R/timing_util_eskf.py $s $O/utileskf_${s}_ms.txt >> $LOG 2>&1; done
log "TIMING_ALL_DONE"
