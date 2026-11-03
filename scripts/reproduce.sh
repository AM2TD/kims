#!/bin/bash
# Re-run KIMS with the paper's final configuration on every evaluated sequence and rebuild the RMSE tables.
#   scripts/reproduce.sh [miluv] [ntu] [util] [iului] [tables]      (default: all five steps)
# Environment: KIMS_LABEL (run label prefix, default repro), KIMS_PARALLEL (runs at a time, default 3; each run uses
# about 0.8 GB of GPU memory and KIMS_OMP_THREADS host threads, default 8).
# Data: scripts/prepare_data.sh (MILUV, NTU VIRAL, UTIL); IULUI ships in data/iului/.
# Outputs: result/<dataset>/diag/<label>_<seq>_*.txt, logs result/runlogs/<label>_<seq>.log, tables
# result/runlogs/<label>_table_v2.txt (MILUV, NTU per sequence) and result/runlogs/<label>_rmse.txt (all four datasets).
# The SFUISE and devkit columns are read from their own runs if present (baselines/README.md), otherwise shown as '-'.
set -e
cd "$(dirname "$0")/.."; R=scripts/revision; KL=${KIMS_LABEL:-repro}; P=${KIMS_PARALLEL:-3}
export KIMS_OMP_THREADS=${KIMS_OMP_THREADS:-8}
STEPS="${*:-miluv ntu util iului tables}"
MILUV="default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
NTU="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
IULUI="SquareTurn CircleSpin AC3_125000"
F="rollout_fp32:=1"                                        # final configuration: FP32 rollout (+ KIMS_OMP_THREADS=8)
run() { echo "$(date +%T) $1: $(wc -l < $2) runs, $P at a time"; xargs -P $P -I{} bash -c '{}' < $2; echo "$(date +%T) $1 done"; }
J=$(mktemp); trap 'rm -f $J' EXIT
for st in $STEPS; do : > $J; case $st in
  miluv) for s in $MILUV; do [ -d data/miluv/$s ] && echo "$R/hp_mix.sh ${KL}_$s $s dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=2048 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1 T:=229 $F > /dev/null 2>&1" >> $J; done; run MILUV $J;;
  ntu)   for s in $NTU; do [ -d data/ntu/$s ] || continue
           I=$(python3 -c "import json; p = json.load(open('data/ntu/init_poses.json'))['$s']; print(' '.join(f'{k}:={p[k]}' for k in ('init_x', 'init_y', 'init_z', 'init_yaw')) + ' anchor_file:=$PWD/' + p['anchor_file'])")
           echo "$R/ntu_run.sh ${KL}_$s $s T:=118 $I N:=2048 still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5 $F > /dev/null 2>&1" >> $J; done; run NTU $J;;
  util)  for d in data/util/*/; do s=$(basename $d); echo "$R/util_run.sh ${KL}_$s $s N:=2048 $F > /dev/null 2>&1" >> $J; done; run UTIL $J;;
  iului) for s in $IULUI; do echo "$R/iu_run.sh ${KL}_$s $s T:=200 N:=2048 head_filt:=1 still_bias:=1 group_same_ts:=1 $F > /dev/null 2>&1" >> $J; done; run IULUI $J;;
  tables) python3 $R/final_table_v2.py $KL > result/runlogs/${KL}_table_v2.txt
          TABLE_V2=result/runlogs/${KL}_table_v2.txt UTIL_LABEL=${KL}_ IULUI_LABEL=${KL}_ OUT=result/runlogs/${KL}_rmse.txt python3 $R/final_rmse_final.py;;
  *) echo "unknown step $st"; exit 1;;
esac; done
