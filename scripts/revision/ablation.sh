#!/bin/bash
# Ablation of the final configuration. Two waves (<= 15 concurrent).
cd "$(dirname "$0")/../.."; R=scripts/revision
A="dc_acc:=0.1 dc_gyr:=0.01 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 head_filt:=1 still_bias:=1"
NT="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
I1=$(python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['eee_01'])")
IS="init_x:=19.312 init_y:=-9.862 init_z:=0.2 init_yaw:=90.0 anchor_file:=$PWD/data/ntu/anchors_sbs_01.txt"
run_m() { $R/hp_mix.sh $1 $2 $A T:=229 $3 > /dev/null 2>&1; }
run_i() { $R/iu_run.sh $1 SquareTurn T:=200 head_filt:=1 still_bias:=1 $2 > /dev/null 2>&1; }
run_n() { $R/ntu_run.sh $1 $2 T:=118 $3 $NT $4 > /dev/null 2>&1; }
# wave 1: MILUV (slow) + IULUI
for v in "notemp ess_min:=0" "noprop seq_alpha:=0" "noarr arr_fixed:=1" "tukey uwb_model:=0" "nostill still_bias:=0"; do set -- $v; lab=$1; shift
  run_m ab_${lab}_hp1 default_1_remoteControlHighPace_1 "$*" &
  run_m ab_${lab}_hp0 default_1_remoteControlHighPace_0 "$*" &
  [ $lab != tukey ] && run_i ab_${lab}_st "$*" &
done
wait
# wave 2: NTU
for v in "notemp ess_min:=0" "noprop seq_alpha:=0" "noarr arr_fixed:=1" "nostill still_bias:=0" "hf head_filt:=1"; do set -- $v; lab=$1; shift
  run_n ab_${lab}_eee01 eee_01 "$I1" "$*" &
  run_n ab_${lab}_sbs01 sbs_01 "$IS" "$*" &
done
wait
echo ABLATION_DONE
