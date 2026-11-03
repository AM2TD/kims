#!/bin/bash
# One KIMS run on a UTIL flight (dataset util, UWB TDOA, one tag), prepared by util_prepare.py into data/util/<seq>/.
# usage: util_run.sh <label> <sequence> <extra args...>   -> result/util/diag/<label>_pose.txt, result/runlogs/<label>.log
# Settings: the pure-sampling configuration of the other datasets (N 8192, K 4, gamma 1, sequential proposal, still_bias)
# with the dataset-provided UTIL values: TDOA std 0.224 m and a 5-sigma cut (devkit ESKF: std sqrt(0.05), Mahalanobis
# gate 5), IMU white-noise densities from the devkit ESKF process noise (2.0 m/s^2 and 0.1 rad/s per 8 ms step ->
# 0.179 m/s^2/sqrt(Hz), 0.0089 rad/s/sqrt(Hz)), devkit tag lever arm. Epochs: the TDOA stream binned at 50 Hz (20 ms),
# window T 100 (2 s). Start pose: the devkit ESKF one for every flight, (1.25, 0, 0.07) and identity attitude (the
# rest-phase TDOA fit in meta.json is off by up to 1.9 m on const4, NLOS at the pad). Gravity alignment over the first 1.0 s (lift-off is 1.6-6 s in). IMU cost weights 0 (the IMU prior
# enters once, through the proposal prior; on MILUV/NTU the preset weights were negligible).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${KIMS_OMP_THREADS:-8}   # host threads of the final configuration (KIMS_OMP_THREADS); lower it when many runs share the CPU
L=result/runlogs; mkdir -p $L result/util/diag
label=$1; seq=$2; shift 2
D=data/util/$seq
I="init_x:=1.25 init_y:=0.0 init_z:=0.07 init_yaw:=0 num_anchors:=$(python3 -c "import json; print(json.load(open('$D/meta.json'))['num_slots'])")"
B="dataset:=util sequence:=$seq pkg_dir:=$PWD data_dir:=$PWD/$D anchor_file:=$PWD/$D/anchors_A.txt anchor_b_file:=$PWD/$D/anchors_B.txt
   tag0_id:=0 tag0_offset_x:=-0.01245 tag0_offset_y:=0.00127 tag0_offset_z:=0.0908 group_same_ts:=1 output_file:=result/util/diag/${label}_pose.txt $I"
V="gamma:=1 w_uwb:=1 w_acc:=0 w_gyr:=0 ctrl_mode:=0 floor_penalty:=0 K:=4 carry_bias:=0 align_still:=1.0 still_bias:=1 fresh_only:=1 dc_acc:=0.1 dc_gyr:=0.01 q_acc:=0.179 q_gyr:=0.0089 uwb_std:=0.224 uwb_delta:=5 dt:=0.02 T:=100"
M="true_time:=1 head:=2 seq:=1 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 seq_newmass:=0 head_filt:=1 seed:=1234"
${KIMS_BUILD:-build}/bin/kims $B $V $M $* > $L/${label}.log 2>&1
echo "${label}_DONE"
