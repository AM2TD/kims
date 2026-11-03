#!/bin/bash
# One NTU VIRAL run of the pure-sampling + filtered-head configuration. usage: ntu_run.sh <label> <sequence> <extra args...>
# uwb_std:=-1 keeps the per-range std column (0.055; the published preset overrides it with 0.127), uwb_delta:=3 restores the
# standard Tukey cut (published 27.6), IMU densities from config/imu_noise_ntu.yaml (datasheet).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${KIMS_OMP_THREADS:-8}   # host threads of the final configuration (KIMS_OMP_THREADS); lower it when many runs share the CPU
L=result/runlogs; mkdir -p $L result/ntu/diag
label=$1; seq=$2; shift 2
V="gamma:=1 w_uwb:=1 ctrl_mode:=0 floor_penalty:=0 K:=4 carry_bias:=0 align_still:=3 fresh_only:=1 dc_acc:=0.1 dc_gyr:=0.01 q_acc:=0.014 q_gyr:=0.00061 uwb_std:=-1 uwb_delta:=3 dt:=0.0169"
M="true_time:=1 head:=2 seq:=1 rb:=0 N:=8192 ess_min:=0.0244 seq_done:=0.05 seq_newmass:=0 head_filt:=1 seed:=1234"
ARGS=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('ntu','kims','$seq','result/ntu/diag/${label}_pose.txt')))")
${KIMS_BUILD:-build}/bin/kims $ARGS $V $M $* > $L/${label}.log 2>&1
echo "${label}_DONE"
