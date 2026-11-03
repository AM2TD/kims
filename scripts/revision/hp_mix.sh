#!/bin/bash
# LOS/NLOS mixture range likelihood, seed 1234, one branch, analytic head.
# usage: hp_mix.sh <label> <sequence> <extra args...>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${KIMS_OMP_THREADS:-8}   # host threads of the final configuration (KIMS_OMP_THREADS); lower it when many runs share the CPU
L=result/runlogs; mkdir -p $L result/miluv/diag
label=$1; seq=$2; shift 2
V="gamma:=1 w_uwb:=1 ctrl_mode:=0 floor_penalty:=0 N:=2048 K:=4 carry_bias:=0 align_still:=3 fresh_only:=1"
Q="q_acc_x:=0.0229 q_acc_y:=0.0229 q_acc_z:=0.0229 q_gyr_x:=0.01135 q_gyr_y:=0.01154 q_gyr_z:=0.01082 dt:=0.0131 T:=229"
M="true_time:=1 head:=2 seq:=1 rb:=1 seq_newmass:=0 ess_min:=0.065 seq_done:=0.1 seed:=1234 uwb_model:=1"
ARGS=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('miluv','kims','$seq','result/miluv/diag/${label}_pose.txt')))")
${KIMS_BUILD:-build}/bin/kims $ARGS $V $Q $M $* > $L/${label}.log 2>&1
echo "${label}_DONE"
