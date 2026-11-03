#!/bin/bash
# Plain SW-PGO (published settings, no Cauchy) on MILUV + NTU, and EKF-RTS on the new NTU sequences.
cd "$(dirname "$0")/../.."; export KIMS_BIN_DIR=$PWD/${KIMS_BUILD:-build}/bin
MIL="default_1_remoteControlHighPace_1 default_1_random3_1 default_3_random3_1 default_3_zigzag_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
NT="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
ntu_init() { case $1 in eee_*) python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])";; nya_01) echo "init_x:=6.468 init_y:=-2.934 init_z:=0.2";; nya_02) echo "init_x:=7.0 init_y:=-2.5 init_z:=0.2";; nya_03) echo "init_x:=6.805 init_y:=-3.815 init_z:=0.2";; sbs_01) echo "init_x:=19.312 init_y:=-9.862 init_z:=0.2";; sbs_02) echo "init_x:=16.039 init_y:=-12.987 init_z:=0.2";; sbs_03) echo "init_x:=13.874 init_y:=-8.847 init_z:=0.2";; esac; }
ntu_anchor() { case $1 in eee_*) echo data/ntu/anchors_eee.txt;; *) echo data/ntu/anchors_$1.txt;; esac; }
run() { A=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('$1','$2','$3','result/baselines/$4_$3_pose.txt')))"); ${KIMS_BUILD:-build}/bin/$2 $A $5 > result/runlogs/$4_$3.log 2>&1; }
n=0
for s in $MIL; do run miluv sw_pgo $s swpgo "window_size:=15" & n=$((n+1)); [ $((n%6)) = 0 ] && wait; done
for s in $NT; do run ntu sw_pgo $s swpgo "window_size:=15 $(ntu_init $s) anchor_file:=$PWD/$(ntu_anchor $s)" & n=$((n+1)); [ $((n%6)) = 0 ] && wait; done
for s in nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do run ntu ekf_rts $s ekfrts "$(ntu_init $s) anchor_file:=$PWD/$(ntu_anchor $s)" & n=$((n+1)); [ $((n%6)) = 0 ] && wait; done
wait; echo FIX_DONE
