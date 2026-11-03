#!/bin/bash
cd "$(dirname "$0")/../.."; export KIMS_BIN_DIR=$PWD/${KIMS_BUILD:-build}/bin
NT="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
init() { case $1 in eee_*) python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])";; nya_01) echo "init_x:=6.468 init_y:=-2.934 init_z:=0.2";; nya_02) echo "init_x:=7.0 init_y:=-2.5 init_z:=0.2";; nya_03) echo "init_x:=6.805 init_y:=-3.815 init_z:=0.2";; sbs_01) echo "init_x:=19.312 init_y:=-9.862 init_z:=0.2";; sbs_02) echo "init_x:=16.039 init_y:=-12.987 init_z:=0.2";; sbs_03) echo "init_x:=13.874 init_y:=-8.847 init_z:=0.2";; esac; }
anc() { case $1 in eee_*) echo "anchor_file:=$PWD/data/ntu/anchors_eee.txt";; *) echo "anchor_file:=$PWD/data/ntu/anchors_$1.txt";; esac; }
run() { A=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('ntu','$1','$2','result/baselines/$3_$2_pose.txt')))"); ${KIMS_BUILD:-build}/bin/$1 $A $(init $2) $(anc $2) $4 > result/runlogs/$3_$2.log 2>&1; }
n=0
for s in $NT; do run ekf_rts $s ekfrts "" & run ekf_rts $s ekfrtsfl "floor_penalty:=50" & run sw_pgo $s swpgofl "window_size:=15 floor_penalty:=50" & n=$((n+3)); [ $((n%6)) = 0 ] && wait; done
wait; echo FLOOR_BASE_DONE
