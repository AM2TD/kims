#!/bin/bash
# NTU nya/sbs: matched LM, published-configuration KIMS (preset path), and the floor prior (z >= 0) ablation for KIMS
# and the published configuration on all 9 sequences. Two GPU waves.
cd "$(dirname "$0")/../.."; export OMP_NUM_THREADS=1; R=scripts/revision; L=result/runlogs; mkdir -p result/ntu/diag
NT="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
init() { case $1 in eee_*) python3 -c "import json; print(json.load(open('result/runlogs/ntu_init_args.json'))['$1'])";; nya_01) echo "init_x:=6.468 init_y:=-2.934 init_z:=0.2";; nya_02) echo "init_x:=7.0 init_y:=-2.5 init_z:=0.2";; nya_03) echo "init_x:=6.805 init_y:=-3.815 init_z:=0.2";; sbs_01) echo "init_x:=19.312 init_y:=-9.862 init_z:=0.2";; sbs_02) echo "init_x:=16.039 init_y:=-12.987 init_z:=0.2";; sbs_03) echo "init_x:=13.874 init_y:=-8.847 init_z:=0.2";; esac; }
anc() { case $1 in eee_*) echo "";; *) echo "anchor_file:=$PWD/data/ntu/anchors_$1.txt";; esac; }
FIN="still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5"
pub() { A=$(python3 -c "import sys; sys.path.insert(0,'scripts'); from run import compose; print(' '.join(compose('ntu','kims','$2','result/ntu/diag/$1_$2_pose.txt')))"); ${KIMS_BUILD:-build}/bin/kims $A $(init $2) $(anc $2) $3 > $L/$1_$2.log 2>&1; }
# wave 1: published configuration (fast) with / without floor on all 9, LM on nya/sbs (slow)
for s in $NT; do pub pub $s "" & pub pubfl $s "floor_penalty:=50" & done
for s in nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do $R/ntu_run.sh lm_$s $s T:=118 $(init $s) $(anc $s) $FIN lm:=1 > /dev/null 2>&1 & done
wait
# wave 2: KIMS + floor on nya/sbs (eee exist as fl_*)
for s in nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03; do $R/ntu_run.sh fl_$s $s T:=118 $(init $s) $(anc $s) $FIN floor_penalty:=50 > /dev/null 2>&1 & done
wait; echo NTU_MORE_DONE
