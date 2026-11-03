#!/bin/bash
# Wait for a sequence zip download (curl) to finish, prepare the sequence, run KIMS (final NTU configuration, GT-free
# still-phase init with z = 0.2 m), and score with the 4-DoF alignment. usage: ntu_new_run.sh <seq>
cd "$(dirname "$0")/../.."
seq=$1; zip=${NTU_RAW:-$HOME/ntu_raw}/$seq.zip; L=result/runlogs
while pgrep -f "curl .* -o $seq.zip" > /dev/null; do sleep 60; done
[ -s $zip ] || { echo "$seq: no zip"; exit 1; }
scripts/revision/ntu_prepare.sh $seq $zip > $L/prep_$seq.log 2>&1 || { echo "$seq: prepare failed"; tail -5 $L/prep_$seq.log; exit 1; }
I=$(tail -1 $L/prep_$seq.log | sed 's/#.*//' | sed 's/init_z:=[^ ]*/init_z:=0.2/')
echo "$seq init: $I"
rm -rf $zip ${NTU_RAW:-$HOME/ntu_raw}/$seq
scripts/revision/ntu_run.sh n_$seq $seq T:=118 $I still_bias:=1 head_filt:=0 uwb_std:=0.3 uwb_delta:=5 anchor_file:=$PWD/data/ntu/anchors_$seq.txt > /dev/null 2>&1
python3 scripts/revision/ntu_align_score.py $seq result/ntu/diag/n_${seq}_pose.txt 117
echo "${seq}_NEW_DONE"
