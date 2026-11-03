#!/bin/bash
# E6 re-run: the dump at seq_dump truncates the shared file, erasing strips recorded before it. So: strips 4400-5200 with the dump
# at the strip start (dg2_, dg2lm_), and a separate dump at the crossing (epoch 4720 ~ 47.2 s; dg3_, dg3lm_) without strips.
# Waits for the Gaussian std 0.3 batch and the first LM diagnostic run. Markers in result/runlogs/diag.log.
cd "$(dirname "$0")/../.."; L=result/runlogs; export KIMS_OMP_THREADS=8
until grep -q "ABLATION7C_DONE" $L/ablation7c.log 2>/dev/null && grep -q "DIAG_LM_DONE" $L/diag.log 2>/dev/null; do sleep 30; done
B="SquareTurn T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1"
scripts/revision/iu_run.sh dg2_SquareTurn $B seq_strip_from:=4400 seq_strip_to:=5200 seq_strip_step:=1 seq_dump:=4400 seq_dump_file:=$L/dump2_final_st.txt > /dev/null 2>&1
scripts/revision/iu_run.sh dg2lm_SquareTurn $B lm:=1 seq_strip_from:=4400 seq_strip_to:=5200 seq_strip_step:=1 seq_dump:=4400 seq_dump_file:=$L/dump2_final_st_lm.txt > /dev/null 2>&1
scripts/revision/iu_run.sh dg3_SquareTurn $B seq_dump:=4720 seq_dump_file:=$L/dump3_final_st.txt > /dev/null 2>&1
scripts/revision/iu_run.sh dg3lm_SquareTurn $B lm:=1 seq_dump:=4720 seq_dump_file:=$L/dump3_final_st_lm.txt > /dev/null 2>&1
echo "$(date +%T) DIAG2_DONE" >> $L/diag.log
