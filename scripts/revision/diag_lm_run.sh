#!/bin/bash
# E6 companion: the matched LM on the same SquareTurn crossing with the same strips/dump (Laplace cloud per epoch), after diag_run.sh.
cd "$(dirname "$0")/../.."; L=result/runlogs
until grep -q "DIAG_DONE" $L/diag.log 2>/dev/null; do sleep 30; done
export KIMS_OMP_THREADS=8
scripts/revision/iu_run.sh dglm_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1 lm:=1 seq_strip_from:=4400 seq_strip_to:=5200 seq_strip_step:=1 seq_dump:=4800 seq_dump_file:=$L/dump_final_st_lm.txt > /dev/null 2>&1
echo "$(date +%T) DIAG_LM_DONE" >> $L/diag.log
