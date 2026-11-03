#!/bin/bash
# E6: sampling diagnostics with the final configuration on IULUI SquareTurn around its first anchor-plane crossing (about 47 s):
# per-epoch z-marginal strips (seq_strip) over epochs 4400-5200 and a full dump (samples, weights, landscape) at epoch 4800.
# Waits for the ablation continuation (ABLATION7B_DONE). Output result/runlogs/dump_final_st.txt, run label dg_SquareTurn.
cd "$(dirname "$0")/../.."; L=result/runlogs
until grep -q "ABLATION7B_DONE" $L/ablation7b.log 2>/dev/null; do sleep 30; done
export KIMS_OMP_THREADS=8
scripts/revision/iu_run.sh dg_SquareTurn SquareTurn T:=200 head_filt:=1 still_bias:=1 rollout_fp32:=1 group_same_ts:=1 seq_strip_from:=4400 seq_strip_to:=5200 seq_strip_step:=1 seq_dump:=4800 seq_dump_file:=$L/dump_final_st.txt > /dev/null 2>&1
echo "$(date +%T) DIAG_DONE" >> $L/diag.log
