#!/bin/bash
# Follow-up to short_window_lm.sh: SFUISE with the matched window AND the wide 1.5 m gate (w30_g15 on three MILUV flights,
# w20_g15 on three NTU flights), to separate the window effect from the gate interaction (with the 0.3 m gate every range is
# rejected once the short-window estimate drifts past 0.3 m). Waits for SHORT_LM_DONE; one run at a time. Log: short_gate.log.
cd "$(dirname "$0")/../.."; R=scripts/revision; L=result/runlogs; LOG=$L/short_gate.log
until grep -q SHORT_LM_DONE $L/short_lm.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) GATE START" > $LOG
SFUISE_PORT=13900 VARIANT=_w30_g15 $R/sfuise_toa_batch.sh $L/jobs_sfw_g15_miluv.txt >> $LOG 2>&1
SFUISE_PORT=13920 VARIANT=_w20_g15 $R/sfuise_toa_batch.sh $L/jobs_sfw_g15_ntu.txt >> $LOG 2>&1
echo "$(date +%T) SHORT_GATE_DONE" >> $LOG
