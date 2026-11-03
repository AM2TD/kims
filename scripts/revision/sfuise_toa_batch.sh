#!/bin/bash
# SFUISE on MILUV / NTU (bags from sfuise_make_bag.py), one run at a time with nothing else running: the given PIDs are
# paused (SIGSTOP) for the batch and resumed at the end. A run is retried up to 3 times when SplineFusion crashes at the
# start (a race in its initialisation, no output); the attempt count is logged.
# usage: [VARIANT=_g15] sfuise_toa_batch.sh <jobs file: "dataset seq" per line> [pids to pause...]
# VARIANT selects <dataset>_<seq><VARIANT>.yaml and writes sfuise<VARIANT>_<seq>_pose.txt (default: published config).
jobs=$1; shift; PAUSE="$*"
cd "$(dirname "$0")/../.."; B=${SFUISE_BAGS:-$HOME/util_raw/kims_bags}; port=${SFUISE_PORT:-13600}
[ -n "$PAUSE" ] && kill -STOP $PAUSE
trap '[ -n "$PAUSE" ] && kill -CONT $PAUSE' EXIT
while read -r ds seq <&3; do
  out=result/$ds/diag/sfuise${VARIANT}_${seq}_pose.txt
  for att in 1 2 3; do
    port=$((port + 1)); rm -f $out
    scripts/revision/sfuise_run.sh ${ds}_${seq}${VARIANT}_a$att $B/${ds}_${seq}.bag $B/${ds}_${seq}${VARIANT}.yaml $port $out < /dev/null > /dev/null 2>&1
    n=$(grep -vc '^#' $out 2>/dev/null); n=${n:-0}
    echo "$ds $seq attempt $att: $n poses"
    [ "$n" -gt 100 ] && break
  done
done 3< $jobs
echo SFUISE_TOA_BATCH_DONE
