#!/bin/bash
jobs=$1; shift; PAUSE="$*"
cd "$(dirname "$0")/../.."; B=${SFUISE_BAGS:-$HOME/util_raw/kims_bags}; port=${SFUISE_PORT:-13600}
[ -n "$PAUSE" ] && kill -STOP $PAUSE
trap '[ -n "$PAUSE" ] && kill -CONT $PAUSE' EXIT
while read -r ds seq <&3; do
  out=result/$ds/sfuise${VARIANT}_${seq}_pose.txt
  for att in 1 2 3; do
    port=$((port + 1)); rm -f $out
    baselines/sfuise/sfuise_run.sh ${ds}_${seq}${VARIANT}_a$att $B/${ds}_${seq}.bag $B/${ds}_${seq}${VARIANT}.yaml $port $out < /dev/null > /dev/null 2>&1
    n=$(grep -vc '^#' $out 2>/dev/null); n=${n:-0}
    echo "$ds $seq attempt $att: $n poses"
    [ "$n" -gt 100 ] && break
  done
done 3< $jobs
echo SFUISE_TOA_BATCH_DONE
