#!/bin/bash
set -e
cd "$(dirname "$0")/.."
DATASETS="${*:-miluv ntu util iului}"; P=${KIMS_PARALLEL:-3}
J=$(mktemp); trap 'rm -f $J' EXIT
for ds in $DATASETS; do
  for m in ${METHODS:-kims}; do
    for s in $(python3 -c "import sys; sys.path.insert(0, 'scripts'); from run import sequences; print(' '.join(sequences('$ds')))"); do
      case $ds in miluv) d=data/miluv/$s;; *) d=data/$ds/$s;; esac
      [ -d "$d" ] && echo "python3 scripts/run.py $m $ds $s" >> $J
    done
  done
done
echo "$(date +%T) $(wc -l < $J) runs, $P at a time"
xargs -P $P -I{} bash -c '{}' < $J
echo "$(date +%T) runs done"
python3 scripts/evaluate.py ${METHODS:-kims}
