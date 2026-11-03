#!/bin/bash
# Download, prepare (CSV only, archive and bag deleted right after extraction), run and score NTU sequences one after
# another. usage: ntu_chain.sh <seq:datafile_id> ...
cd "$(dirname "$0")/../.."; mkdir -p ${NTU_RAW:-$HOME/ntu_raw}
for item in "$@"; do
  seq=${item%%:*}; id=${item##*:}
  curl -L -s -o ${NTU_RAW:-$HOME/ntu_raw}/$seq.zip https://researchdata.ntu.edu.sg/api/access/datafile/$id
  scripts/revision/ntu_new_run.sh $seq >> result/runlogs/ntu_new_$seq.log 2>&1
done
echo CHAIN_DONE
