#!/bin/bash
set -e
cd "$(dirname "$0")/../.."; K=$PWD
seq=$1; zip=$2; raw=${NTU_RAW:-$HOME/ntu_raw}/$seq
mkdir -p $raw data/ntu/$seq
if [ ! -f $raw/$seq.bag ]; then
  cd $raw && unzip -o -q $zip
  for i in 1 2; do b=$(find $raw -name '*.bag' | head -1); [ -n "$b" ] && break; z=$(find $raw -name '*.zip' | head -1); [ -n "$z" ] && unzip -o -q "$z" -d $raw && rm -f "$z"; done
  b=$(find $raw -name '*.bag' | head -1); [ -n "$b" ] || { echo "no bag in $zip"; exit 1; }; [ "$b" = "$raw/$seq.bag" ] || mv "$b" $raw/$seq.bag
  cd $K
fi
[ -s data/ntu/$seq/leica.csv ] || python3 scripts/prepare/ntu_extract_bag.py --bag $raw/$seq.bag --out data/ntu/$seq
python3 - <<PY
import rosbag, numpy as np
bag = rosbag.Bag('$raw/$seq.bag'); pos = {}
for _, msg, _ in bag.read_messages(topics=['/node_pos_sc']):
    if hasattr(msg, 'poses'):
        for ps in msg.poses:
            pos[int(ps.header.seq)] = (float(ps.pose.position.x), float(ps.pose.position.y), float(ps.pose.position.z))
    for f in ('nodes', 'node', 'data'):
        if hasattr(msg, f):
            for n in getattr(msg, f):
                nid = getattr(n, 'id', None) if hasattr(n, 'id') else None
                if nid is None: continue
                p = getattr(n, 'pos', None) or getattr(n, 'position', None)
                if p is None and hasattr(n, 'x'): p = n
                if p is not None: pos[int(nid)] = (float(p.x), float(p.y), float(p.z))
    if len(pos) >= 3: break
if not pos:
    for _, msg, _ in bag.read_messages(topics=['/node_pos_sc']): print(msg); break
    raise SystemExit('anchor extraction: unknown message layout')
with open('data/ntu/anchors_$seq.txt', 'w') as f:
    f.write('# NTU VIRAL anchor positions from /node_pos_sc in $seq.bag (UWB frame W), ID order 100 101 102\n')
    for i in (100, 101, 102): f.write('%.6f %.6f %.6f\n' % pos[i])
print('anchors', {i: pos[i] for i in (100, 101, 102)})
PY
python3 scripts/prepare/ntu_still_init.py $seq data/ntu/anchors_$seq.txt
