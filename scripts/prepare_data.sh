#!/bin/bash
set -e
cd "$(dirname "$0")/.."
MILUV="default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
NTU="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
case $1 in
  miluv) python3 scripts/prepare/fetch_miluv.py $MILUV --robots ifo001;;
  ntu)   [ -d "$2" ] || { echo "usage: $0 ntu <directory with eee_01.zip ... sbs_03.zip>"; exit 1; }
         for s in $NTU; do scripts/prepare/ntu_prepare.sh $s "$(cd "$2" && pwd)/$s.zip"; done;;
  util)  C="$2/dataset/flight-dataset/csv-data"; [ -d "$C" ] || { echo "usage: $0 util <UTIL root containing dataset/flight-dataset/csv-data>"; exit 1; }
         for f in "$C"/const*/*.csv; do python3 scripts/prepare/util_prepare.py "$f"; done;;
  *) echo "usage: $0 miluv | ntu <zip dir> | util <UTIL root>"; exit 1;;
esac
