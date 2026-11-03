#!/bin/bash
# Fetch or convert the public datasets into data/<dataset>/ (IULUI ships with the repository).
#   scripts/prepare_data.sh miluv                 # 13 single-robot MILUV experiments (ifo001 CSVs only, ~350 MB, from figshare)
#   scripts/prepare_data.sh ntu  <dir with zips>  # NTU VIRAL eee/nya/sbs 01-03 from the downloaded <seq>.zip archives (needs ROS rosbag)
#   scripts/prepare_data.sh util <UTIL root>      # all 79 UTIL flights from the unpacked UTIL release (<root>/dataset/flight-dataset/...)
# The anchor files (data/miluv/anchors_constellation*.txt, data/ntu/anchors_*.txt) and NTU start poses
# (data/ntu/init_poses.json) ship with the repository. MILUV constellation 2 uses the corrected anchor order (README).
set -e
cd "$(dirname "$0")/.."
MILUV="default_1_remoteControlHighPace_1 default_1_random3_1 default_1_random3_0 default_1_circular3D_0 default_1_remoteControlLowPace_0_v2 default_1_remoteControlLowPace_0 default_1_circular2D_0 default_1_remoteControlHighPace_0 obstacles_1_random3_0b default_1_random3_2 default_1_remoteControlHighPace_2 noAprilTags_1_random3_2b noAprilTags_1_remoteControlHighPace_2b"
NTU="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
case $1 in
  miluv) python3 scripts/fetch_miluv.py $MILUV --robots ifo001;;
  ntu)   [ -d "$2" ] || { echo "usage: $0 ntu <directory with eee_01.zip ... sbs_03.zip>"; exit 1; }
         for s in $NTU; do scripts/revision/ntu_prepare.sh $s "$(cd "$2" && pwd)/$s.zip"; done;;
  util)  C="$2/dataset/flight-dataset/csv-data"; [ -d "$C" ] || { echo "usage: $0 util <UTIL root containing dataset/flight-dataset/csv-data>"; exit 1; }
         for f in "$C"/const*/*.csv; do python3 scripts/revision/util_prepare.py "$f"; done;;
  *) echo "usage: $0 miluv | ntu <zip dir> | util <UTIL root>"; exit 1;;
esac
