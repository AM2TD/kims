#!/bin/bash
set -e
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$WS"
source /opt/ros/noetic/setup.bash
catkin_make -j"$(nproc)"
source devel/setup.bash
python3 src/kims/scripts/run.py --method all --dataset all
python3 src/kims/scripts/plot.py
