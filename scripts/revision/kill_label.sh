#!/bin/bash
# Stop the kims process writing result/miluv/diag/<label>_pose.txt (and its hp_mix.sh wrapper).
label=$1
pkill -f "diag/${label}_pose[.]txt"
pkill -f "hp_mix[.]sh ${label} "
sleep 1
echo "left: $(pgrep -fc "diag/${label}_pose")"
