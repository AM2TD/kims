#!/bin/bash
# Standalone SFUISE timing run (same settings as the accuracy runs: MILUV 1.5 m-gate config, NTU published ToA config,
# UTIL published config) with the pose logger and the per-window time logger. 1x playback, private ROS master.
# usage: timing_sfuise.sh <miluv|ntu|util> <seq> <port>  -> result/timing/sfuise_<seq>_{pose,ms}.txt
cd "$(dirname "$0")/../.."
ds=$1; seq=$2; port=$3; UR=${UTIL_ROOT:-$HOME/util_raw}; K=$PWD; B=$UR/kims_bags
LOG=$UR/sfuise_logs/timing_$seq; mkdir -p $LOG; out=$K/result/timing/sfuise_${seq}_pose.txt; tf=$K/result/timing/sfuise_${seq}_ms.txt
source /opt/ros/noetic/setup.bash; source $UR/sfuise_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:$port ROS_LOG_DIR=$LOG ROS_HOME=$LOG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1
if [ $ds = util ]; then
  AN=$UR/dataset/flight-dataset/survey-results/anchor_${seq%%-*}_survey.txt; bag=$UR/filtered_bags/$seq.bag; TOP="/imu_data /tdoa_data /pose_data"
  P="<rosparam command=\"load\" file=\"\$(find sfuise)/config/config_test_util.yaml\"/><param name=\"anchor_path\" value=\"$AN\"/>"
else
  V=""; [ $ds = miluv ] && V=_g15; bag=$B/${ds}_${seq}.bag; TOP="/imu_data /rtls_flares /anchor_list /pose_data"
  P="<rosparam command=\"load\" file=\"$B/${ds}_${seq}${V}.yaml\"/>"
fi
cat > $LOG/run.launch <<EOL
<launch>
  <node pkg="sfuise" type="EstimationInterface" name="EstimationInterface" output="log">$P</node>
  <node pkg="sfuise" type="SplineFusion" name="SplineFusion" output="log">$P</node>
</launch>
EOL
roslaunch -p $port $LOG/run.launch > $LOG/launch.out 2>&1 &
LP=$!
trap 'kill -INT $GP $TP $LP 2>/dev/null; sleep 3; pkill -9 -f "rosmaster --core -p $port" 2>/dev/null; pkill -9 -P $LP 2>/dev/null' EXIT
n=0; until rostopic list > /dev/null 2>&1; do sleep 0.5; n=$((n+1)); [ $n -gt 120 ] && { echo "$seq SFUISE_FAILED master did not start on port $port"; exit 1; }; done; sleep 2
python3 $K/scripts/revision/util_sfuise_logger.py $out > $LOG/logger.out 2>&1 &
GP=$!
python3 $K/scripts/revision/sfuise_time_logger.py $tf > $LOG/time_logger.out 2>&1 &
TP=$!; sleep 3
rosbag play --quiet $bag --topics $TOP > $LOG/play.out 2>&1
sleep 8
kill -INT $GP $TP; sleep 3; kill -INT $LP; wait $LP 2>/dev/null
echo "$seq SFUISE_DONE $(grep -vc '^#' $out) poses, $(grep -vc '^#' $tf) windows"
