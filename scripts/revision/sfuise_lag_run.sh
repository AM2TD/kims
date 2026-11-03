#!/bin/bash
# SFUISE run with the output-lag logger (sfuise_lag_logger.py) and the per-window time logger, same settings as the
# accuracy runs. 1x playback, private ROS master.
# usage: sfuise_lag_run.sh <miluv|ntu|util> <seq> <port>  -> result/<ds>/diag/sfuiselag_<seq>_lag<L>_pose.txt
ds=$1; seq=$2; port=$3; UR=${UTIL_ROOT:-$HOME/util_raw}; K=$(cd "$(dirname "$0")/../.." && pwd); B=${SFUISE_BAGS:-$UR/kims_bags}
LOG=$UR/sfuise_logs/lag_$seq; mkdir -p $LOG; out=$K/result/$ds/diag/sfuiselag_${seq}; tf=$K/result/$ds/diag/sfuiselag_${seq}_ms.txt
source /opt/ros/noetic/setup.bash; source ${SFUISE_WS:-$UR/sfuise_ws}/devel/setup.bash
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
python3 $K/scripts/revision/sfuise_lag_logger.py $out > $LOG/logger.out 2>&1 &
GP=$!
python3 $K/scripts/revision/sfuise_time_logger.py $tf > $LOG/time_logger.out 2>&1 &
TP=$!; sleep 3
rosbag play --quiet $bag --topics $TOP > $LOG/play.out 2>&1
sleep 8
kill -INT $GP $TP; sleep 3; kill -INT $LP; wait $LP 2>/dev/null
echo "$seq SFUISE_LAG_DONE $(grep -vc "^#" ${out}_lag0_pose.txt) stamps, $(grep -vc "^#" $tf) windows"
