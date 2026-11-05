#!/bin/bash
seq=$1; port=$2; const=${seq%%-*}
CFG=${SFUISE_CFG:-\$(find sfuise)/config/config_test_util.yaml}; PRE=${SFUISE_PREFIX:-sfuisetag_}
UR=${UTIL_ROOT:-$HOME/util_raw}; K=$(cd "$(dirname "$0")/../.." && pwd); WS=${SFUISE_WS:-$HOME/util_raw/sfuise_ws}; LOG=${SFUISE_LOGS:-$HOME/util_raw/sfuise_logs}/${SFUISE_PREFIX}$seq; mkdir -p $LOG
source /opt/ros/noetic/setup.bash; source $WS/devel/setup.bash
export ROS_MASTER_URI=http://localhost:$port ROS_LOG_DIR=$LOG ROS_HOME=$LOG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1
AN=$UR/dataset/flight-dataset/survey-results/anchor_${const}_survey.txt
cat > $LOG/run.launch <<EOL
<launch>
  <node pkg="sfuise" type="EstimationInterface" name="EstimationInterface" output="log">
    <rosparam command="load" file="$CFG"/>
    <param name="anchor_path" value="$AN"/>
  </node>
  <node pkg="sfuise" type="SplineFusion" name="SplineFusion" output="log">
    <rosparam command="load" file="$CFG"/>
    <param name="anchor_path" value="$AN"/>
  </node>
</launch>
EOL
roslaunch -p $port $LOG/run.launch > $LOG/launch.out 2>&1 &
LP=$!
trap 'kill -INT $GP $LP 2>/dev/null; sleep 3; pkill -9 -f "rosmaster --core -p $port" 2>/dev/null; pkill -9 -P $LP 2>/dev/null' EXIT
n=0; until rostopic list > /dev/null 2>&1; do sleep 0.5; n=$((n+1)); [ $n -gt 120 ] && { echo "${seq} SFUISE_FAILED master did not start on port $port"; exit 1; }; done; sleep 2
python3 $K/baselines/sfuise/util_sfuise_logger.py $K/result/util/${PRE}${seq}_pose.txt > $LOG/logger.out 2>&1 &
GP=$!; sleep 2
rosbag play --quiet $UR/filtered_bags/$seq.bag --topics /imu_data /tdoa_data /pose_data > $LOG/play.out 2>&1
sleep 8
kill -INT $GP; sleep 3; kill -INT $LP; wait $LP 2>/dev/null
echo "${seq} SFUISE_DONE $(grep -c . $K/result/util/${PRE}${seq}_pose.txt) lines"
