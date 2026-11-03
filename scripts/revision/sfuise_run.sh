#!/bin/bash
# One SFUISE run (patched workspace $SFUISE_WS, default ~/util_raw/sfuise_ws; baselines/sfuise) on a bag made by sfuise_make_bag.py, private ROS master,
# 1x playback, Release build, single-thread BLAS; output via util_sfuise_logger.py (last window estimate per GT stamp).
# usage: sfuise_run.sh <name> <bag> <config.yaml> <port> <out_pose_file>
name=$1; bag=$2; cfg=$3; port=$4; out=$5
K=$(cd "$(dirname "$0")/../.." && pwd); WS=${SFUISE_WS:-$HOME/util_raw/sfuise_ws}; LOG=${SFUISE_LOGS:-$HOME/util_raw/sfuise_logs}/$name; mkdir -p $LOG
source /opt/ros/noetic/setup.bash; source $WS/devel/setup.bash
export ROS_MASTER_URI=http://localhost:$port ROS_LOG_DIR=$LOG ROS_HOME=$LOG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1
cat > $LOG/run.launch <<EOL
<launch>
  <node pkg="sfuise" type="EstimationInterface" name="EstimationInterface" output="log"><rosparam command="load" file="$cfg"/></node>
  <node pkg="sfuise" type="SplineFusion" name="SplineFusion" output="log"><rosparam command="load" file="$cfg"/></node>
</launch>
EOL
roslaunch -p $port $LOG/run.launch > $LOG/launch.out 2>&1 &
LP=$!
trap 'kill -INT $GP $LP 2>/dev/null; sleep 3; pkill -9 -f "rosmaster --core -p $port" 2>/dev/null; pkill -9 -P $LP 2>/dev/null' EXIT
n=0; until rostopic list > /dev/null 2>&1; do sleep 0.5; n=$((n+1)); [ $n -gt 120 ] && { echo "$name SFUISE_FAILED master did not start on port $port"; exit 1; }; done; sleep 2
python3 $K/scripts/revision/util_sfuise_logger.py $out > $LOG/logger.out 2>&1 &
GP=$!; sleep 2
rosbag play --quiet $bag --topics /imu_data /rtls_flares /anchor_list /pose_data > $LOG/play.out 2>&1
sleep 8
kill -INT $GP; sleep 3; kill -INT $LP; wait $LP 2>/dev/null
echo "$name SFUISE_DONE $(grep -c . $out) lines"
