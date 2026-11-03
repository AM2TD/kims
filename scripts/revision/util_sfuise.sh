#!/bin/bash
# SFUISE (Li, Cao, Hanebeck RA-L 2023, github.com/ASIG-X/SFUISE, built in $SFUISE_WS, default ~/util_raw/sfuise_ws) on one UTIL flight
# with its published UTIL configuration (config_test_util.yaml; only anchor_path set to the flight's constellation).
# The bag (impossible TDOA removed by util_filter_bag.py, the rule used for KIMS) is played at 1x on a private ROS
# master; output result/util/diag/sfuise_<seq>_pose.txt. Release build (-O3 -DNDEBUG) and single-thread BLAS: with the
# repository Debug build and threaded BLAS it ran ~9x slower than real time on this machine and dropped data.
# usage: [SFUISE_CFG=<yaml>] [SFUISE_PREFIX=sfuise_w20_] util_sfuise.sh <seq> <port>   (defaults: the published config, sfuise_)
seq=$1; port=$2; const=${seq%%-*}
CFG=${SFUISE_CFG:-\$(find sfuise)/config/config_test_util.yaml}; PRE=${SFUISE_PREFIX:-sfuise_}
UR=${UTIL_ROOT:-$HOME/util_raw}; K=$(cd "$(dirname "$0")/../.." && pwd); WS=${SFUISE_WS:-$HOME/util_raw/sfuise_ws}; LOG=${SFUISE_LOGS:-$HOME/util_raw/sfuise_logs}/${SFUISE_PREFIX}$seq; mkdir -p $LOG
source /opt/ros/noetic/setup.bash; source $WS/devel/setup.bash
export ROS_MASTER_URI=http://localhost:$port ROS_LOG_DIR=$LOG ROS_HOME=$LOG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1   # cholmod BLAS threads oversubscribe otherwise
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
python3 $K/scripts/revision/util_sfuise_logger.py $K/result/util/diag/${PRE}${seq}_pose.txt > $LOG/logger.out 2>&1 &
GP=$!; sleep 2
rosbag play --quiet $UR/filtered_bags/$seq.bag --topics /imu_data /tdoa_data /pose_data > $LOG/play.out 2>&1   # util_filter_bag.py: impossible TDOA removed
sleep 8
kill -INT $GP; sleep 3; kill -INT $LP; wait $LP 2>/dev/null
echo "${seq} SFUISE_DONE $(grep -c . $K/result/util/diag/${PRE}${seq}_pose.txt) lines"
