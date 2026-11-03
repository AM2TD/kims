#!/usr/bin/env python3
"""Per-window SFUISE processing time. SplineFusion publishes on est_window, once per window (10 Hz knots), the running mean
of its own window timer (TicToc over initialization + optimization + publishing, ms). The per-window values follow exactly
from t_n = n * mean_n - (n - 1) * mean_(n-1).  usage: sfuise_time_logger.py <out_ms_file>   (ROS_MASTER_URI of the run)"""
import sys, rospy
from sfuise_msgs.msg import Estimate
out = sys.argv[1]; avg = []
def cb(m): avg.append(m.runtime.data)
def dump():
    with open(out, "w") as f:
        f.write(f"# SFUISE per-window time [ms], {len(avg)} windows (from the published running mean)\n")
        prev = 0.0
        for n, a in enumerate(avg, 1): f.write(f"{n * a - (n - 1) * prev:.4f}\n"); prev = a
rospy.init_node("sfuise_time_logger", anonymous=True)
rospy.Subscriber("/SplineFusion/est_window", Estimate, cb, queue_size=1000)
rospy.on_shutdown(dump)
rospy.spin()
