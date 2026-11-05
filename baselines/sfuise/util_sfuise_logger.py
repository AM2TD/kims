#!/usr/bin/env python3
import sys, rospy
from nav_msgs.msg import Path
out = sys.argv[1]; est = {}; n = [0]
def cb(m):
    n[0] += 1
    for p in m.poses:
        q = p.pose.orientation; x = p.pose.position
        est[p.header.stamp.to_nsec()] = (x.x, x.y, x.z, q.x, q.y, q.z, q.w)
def dump():
    with open(out, "w") as f:
        f.write(f"# time x y z qx qy qz qw  (SFUISE, last window estimate per GT stamp, {n[0]} window messages)\n")
        for k in sorted(est): f.write(f"{k * 1e-9:.6f} " + " ".join(f"{v:.6f}" for v in est[k]) + "\n")
rospy.init_node("sfuise_logger", anonymous=True)
rospy.Subscriber("/EstimationInterface/bspline_optimization_window", Path, cb, queue_size=50)
rospy.on_shutdown(dump)
rospy.spin()
