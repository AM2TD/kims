#!/usr/bin/env python3
"""SFUISE estimates at fixed output lags. At every 10 Hz window solve SFUISE re-evaluates all GT stamps in its 10 s window;
the accuracy runs (util_sfuise_logger.py) kept the LAST estimate before a stamp left the window, i.e. an output lag of about
10 s. This logger keeps, for each lag L, the first estimate made once the window end (newest GT stamp in the window) had
passed the stamp by L seconds; L = 0 is the first estimate of the stamp. Stamps that the end of the data leaves short of L
take the last estimate (KIMS also flushes its last window).
usage: sfuise_lag_logger.py <out_prefix> [lags s ...]  -> <out_prefix>_lag<L>_pose.txt, <out_prefix>_lastwin_pose.txt"""
import sys, rospy
from nav_msgs.msg import Path
out = sys.argv[1]; LAGS = [float(x) for x in sys.argv[2:]] or [0.0, 0.5, 1.0, 2.0, 3.3, 5.0]
est = {L: {} for L in LAGS}; last = {}; n = [0]
def cb(m):
    if not m.poses: return
    n[0] += 1; st = [p.header.stamp.to_nsec() for p in m.poses]; E = max(st)
    for p, t in zip(m.poses, st):
        q = p.pose.orientation; x = p.pose.position; v = (x.x, x.y, x.z, q.x, q.y, q.z, q.w); last[t] = v
        for L in LAGS:
            if t not in est[L] and E - t >= L * 1e9: est[L][t] = v
def dump():
    for L in LAGS + [None]:
        d = last if L is None else est[L]; fn = f"{out}_{'lastwin' if L is None else 'lag%g' % L}_pose.txt"
        with open(fn, "w") as f:
            f.write(f"# time x y z qx qy qz qw  (SFUISE tag position, output lag {'last window' if L is None else L} s, {n[0]} window messages)\n")
            for k in sorted(last): f.write(f"{k * 1e-9:.6f} " + " ".join(f"{a:.6f}" for a in d.get(k, last[k])) + "\n")
rospy.init_node("sfuise_lag_logger", anonymous=True)
rospy.Subscriber("/EstimationInterface/bspline_optimization_window", Path, cb, queue_size=200)
rospy.on_shutdown(dump)
rospy.spin()
