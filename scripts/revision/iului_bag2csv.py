#!/usr/bin/env python3
"""IULUI converted bag (mppi_smoothing/data/conv/<cfg>_<name>.bag from convert0806.py: /mavros/imu/data, /nlink_linktrack_tagframe0
with dis_arr[0..3] = anchors 0-3 already bias-corrected, /qualisys/gogogo/pose) -> KIMS csv input in data/iului/<out>/:
  imu.csv        header stamp, angular_velocity xyz, linear_acceleration xyz
  uwb_range.csv  bag time (10 ms grid shared by both tags), from_id = tag id, to_id = anchor index, range, std 0.1 (zero = no range)
  mocap.csv      bag time (microseconds), position xyz, orientation xyzw
and copies the sidecar json (anchors, tag ids/offsets, initial position) to data/iului/<out>.json.
Reproduces the existing AC1 csv files exactly (checked on SquareTurn).   usage: iului_bag2csv.py <conv bag> <out name> [out root]"""
import sys, os, json, shutil, rosbag
bag, name = sys.argv[1], sys.argv[2]; root = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/iului")
out = os.path.join(root, name); os.makedirs(out, exist_ok=True)
fi = open(f"{out}/imu.csv", "w"); fu = open(f"{out}/uwb_range.csv", "w"); fm = open(f"{out}/mocap.csv", "w")
fi.write("timestamp,angular_velocity.x,angular_velocity.y,angular_velocity.z,linear_acceleration.x,linear_acceleration.y,linear_acceleration.z\n")
fu.write("timestamp,from_id,to_id,range,std\n")
fm.write("timestamp,pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w\n")
n = dict(imu=0, uwb=0, gt=0)
for topic, m, t in rosbag.Bag(bag).read_messages(topics=["/mavros/imu/data", "/nlink_linktrack_tagframe0", "/qualisys/gogogo/pose"]):
    if topic == "/mavros/imu/data":
        g, a = m.angular_velocity, m.linear_acceleration
        fi.write(",".join(repr(v) for v in [m.header.stamp.to_sec(), g.x, g.y, g.z, a.x, a.y, a.z]) + "\n"); n["imu"] += 1
    elif topic == "/nlink_linktrack_tagframe0":
        for k in range(4):
            if m.dis_arr[k] > 0.0: fu.write(f"{repr(t.to_sec())},{m.id},{k},{repr(float(m.dis_arr[k]))},0.1\n"); n["uwb"] += 1
    else:
        p, q = m.pose.position, m.pose.orientation
        fm.write(f"{t.to_sec():.6f}," + ",".join(repr(v) for v in [p.x, p.y, p.z, q.x, q.y, q.z, q.w]) + "\n"); n["gt"] += 1
for f in (fi, fu, fm): f.close()
shutil.copy(bag.replace(".bag", ".json"), os.path.join(root, f"{name}.json"))
print(name, n)
