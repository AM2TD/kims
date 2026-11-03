#!/usr/bin/env python3
"""Physically-impossible TDOA filter for the SFUISE runs, the same rule util_prepare.py applies for KIMS:
drop |tdoa| > |a_A - a_B| + 0.5 m (a TDOA can never exceed the anchor separation). Keeps /imu_data, /tdoa_data, /pose_data.
usage: util_filter_bag.py <seq>   -> $UTIL_ROOT/filtered_bags/<seq>.bag (default ~/util_raw)   (run with ROS noetic sourced)"""
import sys, os, numpy as np, rosbag
UR = os.environ.get("UTIL_ROOT", os.path.expanduser("~/util_raw")); seq = sys.argv[1]; const = seq.split("-")[0]
A = {}
for l in open(f"{UR}/dataset/flight-dataset/survey-results/anchor_{const}_survey.txt"):
    k, *v = l.strip().split(",")
    if k.endswith("_p"): A[int(k[2:-2])] = np.array([float(x) for x in v])
os.makedirs(f"{UR}/filtered_bags", exist_ok=True)
n_in = n_drop = 0
with rosbag.Bag(f"{UR}/filtered_bags/{seq}.bag", "w") as out:
    for topic, msg, t in rosbag.Bag(f"{UR}/dataset/flight-dataset/rosbag-data/{const}/{seq}.bag").read_messages(topics=["/imu_data", "/tdoa_data", "/pose_data"]):
        if topic == "/tdoa_data":
            n_in += 1
            if abs(msg.data) > np.linalg.norm(A[msg.idA] - A[msg.idB]) + 0.5: n_drop += 1; continue
        out.write(topic, msg, t)
print(f"{seq}: dropped {n_drop} of {n_in} TDOA")
