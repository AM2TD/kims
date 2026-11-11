#!/usr/bin/env python3
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
print(f"{seq}: dropped {n_drop} of {n_in} TDoA")
