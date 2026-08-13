#!/usr/bin/env python3
# @file replay.py
# @brief Replay a dataset sequence as ROS topics, so the online node can run on
#        any of the three datasets and not only the one distributed as bags.

import csv
import os
import sys

import rospy
from kims.msg import ImuSample, UwbRange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CONFIG

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path, cols):
    """Return rows as float tuples for the requested column names."""
    out = []
    with open(path) as fh:
        rd = csv.reader(fh)
        header = [h.strip() for h in next(rd)]
        idx = [header.index(c) for c in cols]
        for row in rd:
            if not row or row[0].startswith("#"):
                continue
            try:
                out.append(tuple(float(row[i]) for i in idx))
            except (ValueError, IndexError):
                continue
    return out


def publish(pub, msg):
    """A subscriber going away must not abort the replay."""
    try:
        pub.publish(msg)
        return True
    except rospy.ROSException:
        return False


def main():
    rospy.init_node("replay")
    dataset = rospy.get_param("~dataset", "iului")
    sequence = rospy.get_param("~sequence", "")
    rate_mult = float(rospy.get_param("~rate", 1.0))
    loop = bool(rospy.get_param("~loop", False))

    ds = CONFIG[dataset]
    if not sequence:
        sequence = ds["sequences"][0]
    data_dir = os.path.join(PKG, ds["data_dir"].format(seq=sequence))

    imu_name = "imu_px4.csv" if dataset == "miluv" else "imu.csv"
    imu_path = os.path.join(data_dir, imu_name)
    uwb_path = os.path.join(data_dir, "uwb_range.csv")
    for p in (imu_path, uwb_path):
        if not os.path.isfile(p):
            rospy.logfatal("[replay] missing %s", p)
            return

    imu_rows = read_csv(imu_path, [
        "timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"])
    uwb_rows = read_csv(uwb_path, ["timestamp", "range", "from_id", "to_id", "std"])

    # The recorded rows are not always in time order (NTU has backward steps);
    # the offline loader sorts them, so the replay must too.
    imu_rows.sort(key=lambda r: r[0])
    uwb_rows.sort(key=lambda r: r[0])

    tag_ids = set(ds["tag_ids"])
    anchor_ids = set(ds.get("anchor_ids", range(ds.get("num_anchors", 0))))
    uwb_rows = [r for r in uwb_rows
                if int(r[2]) in tag_ids and (not anchor_ids or int(r[3]) in anchor_ids)]

    if not imu_rows or not uwb_rows:
        rospy.logfatal("[replay] nothing to replay (imu %d, uwb %d)",
                       len(imu_rows), len(uwb_rows))
        return

    pub_imu = rospy.Publisher("imu_sample", ImuSample, queue_size=1000)
    pub_uwb = rospy.Publisher("uwb", UwbRange, queue_size=1000)

    # Do not start until the estimator is actually listening: a message dropped
    # here changes the first window and the run stops matching the offline one.
    wait_for = float(rospy.get_param("~wait_for_subscriber", 30.0))
    t_wait = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        if pub_imu.get_num_connections() > 0 and pub_uwb.get_num_connections() > 0:
            rospy.sleep(0.5)      # let the connections settle
            break
        if rospy.Time.now().to_sec() - t_wait > wait_for:
            rospy.logwarn("[replay] no subscriber after %.0f s, replaying anyway", wait_for)
            break
        rospy.sleep(0.05)

    rospy.loginfo("[replay] %s/%s: %d IMU, %d UWB rows, rate x%.1f",
                  dataset, sequence, len(imu_rows), len(uwb_rows), rate_mult)

    while not rospy.is_shutdown():
        t0 = min(imu_rows[0][0], uwb_rows[0][0])
        wall0 = rospy.Time.now().to_sec()
        i = j = 0

        while not rospy.is_shutdown() and (i < len(imu_rows) or j < len(uwb_rows)):
            t_imu = imu_rows[i][0] if i < len(imu_rows) else float("inf")
            t_uwb = uwb_rows[j][0] if j < len(uwb_rows) else float("inf")
            t_next = min(t_imu, t_uwb)

            target = wall0 + (t_next - t0) / max(rate_mult, 1e-6)
            sleep = target - rospy.Time.now().to_sec()
            if sleep > 0:
                rospy.sleep(sleep)

            # keep the recorded stamp: rows of one UWB epoch share it
            stamp = rospy.Time.from_sec(t_next)
            if t_imu <= t_uwb:
                r = imu_rows[i]; i += 1
                m = ImuSample()
                m.header.stamp = stamp
                m.header.frame_id = "imu"
                m.stamp = r[0]                      # exact sensor time
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = r[1], r[2], r[3]
                m.linear_acceleration.x = r[4]
                m.linear_acceleration.y = r[5]
                m.linear_acceleration.z = r[6]
                publish(pub_imu, m)
            else:
                r = uwb_rows[j]; j += 1
                m = UwbRange()
                m.header.stamp = stamp
                m.stamp   = r[0]                    # exact sensor time
                m.from_id = int(r[2])
                m.to_id   = int(r[3])
                m.range   = r[1]
                m.std     = r[4]
                publish(pub_uwb, m)

        rospy.loginfo("[replay] sequence finished")
        if not loop:
            break

    rospy.signal_shutdown("replay complete")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
