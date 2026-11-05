#!/usr/bin/env python3

import argparse
import os
import rosbag

RANGE_BIAS = -0.75

def extract(bag_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    imu_path   = os.path.join(out_dir, 'imu.csv')
    uwb_path   = os.path.join(out_dir, 'uwb_range.csv')
    leica_path = os.path.join(out_dir, 'leica.csv')

    bag = rosbag.Bag(bag_path)

    topics = ['/imu/imu', '/uwb_endorange_info', '/leica/pose/relative']
    info = bag.get_type_and_topic_info().topics
    for t in topics:
        n = info[t].message_count if t in info else 0
        print(f'  {t}: {n} messages')

    with open(imu_path, 'w') as f:
        f.write('timestamp,'
                'angular_velocity.x,angular_velocity.y,angular_velocity.z,'
                'linear_acceleration.x,linear_acceleration.y,linear_acceleration.z\n')
        for _, msg, _ in bag.read_messages(topics=['/imu/imu']):
            ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            gx = msg.angular_velocity.x
            gy = msg.angular_velocity.y
            gz = msg.angular_velocity.z
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            f.write(f'{ts:.9f},{gx:.9f},{gy:.9f},{gz:.9f},{ax:.9f},{ay:.9f},{az:.9f}\n')
    print(f'[ok] IMU written → {imu_path}')

    UAV_NODES   = {200, 201}
    ANCHOR_IDS  = {100, 101, 102}

    with open(uwb_path, 'w') as f:
        f.write('timestamp,range,from_id,to_id,std\n')
        for _, msg, _ in bag.read_messages(topics=['/uwb_endorange_info']):
            req_id = msg.requester_id
            ant    = msg.antenna
            rsp_id = msg.responder_id
            if req_id not in UAV_NODES or rsp_id not in ANCHOR_IDS:
                continue
            ts       = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            from_id  = req_id * 10 + ant
            rng      = msg.distance + RANGE_BIAS
            std_dev  = max(msg.distance_err, 0.01)
            f.write(f'{ts:.9f},{rng:.6f},{from_id},{rsp_id},{std_dev:.6f}\n')
    print(f'[ok] UWB written → {uwb_path}')

    with open(leica_path, 'w') as f:
        f.write('timestamp,x,y,z\n')
        for _, msg, _ in bag.read_messages(topics=['/leica/pose/relative']):
            ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            x  = msg.pose.position.x
            y  = msg.pose.position.y
            z  = msg.pose.position.z
            f.write(f'{ts:.9f},{x:.9f},{y:.9f},{z:.9f}\n')
    print(f'[ok] Leica GT written → {leica_path}')

    bag.close()
    print('[done] Extraction complete.')

def main():
    parser = argparse.ArgumentParser(description='Extract NTU VIRAL bag to CSV files.')
    parser.add_argument('--bag', required=True, help='Path to .bag file')
    parser.add_argument('--out', default=None,
                        help='Output directory (default: same dir as bag file + /data)')
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.bag)), 'data')

    print(f'Extracting: {args.bag}  →  {args.out}')
    extract(args.bag, args.out)

if __name__ == '__main__':
    main()
