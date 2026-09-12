#!/usr/bin/env python3
"""Convert a rosbag of /joint_states into a CSV: one row per message,
columns `t` (seconds from first message) then one per joint (radians, or
metres for the prismatic lifts).

    ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv
    ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv --joints rj0 rj1 rj2 rj3 rj4 rj5 rj6

This is the "demonstration file": a plain table of joint angles over time that
the demonstrator module plays back and the ML side trains against.
"""

import argparse
import csv
import math

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import JointState


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag2 directory written by record_demo.sh")
    ap.add_argument("csv", help="output CSV path")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--joints", nargs="+", help="joint subset/order to export (default: all, in first-message order)")
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id=""), ConverterOptions("", ""))

    names = args.joints
    rows = []
    t0 = None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, JointState)
        if names is None:
            names = list(msg.name)
        if t0 is None:
            t0 = t_ns
        pos = dict(zip(msg.name, msg.position))
        rows.append([(t_ns - t0) / 1e9] + [pos.get(n, math.nan) for n in names])

    if not rows:
        raise SystemExit(f"no {args.topic} messages found in {args.bag}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + names)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows x {len(names)} joints ({rows[-1][0]:.1f} s) -> {args.csv}")


if __name__ == "__main__":
    main()
