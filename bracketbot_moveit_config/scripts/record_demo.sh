#!/usr/bin/env bash
# Record a teleop demonstration (joint angles over time) as a rosbag.
#
#   ros2 run bracketbot_moveit_config record_demo.sh reach_01
#     -> $DEMO_DIR/reach_01/   (default DEMO_DIR=~/demos)
#
# Ctrl-C to stop. Convert to CSV afterwards with:
#   ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv
set -euo pipefail

NAME="${1:?usage: record_demo.sh <demo-name>}"
OUT="${DEMO_DIR:-$HOME/demos}/$NAME"

if [ -e "$OUT" ]; then
    echo "refusing to overwrite existing recording: $OUT" >&2
    exit 1
fi

echo "recording /joint_states -> $OUT  (Ctrl-C to stop)"
exec ros2 bag record -o "$OUT" /joint_states
