#!/usr/bin/env bash
# One-time WSL setup for the live MediaPipe node (arm_xy_node.py). Needs your sudo password.
#
#   ./MediaPipe/setup_wsl.sh
set -euo pipefail
cd "$(dirname "$0")"

sudo apt install -y python3-pip python3-venv v4l-utils
# /dev/video* is group "video"; without this the camera opens as permission denied.
sudo usermod -aG video "$USER"

# System site packages so the venv still sees rclpy and the ROS message packages.
python3 -m venv --system-site-packages .venv
# 0.10.21 is the newest Linux/Python 3.12 build that still ships mp.solutions, which arm_xy.py uses.
.venv/bin/pip install "mediapipe==0.10.21"

cat <<'EOF'

Done. Remaining steps:
  1. From Windows PowerShell:  wsl --terminate Ubuntu-24.04   (so the video group applies)
  2. Admin PowerShell, once:   usbipd bind --busid <BUSID>     (webcam's BUSID from `usbipd list`)
  3. PowerShell, each session: usbipd attach --wsl --busid <BUSID>
  4. In WSL, check it shows:   v4l2-ctl --list-devices
EOF
