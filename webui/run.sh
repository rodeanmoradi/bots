#!/usr/bin/env bash
# Start the web interface, then open http://localhost:8765 (works from a Windows browser too).
#
#   webui/run.sh
#   webui/run.sh --camera http://<phone-ip>:4747/video
#   webui/run.sh --camera fake          # no camera: coach runs with a simulated child
set -e
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
  source ros2_ws/install/setup.bash
fi
exec MediaPipe/.venv/bin/python webui/server.py "$@"
