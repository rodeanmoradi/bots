#!/usr/bin/env bash
# Build (first time) and start the ROS 2 + MoveIt + Gazebo container.
# The desktop appears at http://localhost:6080 — open it in a browser.
#
#   ./docker/run.sh          # build image if needed, then start container
#   ./docker/run.sh shell    # bash inside the running container
#   ./docker/run.sh stop     # stop and remove the container
#
# The repo is mounted at ~/bots inside the container; build into ~/ws so the
# container's build artifacts never mix with a native build on the host:
#
#   cd ~/bots/ros2_ws
#   colcon build --symlink-install --build-base ~/ws/build --install-base ~/ws/install
#   source ~/ws/install/setup.bash
set -euo pipefail
export PATH="$HOME/.docker/bin:$PATH"   # Docker Desktop on macOS

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=bracketbot-ros
NAME=bracketbot

case "${1:-start}" in
  shell)
    exec docker exec -it -u ubuntu -w /home/ubuntu/bots/ros2_ws "$NAME" bash
    ;;
  stop)
    docker rm -f "$NAME"
    ;;
  start)
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      docker build -t "$IMAGE" "$REPO/docker"
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" \
      -p 6080:80 \
      --shm-size=1g \
      -v "$REPO:/home/ubuntu/bots" \
      "$IMAGE"
    echo "started — open http://localhost:6080  (user/password: ubuntu/ubuntu)"
    ;;
  *)
    echo "usage: $0 [start|shell|stop]" >&2; exit 1
    ;;
esac
