# ROS 2 Jazzy desktop + MoveIt 2 + Gazebo Harmonic, with the desktop viewable
# in a browser via noVNC. For machines without a native ROS install (macOS, or
# Linux/WSL where you'd rather not install ROS system-wide).
#
# Base: Ubuntu 24.04 + ros-jazzy-desktop + an LXDE desktop served on port 80.
# The desktop session runs as user `ubuntu` (password `ubuntu`).
FROM tiryoh/ros2-desktop-vnc:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-moveit \
        ros-jazzy-moveit-setup-assistant \
        ros-jazzy-moveit-ros-visualization \
        ros-jazzy-moveit-configs-utils \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-joint-state-publisher-gui \
        ros-jazzy-rosbag2 \
        ros-jazzy-xacro \
        ros-jazzy-ros-gz \
        ros-jazzy-gz-ros2-control \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Every shell (including the desktop's terminal) gets ROS and, once built, the
# workspace overlay. System-wide so it applies to root and `ubuntu` alike.
RUN printf '%s\n' \
    'source /opt/ros/jazzy/setup.bash' \
    '[ -f "$HOME/ws/install/setup.bash" ] && source "$HOME/ws/install/setup.bash"' \
    'export GZ_SIM_RESOURCE_PATH="$HOME/ws/install/chopped_urdf_v2/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"' \
    >> /etc/bash.bashrc
