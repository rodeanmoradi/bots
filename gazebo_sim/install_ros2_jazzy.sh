#!/usr/bin/env bash
# One-time setup: ROS 2 Jazzy + ros_gz on Ubuntu 24.04 (Noble).
#
# Run this yourself in a WSL terminal (it needs your sudo password, which
# Claude can't supply). Re-run is safe (apt/curl steps are idempotent-ish).
#
#   chmod +x gazebo_sim/install_ros2_jazzy.sh
#   ./gazebo_sim/install_ros2_jazzy.sh
#
# Afterwards, open a new shell (or `source ~/.bashrc`) so ROS 2 is on PATH.

set -euo pipefail

echo "== locale =="
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "== enable universe, add ROS 2 apt repo =="
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb"
sudo apt install -y /tmp/ros2-apt-source.deb

echo "== install ROS 2 Jazzy + ros_gz + tools =="
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-ros-gz \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-rviz2 \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-moveit \
  ros-jazzy-moveit-setup-assistant \
  python3-colcon-common-extensions \
  python3-rosdep

echo "== rosdep =="
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update

echo "== persist environment =="
if ! grep -q "source /opt/ros/jazzy/setup.bash" ~/.bashrc; then
  echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
fi

echo
echo "Done. Open a new shell (or 'source ~/.bashrc') and verify with:"
echo "  ros2 doctor --report | head -20"
echo "  ros2 pkg list | grep ros_gz"
