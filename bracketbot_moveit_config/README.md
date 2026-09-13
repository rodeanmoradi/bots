# bracketbot_moveit_config

MoveIt 2 config for BracketBot: the simulated robot, RViz teleop, MediaPipe
launch files, and demo record/play tools.

## Setup

    sudo apt install ros-jazzy-moveit ros-jazzy-moveit-setup-assistant \
        ros-jazzy-moveit-configs-utils ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers ros-jazzy-rosbag2
    cd ros2_ws && colcon build --symlink-install && source install/setup.bash

## Launch files

| Launch | What |
| --- | --- |
| `demo.launch.py` | simulated robot + RViz (used by the web interface) |
| `mediapipe_live.launch.py` | live mirroring from a camera |
| `mediapipe_replay.launch.py` | replay a recording |
| `moveit_gazebo.launch.py` | MoveIt attached to Gazebo |
| `setup_assistant.launch.py` | edit the config |

Useful arguments:

- `demo.launch.py`: `rviz_config`, `rviz_respawn`
- `mediapipe_live.launch.py`: `camera`, `side`, `plane`, `mirrored`, `smoothing`, `rviz`

Mirroring uses position-only IK and leaves the mast joint out.

## RViz teleop

    ros2 launch bracketbot_moveit_config demo.launch.py

1. Drag the orange marker.
2. **Planning** tab → **Plan & Execute**.

Change the arm in **Planning Group**. The **Joints** tab has sliders.

## With Gazebo

    ros2 launch chopped_urdf_v2_gazebo sim.launch.py rviz:=false
    ros2 launch bracketbot_moveit_config moveit_gazebo.launch.py   # after ~10 s

Gazebo ignores `<mimic>`: only one gripper finger moves.

## Record and play demos

    ros2 run bracketbot_moveit_config record_demo.sh reach_01                     # Ctrl+C to stop
    ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv --speed 0.5 --size 0.5
    ros2 run bracketbot_moveit_config play_demo.py --serve 5555                   # for the coach

Recording with the web interface is usually easier.

## Config

| File | Role |
| --- | --- |
| `chopped_urdf_v2.urdf.xacro` | robot description + mock hardware |
| `chopped_urdf_v2.srdf` | planning groups, named poses |
| `kinematics.yaml` | KDL IK |
| `joint_limits.yaml` | speed limits, one-way elbows |
| `ros2_controllers.yaml` | mock-hardware controllers |
| `moveit_controllers*.yaml` | MoveIt → controllers (mock / Gazebo) |
| `moveit.rviz`, `mediapipe_replay.rviz` | RViz layouts (teleop / mirroring) |

Collision checking is off (no collision meshes).

## Troubleshooting

- **No marker**: MotionPlanning → Planning Request → tick *Query Goal State*.
- **"Unable to identify any set of controllers"**: wait for the spawners, retry.
- **IK fails at full stretch**: workspace edge; move the marker closer.
- **Gazebo arm doesn't move**: start the sim first.
