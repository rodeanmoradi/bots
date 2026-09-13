"""Drive the robot in RViz from the live webcam, through MoveIt IK.

One-time setup: MediaPipe/setup_wsl.sh. Each session, in a Windows PowerShell:
    usbipd attach --wsl --busid <BUSID of the webcam from `usbipd list`>
Then:
    ros2 launch bracketbot_moveit_config mediapipe_live.launch.py
    ros2 launch bracketbot_moveit_config mediapipe_live.launch.py camera:=/dev/video2 side:=left claw:=true

Starts robot_state_publisher, move_group (serving /compute_ik), RViz,
main_controller.py (source:=live) and MediaPipe/arm_xy_node.py, which reads the
camera and publishes /mediapipe/arm_landmarks.
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# --symlink-install links this file back to the source tree, two levels below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_common():
    spec = importlib.util.spec_from_file_location("_common", Path(__file__).with_name("_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description():
    common = _load_common()
    moveit_config = common.build_moveit_config(
        controllers_file="config/moveit_controllers.yaml",
        publish_robot_description=False,  # robot_state_publisher below owns the topic
    )
    common.enable_position_only_ik(moveit_config)

    arg = LaunchConfiguration
    arguments = [
        DeclareLaunchArgument("camera", default_value="0",
                              description="V4L2 index, /dev/video path, or http:// / rtsp:// stream URL"),
        DeclareLaunchArgument("side", default_value="right", description="tracked arm: left or right"),
        DeclareLaunchArgument("plane", default_value="side", description="side, front or top"),
        DeclareLaunchArgument("flip", default_value="false", description="mirror the camera image"),
        DeclareLaunchArgument("mirrored", default_value="false",
                              description="landmarks come from a mirrored image (see main_controller.py)"),
        DeclareLaunchArgument("claw", default_value="false", description="also publish /mediapipe/claw_open"),
        DeclareLaunchArgument("preview", default_value="true", description="show the camera window"),
        DeclareLaunchArgument("smoothing", default_value="0.5",
                              description="0 = raw targets, closer to 1 = smoother"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("mediapipe_python", default_value=str(REPO_ROOT / "MediaPipe/.venv/bin/python"),
                              description="interpreter with mediapipe installed (MediaPipe/setup_wsl.sh)"),
    ]

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    controller = ExecuteProcess(
        cmd=[
            "python3", str(REPO_ROOT / "main_controller.py"), "--ros-args",
            "-p", "source:=live",
            "-p", ["side:=", arg("side")],
            "-p", ["plane:=", arg("plane")],
            "-p", ["mirrored:=", arg("mirrored")],
            "-p", ["smoothing:=", arg("smoothing")],
        ],
        output="screen",
    )

    tracker = ExecuteProcess(
        cmd=[
            arg("mediapipe_python"), str(REPO_ROOT / "MediaPipe/arm_xy_node.py"), "--ros-args",
            "-p", ["camera:=", arg("camera")],
            "-p", ["side:=", arg("side")],
            "-p", ["plane:=", arg("plane")],
            "-p", ["flip:=", arg("flip")],
            "-p", ["claw:=", arg("claw")],
            "-p", ["preview:=", arg("preview")],
        ],
        output="screen",
    )

    return LaunchDescription(
        arguments
        + [
            robot_state_publisher,
            common.move_group_node(moveit_config),
            common.rviz_node(
                moveit_config,
                config="config/mediapipe_replay.rviz",
                condition=IfCondition(arg("rviz")),
            ),
            controller,
            tracker,
        ]
    )
