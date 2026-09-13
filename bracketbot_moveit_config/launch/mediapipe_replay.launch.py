"""Replay a MediaPipe arm recording on the robot in RViz, through MoveIt IK.

    ros2 launch bracketbot_moveit_config mediapipe_replay.launch.py
    ros2 launch bracketbot_moveit_config mediapipe_replay.launch.py \
        recording:=trajectories/right_side_20260912-170357.json rate_hz:=15

Starts robot_state_publisher, move_group (serving /compute_ik), RViz and the
repo-root main_controller.py. No ros2_control: the controller publishes
/joint_states itself. `recording` paths are relative to the repo root.
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# --symlink-install links this file back to the source tree, two levels below the repo root.
DEFAULT_CONTROLLER = Path(__file__).resolve().parents[2] / "main_controller.py"


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

    arguments = [
        DeclareLaunchArgument("recording", default_value="latest",
                              description="trajectories/*.json file, or 'latest'"),
        DeclareLaunchArgument("rate_hz", default_value="20.0", description="playback rate"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("smoothing", default_value="0.5",
                              description="0 = raw targets, closer to 1 = smoother"),
        DeclareLaunchArgument("mirrored", default_value="true",
                              description="recording came from a mirrored camera feed"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("controller_script", default_value=str(DEFAULT_CONTROLLER)),
    ]

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    controller = ExecuteProcess(
        cmd=[
            "python3", LaunchConfiguration("controller_script"), "--ros-args",
            "-p", ["recording:=", LaunchConfiguration("recording")],
            "-p", ["rate_hz:=", LaunchConfiguration("rate_hz")],
            "-p", ["loop:=", LaunchConfiguration("loop")],
            "-p", ["smoothing:=", LaunchConfiguration("smoothing")],
            "-p", ["mirrored:=", LaunchConfiguration("mirrored")],
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
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            controller,
        ]
    )
