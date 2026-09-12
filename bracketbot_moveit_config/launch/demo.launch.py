"""Standalone MoveIt teleop — no simulator required.

    ros2 launch bracketbot_moveit_config demo.launch.py

Starts robot_state_publisher, move_group, RViz (MotionPlanning display with the
interactive marker on the right gripper), and a ros2_control controller_manager
on mock hardware with one trajectory controller per arm and per gripper.

Drag the marker, press "Plan & Execute" in the MotionPlanning panel, and the
arm follows. Record with `ros2 run bracketbot_moveit_config record_demo.sh <name>`.
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


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

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # Jazzy controller_manager takes the URDF from a topic, not a parameter.
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[str(moveit_config.package_path / "config/ros2_controllers.yaml")],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
    )

    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager"],
            output="screen",
        )
        for name in (
            "joint_state_broadcaster",
            "right_arm_controller",
            "left_arm_controller",
            "right_gripper_controller",
            "left_gripper_controller",
        )
    ]

    return LaunchDescription(
        [
            robot_state_publisher,
            ros2_control_node,
            # give controller_manager a moment to receive the URDF before spawning
            TimerAction(period=2.0, actions=spawners),
            common.move_group_node(moveit_config),
            common.rviz_node(moveit_config),
        ]
    )
