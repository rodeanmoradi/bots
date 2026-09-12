"""Bring up chopped_urdf_v2 in Gazebo Sim (gz sim) with ros2_control.

    ros2 launch chopped_urdf_v2_gazebo sim.launch.py
    ros2 launch chopped_urdf_v2_gazebo sim.launch.py rviz:=false

Starts: gz sim (chopped_bot_world) -> spawns the robot from /robot_description
-> controller_manager (via the gz_ros2_control plugin baked into the URDF)
-> joint_state_broadcaster + right_arm_controller + left_arm_controller
-> robot_state_publisher, optionally rviz2.

The base is fixed to the world (see gazebo_sim/generate_gazebo_urdf.py in the
repo root) — this model's wheel joints are fixed, not drivable, so only the
arms/grippers move.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    description_pkg = get_package_share_directory("chopped_urdf_v2")
    gazebo_pkg = get_package_share_directory("chopped_urdf_v2_gazebo")

    xacro_file = os.path.join(description_pkg, "urdf", "chopped_urdf_v2.gazebo.urdf.xacro")
    controllers_yaml = os.path.join(description_pkg, "config", "controllers.yaml")
    world_file = os.path.join(gazebo_pkg, "worlds", "chopped_bot.sdf")
    rviz_config = os.path.join(gazebo_pkg, "rviz", "chopped_bot.rviz")

    rviz_arg = DeclareLaunchArgument("rviz", default_value="true")

    # sdformat's URDF->SDF conversion turns `package://chopped_urdf_v2/...`
    # mesh URIs into `model://chopped_urdf_v2/...`; gz-sim resolves `model://`
    # by searching GZ_SIM_RESOURCE_PATH for <path>/chopped_urdf_v2/..., so the
    # parent of the installed share dir needs to be on that path.
    share_parent = os.path.dirname(description_pkg)
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        share_parent + (":" + existing_resource_path if existing_resource_path else ""),
    )

    robot_description = {
        "robot_description": Command(
            [
                "xacro ", xacro_file,
                " controllers_config:=", controllers_yaml,
            ]
        )
    }

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
            )
        ),
        launch_arguments={"gz_args": f"-r {world_file}"}.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "chopped_bot",
            "-z", "1.0",
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller"],
        output="screen",
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller"],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # gz_ros2_control's controller_manager comes up only once the model is
    # spawned; give it a few seconds before the spawners try to reach it.
    delayed_controllers = TimerAction(
        period=5.0,
        actions=[
            joint_state_broadcaster_spawner,
            right_arm_controller_spawner,
            left_arm_controller_spawner,
        ],
    )

    return LaunchDescription(
        [
            rviz_arg,
            gz_resource_path,
            gz_sim,
            clock_bridge,
            robot_state_publisher,
            spawn_robot,
            delayed_controllers,
            rviz,
        ]
    )
