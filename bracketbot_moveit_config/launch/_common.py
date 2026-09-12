"""Shared MoveIt config + node factories for the launch files in this package.

Launch files load this by path (see `_load_common()` in each) because ROS 2
launch files are plain scripts, not an importable package.
"""

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE = "bracketbot_moveit_config"


def build_moveit_config(controllers_file, publish_robot_description):
    """One place for every config path, so the launch files can't drift apart.

    controllers_file: which moveit_controllers*.yaml maps groups -> controllers.
    publish_robot_description: False when something else (the Gazebo launch's
        robot_state_publisher) already owns the /robot_description topic.
    """
    return (
        MoveItConfigsBuilder("chopped_urdf_v2", package_name=PACKAGE)
        .robot_description(file_path="config/chopped_urdf_v2.urdf.xacro")
        .robot_description_semantic(file_path="config/chopped_urdf_v2.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path=controllers_file)
        # OMPL only; its params come from moveit_configs_utils' defaults, which
        # track whatever MoveIt version is installed.
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .planning_scene_monitor(
            publish_robot_description=publish_robot_description,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .to_moveit_configs()
    )


def move_group_node(moveit_config, use_sim_time=False):
    return Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
    )


def rviz_node(moveit_config, use_sim_time=False):
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", str(moveit_config.package_path / "config/moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )
