"""Shared MoveIt config + node factories for the launch files in this package.

Launch files load this by path (see `_load_common()` in each) because ROS 2
launch files are plain scripts, not an importable package.
"""

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE = "bracketbot_moveit_config"
RVIZ_RESPAWNS = 10


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


def enable_position_only_ik(moveit_config):
    """IK solvers for the *_arm_no_mast groups the MediaPipe pipelines use, solving for the gripper
    position only (MediaPipe gives no usable hand orientation). The other groups, used by RViz teleop,
    keep their full-pose solvers."""
    kinematics = moveit_config.robot_description_kinematics["robot_description_kinematics"]
    for side in ("right", "left"):
        kinematics[f"{side}_arm_no_mast"] = {**kinematics[f"{side}_arm"], "position_only_ik": True}


def move_group_node(moveit_config, use_sim_time=False):
    return Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
    )


def rviz_node(moveit_config, use_sim_time=False, config="config/moveit.rviz", condition=None, respawn=False):
    """respawn: reopen RViz when its window is closed (a bool or a launch substitution), up to
    RVIZ_RESPAWNS times, so an RViz that can't start (no display) doesn't restart forever."""
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        condition=condition,
        respawn=respawn,
        respawn_delay=2.0,
        respawn_max_retries=RVIZ_RESPAWNS,
        # a launch substitution (e.g. from a launch argument) is passed through as-is
        arguments=["-d", str(moveit_config.package_path / config) if isinstance(config, str) else config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )
