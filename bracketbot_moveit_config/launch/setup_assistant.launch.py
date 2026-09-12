"""Re-open this config in the MoveIt Setup Assistant to edit groups, poses, or
generate a self-collision matrix.

    ros2 launch bracketbot_moveit_config setup_assistant.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("bracketbot_moveit_config")
    return LaunchDescription(
        [
            Node(
                package="moveit_setup_assistant",
                executable="moveit_setup_assistant",
                output="screen",
                arguments=["--config_pkg", pkg],
            )
        ]
    )
