"""MoveIt teleop attached to the running Gazebo sim.

Terminal 1 (the sim, without its own RViz):
    ros2 launch chopped_urdf_v2_gazebo sim.launch.py rviz:=false
Terminal 2 (this):
    ros2 launch bracketbot_moveit_config moveit_gazebo.launch.py

Only move_group and RViz start here. robot_state_publisher, /joint_states,
/clock and the arm controllers all come from the sim launch, so this must be
started second. Everything runs on sim time.
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription


def _load_common():
    spec = importlib.util.spec_from_file_location("_common", Path(__file__).with_name("_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description():
    common = _load_common()
    moveit_config = common.build_moveit_config(
        controllers_file="config/moveit_controllers_gazebo.yaml",
        publish_robot_description=False,  # the sim's robot_state_publisher owns it
    )
    return LaunchDescription(
        [
            common.move_group_node(moveit_config, use_sim_time=True),
            common.rviz_node(moveit_config, use_sim_time=True),
        ]
    )
