# gazebo_sim

Optional physics sim. The web interface doesn't need it.

## Install

    ./gazebo_sim/install_ros2_jazzy.sh    # ROS 2 Jazzy + ros_gz
    source ~/.bashrc

If `gz sim` picks the wrong version, add `export GZ_VERSION=harmonic` to `~/.bashrc`.

## Generate the Gazebo URDF

    python3 gazebo_sim/generate_gazebo_urdf.py

Adds collisions, a fixed base and ros2_control to the URDF, and writes
`chopped_urdf_v2/config/controllers.yaml`. Re-run after changing the URDF.

## Run

    cd ros2_ws && colcon build --symlink-install && source install/setup.bash
    ros2 launch chopped_urdf_v2_gazebo sim.launch.py      # rviz:=false to skip RViz

Move an arm:

    ros2 topic pub --once /right_arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
      '{joint_names: [rj0, rj1, rj2, rj3, rj4, rj5, rj6], points: [{positions: [0, 0.5, 0, 0, 0, 0, 0], time_from_start: {sec: 2}}]}'

Gazebo ignores `<mimic>`, so command both gripper fingers.
