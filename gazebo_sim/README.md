# Gazebo Sim setup for chopped_urdf_v2

Ties `chopped_urdf_v2`'s URDF into a ROS 2 + Gazebo Sim (`gz sim`) simulation.

## 1. One-time environment setup

No ROS is installed in this WSL environment yet, but Gazebo Sim (`gz-sim10`,
codename Jetty) already is. Install ROS 2 Jazzy + ros_gz yourself (needs your
sudo password, so this can't be run for you):

    chmod +x gazebo_sim/install_ros2_jazzy.sh
    ./gazebo_sim/install_ros2_jazzy.sh
    source ~/.bashrc

`ros-jazzy-ros-gz` targets Gazebo Harmonic (gz-sim8); it will pull that in
alongside the Jetty (gz-sim10) already on the system. That's expected — apt
versions the `gz-sim*` packages side by side. If `gz sim` ends up resolving to
the wrong major version when launched through ROS 2 (check with
`gz sim --versions` before vs. after a `ros2 launch`), pin it with
`export GZ_VERSION=harmonic` in `~/.bashrc`.

## 2. What's generated vs. hand-written

`chopped_urdf_v2/urdf/chopped_urdf_v2.urdf` (the Onshape export) has visual +
inertial geometry but no collisions, no ros2_control, and no world anchor —
enough for RViz/Rerun, not enough for physics. Regenerating instead of
hand-editing keeps it in sync with the source URDF:

    python3 gazebo_sim/generate_gazebo_urdf.py

This writes:
- `chopped_urdf_v2/urdf/chopped_urdf_v2.gazebo.urdf.xacro` — the source URDF
  plus a `<collision>` for every `<visual>` (reusing the same STL — the
  meshes are precise and this is the simplest correct option; if physics
  turns out too slow, simplifying collisions to primitives is the fallback),
  a `world` -> `root` fixed joint (this model's wheel joints are fixed, not
  drivable — see the top-level README — so the base is bolted down and only
  the arms/grippers are simulated), and a `<ros2_control>` block exposing the
  16 independently-actuated joints as position interfaces.
- `chopped_urdf_v2/config/controllers.yaml` — `joint_state_broadcaster` +
  one `joint_trajectory_controller` per arm.

**Known limitation**: the URDF's `<mimic>` tags (each gripper's second finger
joint mirrors the first) are ignored by gz-sim's default `dartsim` physics
engine — confirmed via a smoke-test run ("chosen physics engine does not
support mimic constraints"). Both finger joints per gripper are exposed as
independent command interfaces instead, so send matching setpoints to both if
you want them to move together.

## 3. Build

    cd ros2_ws
    colcon build --symlink-install
    source install/setup.bash

(`ros2_ws/src/chopped_urdf_v2*` are symlinks back to the repo-root packages —
nothing is duplicated.)

## 4. Run

    ros2 launch chopped_urdf_v2_gazebo sim.launch.py

Opens gz sim with the robot spawned (base fixed in place, 1 m up so nothing
starts interpenetrating the ground plane), starts
`joint_state_broadcaster` + `right_arm_controller` + `left_arm_controller`,
and opens rviz2 (`rviz:=false` to skip it). Command a joint trajectory, e.g.:

    ros2 topic pub /right_arm_controller/joint_trajectory \
      trajectory_msgs/msg/JointTrajectory \
      '{joint_names: [rj0, rj1, rj2, rj3, rj4, rj5, rj6, right_left_gripper, right_right_gripper],
        points: [{positions: [0,0.5,0,0,0,0,0,0,0], time_from_start: {sec: 2}}]}' --once
