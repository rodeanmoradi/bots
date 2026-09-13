# bracketbot_moveit_config

MoveIt 2 (Jazzy) config for BracketBot. Drag an interactive marker in RViz and
the 7-DOF arm follows — the "sim teleop" used to record therapy demonstrations
as joint-angle trajectories.

Two ways to run it:

| | `demo.launch.py` | `moveit_gazebo.launch.py` |
| --- | --- | --- |
| Simulator | none — ros2_control mock hardware | Gazebo Sim (`chopped_urdf_v2_gazebo`) |
| Physics | no | yes |
| Setup cost | zero | needs the Gazebo stack working first |
| Use for | recording demos, fast iteration | showing the "real" sim |

Start with `demo.launch.py`. Physics adds nothing to a therapy demonstration.

## Requirements

Ubuntu 24.04 with ROS 2 Jazzy. On top of `gazebo_sim/install_ros2_jazzy.sh`:

    sudo apt install ros-jazzy-moveit ros-jazzy-moveit-setup-assistant \
        ros-jazzy-moveit-configs-utils ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers ros-jazzy-rosbag2

No ROS on the machine (macOS, or a bare Linux box)? Use the container instead —
everything above is preinstalled and the desktop shows up in a browser:

    ./docker/run.sh                 # then open http://localhost:6080
    # in the desktop's terminal:
    cd ~/bots/ros2_ws
    colcon build --symlink-install --build-base ~/ws/build --install-base ~/ws/install
    source ~/ws/install/setup.bash

## Build

    cd ros2_ws
    colcon build --symlink-install
    source install/setup.bash

`ros2_ws/src/bracketbot_moveit_config` is a symlink to this directory.

## Run — standalone

    ros2 launch bracketbot_moveit_config demo.launch.py

RViz opens with the MotionPlanning display already on `right_arm` and the
orange goal-state marker attached to the right gripper.

1. Drag the marker (arrows translate, rings rotate). The orange ghost arm
   follows via IK.
2. In the MotionPlanning panel → **Planning** tab → **Plan & Execute**.
3. The real (white) arm moves there.

Switch to `left_arm` in the panel's **Planning Group** dropdown for the other
arm. `both_arms` plans both at once. Group `right_gripper` / `left_gripper`
with the named states `open` / `closed` works the gripper.

Marker-less alternative: the **Joints** tab in the same panel gives you a
slider per joint of the current group.

## Run — attached to Gazebo

    # terminal 1
    ros2 launch chopped_urdf_v2_gazebo sim.launch.py rviz:=false
    # terminal 2, once the arm controllers are up (~10 s)
    ros2 launch bracketbot_moveit_config moveit_gazebo.launch.py

Same RViz workflow; executions go to the sim's `right_arm_controller` /
`left_arm_controller`. Known: gz-sim ignores `<mimic>`, so a gripper plan
moves only the first finger there.

## Recording a demonstration

    # terminal A — while the teleop is running
    ros2 run bracketbot_moveit_config record_demo.sh reach_01
    # ... drag / Plan & Execute a few times, then Ctrl-C ...

    ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv
    ros2 run bracketbot_moveit_config bag_to_csv.py ~/demos/reach_01 reach_01.csv \
        --joints rj0 rj1 rj2 rj3 rj4 rj5 rj6      # right arm only

The CSV is `t, <joint>, <joint>, ...` at the joint_state_broadcaster's rate
(100 Hz). That table *is* the demonstration.

## Playing a demonstration

    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv --speed 0.5 --size 0.5

Pauses between the teleop's moves are cut out first (`--raw` keeps them); the
arm glides to the start pose over 2 s, then plays. For the therapy loop, run
it as a server and point `therapy/coach.py` at this machine:

    ros2 run bracketbot_moveit_config play_demo.py --serve 5555

It needs the `therapy` package importable (it is, via the repo checkout; or
`export PYTHONPATH=/path/to/bots:$PYTHONPATH`).

## What's in `config/`

| File | Role |
| --- | --- |
| `chopped_urdf_v2.urdf.xacro` | robot_description: the plain URDF + mock ros2_control block |
| `chopped_urdf_v2.ros2_control.xacro` | that mock block (18 joints; mimic fingers state-only) |
| `chopped_urdf_v2.srdf` | planning groups, end effectors, named poses |
| `kinematics.yaml` | KDL IK for each arm |
| `joint_limits.yaml` | velocity/accel limits for trajectory timing (URDF's 10 rad/s is not real) |
| `ros2_controllers.yaml` | mock-hardware controllers (standalone only) |
| `moveit_controllers.yaml` | MoveIt → mock controllers |
| `moveit_controllers_gazebo.yaml` | MoveIt → the Gazebo launch's controllers |
| `moveit.rviz` | RViz layout with the marker enabled |

## Collision checking is off — on purpose

The URDF MoveIt loads has visual meshes only, so there is no self-collision
matrix to maintain and planning never fails on "start state in collision".
The arm *can* be dragged through the mast; just don't. To turn checking on
later: point the include in `chopped_urdf_v2.urdf.xacro` at the `.gazebo`
xacro (it has collision meshes), then run
`ros2 launch bracketbot_moveit_config setup_assistant.launch.py` →
Self-Collisions → Generate, and save.

## Troubleshooting

- **No marker in RViz** — MotionPlanning display → Planning Request → tick
  *Query Goal State*; check *Planning Group* is set.
- **"Unable to identify any set of controllers"** — the spawners haven't
  finished; wait a few seconds and Plan & Execute again. `ros2 control list_controllers`
  should show all five `active`.
- **Plan succeeds, arm doesn't move (Gazebo)** — `allow_partial_joints_goal`
  must be `true` in `chopped_urdf_v2/config/controllers.yaml` (it is) and the
  sim must be started before this launch. Check `use_sim_time`: both nodes here
  set it; the RViz clock should be advancing.
- **Execution aborted PATH_TOLERANCE_VIOLATED (Gazebo)** — physics is lagging;
  lower `default_velocity_scaling_factor` in `joint_limits.yaml`.
- **IK fails near the mast / at full stretch** — that's the workspace edge; the
  ghost turns red. Move the marker back toward the body.
