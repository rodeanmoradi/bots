# chopped_urdf_v2

Self-contained copy of the BracketBot `chopped_urdf_v2` model: the URDF, every
mesh it references, and a standalone Rerun viewer.

    chopped_urdf_v2/
      urdf/chopped_urdf_v2.urdf       the model (54 links, 53 joints)
      urdf/*.gazebo.urdf.xacro        generated: + collisions/ros2_control (see gazebo_sim/)
      meshes/*.stl                    all 50 referenced meshes
      config/controllers.yaml         generated: ros2_control controller config
      package.xml, CMakeLists.txt     ROS 2 (ament_cmake) package, so package:// resolves
      launch/chopped_urdf_v2.launch   RViz + robot_state_publisher (ROS 1, reference only)
      manifest.json, draco/*.glb      Draco-compressed meshes for web viewers
    chopped_urdf_v2_gazebo/           ROS 2 package: gz sim world + launch + rviz config
    ros2_ws/src/                      symlinks to the two packages above, for colcon
    gazebo_sim/                       ROS 2 + Gazebo Sim setup — see gazebo_sim/README.md
    visualize_urdf.py, view_urdf.sh   Rerun viewer (no ROS needed)
    pyproject.toml                    deps for the viewer

The URDF refers to its meshes as `package://chopped_urdf_v2/meshes/<name>.stl`,
so keep the `chopped_urdf_v2/` directory intact — loaders resolve those paths
relative to the package root.

## Load it in Rerun (no ROS)

    ./view_urdf.sh --urdf chopped_urdf_v2/urdf/chopped_urdf_v2.urdf

First run provisions `rerun-sdk` and `numpy` into `./.venv` (via `uv` if
installed, otherwise the system `python3`). This opens the Rerun viewer plus a
browser slider panel for posing the joints. Other modes:

    ./view_urdf.sh --urdf ... --no-sliders          # viewer only, zero pose
    ./view_urdf.sh --urdf ... --list-joints         # print the movable joints
    ./view_urdf.sh --urdf ... --joint lj2=0.8       # one-shot pose (rad / m)
    ./view_urdf.sh --urdf ... --save bot.rrd        # headless recording

## Load it in ROS 2 / Gazebo Sim

`chopped_urdf_v2/` is now an ament_cmake (ROS 2) package — see
[`gazebo_sim/README.md`](gazebo_sim/README.md) for the one-time ROS 2 Jazzy +
ros_gz install, the generated collision/ros2_control URDF, and
`ros2 launch chopped_urdf_v2_gazebo sim.launch.py` to open it in `gz sim`.

`chopped_urdf_v2/launch/chopped_urdf_v2.launch` (ROS 1 / catkin, RViz only,
no Gazebo) is kept for reference but the package.xml no longer builds under
catkin.

## The model

Root link is `root`, at the wheel-axle midpoint with the tyres on z=0.

18 movable joints:

| Joint | Type | Range |
| --- | --- | --- |
| `rj0` / `lj0` | prismatic | -1.03044 .. 0 m (mast carriage) |
| `rj1`-`rj6` / `lj1`-`lj6` | revolute | ±2.094395 rad |
| `right_left_gripper`, `right_right_gripper` | revolute | 0 .. 1 rad |
| `left_left_gripper`, `left_right_gripper` | revolute | 0 .. 1 rad |

Each arm's second gripper joint `mimic`s the first, so a gripper is one DOF.
Every joint axis is exactly `(0,0,1)` — this was prepared for QP IK, where a raw
Onshape export snapped to the nearest principal axis puts FK ~27 cm out with no
warning. For IK the arm base link is `arm_base` and the end effectors are
`right_eef` / `left_eef`.

## Layout

| Dir | What |
| --- | --- |
| `chopped_urdf_v2/` | BracketBot URDF + meshes (ROS 2 package) |
| `chopped_urdf_v2_gazebo/`, `gazebo_sim/` | Gazebo Sim physics setup |
| `bracketbot_moveit_config/` | MoveIt 2: RViz interactive-marker teleop + demo recording (see its README) |
| `therapy/` | the AOT loop: sim robot demonstrates → webcam pose estimation scores the child → demo adapts (no ROS; see its README) |
| `webui/` | browser interface: camera + landmarks, record motions, coach, live mirror (`webui/run.sh [--camera URL]`, then http://localhost:8765) |
| `local/` | git-ignored data written on this machine: `recordings/` (recorded motions), `demos/` (robot demos made from recordings with IK), `sessions/` (coaching logs) |
| `ros2_ws/` | colcon workspace (symlinks to the packages above) |
| `docker/` | ROS 2 + MoveIt + Gazebo container for machines without ROS (`./docker/run.sh`) |
