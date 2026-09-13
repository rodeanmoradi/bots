# AOT BracketBot

Action Observation Therapy with a simulated BracketBot. The robot demonstrates
an arm movement in RViz, a webcam scores how well you copy it, and the next
demonstration adapts.

Ubuntu 24.04 (WSL works), ROS 2 Jazzy, MoveIt 2, MediaPipe.

## Quick start

Setup:

    ./gazebo_sim/install_ros2_jazzy.sh
    sudo apt install ros-jazzy-moveit ros-jazzy-moveit-configs-utils \
        ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-rosbag2
    ./MediaPipe/setup_wsl.sh
    cd ros2_ws && colcon build --symlink-install && cd ..

Run webpage:

    webui/run.sh --camera http://<phone-ip>:4747/video   # or --camera 0, or --camera fake
    # open http://localhost:8765

- **Record**: save a movement while the robot mirrors you.
- **Coach**: the robot demonstrates, you copy, it adapts.
- **Mirror**: the robot copies you live.

## How it works

```mermaid
flowchart TB
    subgraph perception["1. Perception"]
        cam["Camera<br/>OpenCV"] --> mp["MediaPipe Pose<br/>arm_xy.py"] --> web["webui/server.py"]
    end

    subgraph ik["2. Retarget + IK"]
        mc["main_controller.py"]
        mg["move_group<br/>MoveIt"]
    end

    subgraph robot["3. Robot: ros2_control"]
        jtc["arm controller"]
        jsb["joint_state_broadcaster"]
    end

    subgraph viz["4. Visualization"]
        rsp["robot_state_publisher"]
        rviz["RViz"]
    end

    subgraph coaching["Coaching"]
        coach["therapy/coach.py"] -->|"TCP"| pd["play_demo.py"]
    end

    web -->|"/mediapipe/arm_landmarks"| mc
    mc -.->|"/compute_ik"| mg
    mc -->|"/right_arm_controller/joint_trajectory"| jtc
    pd ==>|"follow_joint_trajectory"| jtc
    jtc --> jsb
    jsb -->|"/joint_states"| rsp
    rsp -->|"/tf"| rviz
    mc -->|"/mediapipe/arm_targets"| rviz
```

Solid arrow: topic, from publisher to subscriber. Dotted: service call. Thick:
action. The left arm uses `left_arm_controller`.

1. **Perception**: OpenCV reads the camera; MediaPipe finds shoulder, elbow,
   wrist and hand. The server publishes them on `/mediapipe/arm_landmarks`.
2. **Retarget + IK**: `main_controller.py` maps the landmarks onto the robot's
   arm (`therapy/retarget.py`) and asks MoveIt's `/compute_ik` for joint angles.
3. **Robot**: the angles go to the arm controller, which publishes the arm's
   position on `/joint_states`.
4. **Visualization**: `robot_state_publisher` turns joint states into `/tf`
   frames; RViz draws the robot and the retarget markers.

**Coaching** skips perception and IK at play time: the recording is converted
with `/compute_ik` beforehand, and `play_demo.py` sends the whole demo as an
action.

Not shown: `/robot_description`, the robot's URDF published once as text by
`robot_state_publisher`, so every node loads the same model. It's "latched":
nodes that start later still receive it.

## Adaptive coaching

```mermaid
flowchart LR
    demo["Robot demonstrates"] --> copy["Person copies"]
    copy --> score["Score the attempt"]
    score --> good{"Good rep?"}
    good -->|yes| harder["Step back toward<br/>the full movement"]
    good -->|no| easier["Ease the weakest part"]
    harder --> demo
    easier --> demo
```

Robot and person are compared as hand paths relative to the shoulder, in arm
lengths. Seven scores (0–1); a weighted total of 75+ is a good rep.

- **Shape**: same path? Else slower.
- **Reach**: as far? Else smaller and slower.
- **Height**: as high? Else pause at the top.
- **Direction**: right way? Else slower.
- **Elbow**: straightened as much? Else show only that part, slower.
- **Smoothness**: one clean movement? Else slower.
- **Tempo**: same pace? Else adjust speed.

3 good reps in a row: back toward the full movement. 3 worsening reps: short
rest. Details in [therapy/README.md](therapy/README.md).

## Layout

| Path | What |
| --- | --- |
| `webui/` | web interface ([README](webui/README.md)) |
| `therapy/` | coaching loop ([README](therapy/README.md)) |
| `MediaPipe/` | webcam → arm landmarks |
| `main_controller.py` | landmarks → robot joints via IK |
| `bracketbot_moveit_config/` | MoveIt config, launch files ([README](bracketbot_moveit_config/README.md)) |
| `chopped_urdf_v2/` | robot URDF and meshes |
| `chopped_urdf_v2_gazebo/`, `gazebo_sim/` | optional Gazebo sim ([README](gazebo_sim/README.md)) |
| `trajectories/` | example recordings |
| `ros2_ws/` | colcon workspace |
| `visualize_urdf.py`, `view_urdf.sh` | URDF viewer (Rerun, no ROS) |
| `local/` | git-ignored: recordings, demos, sessions |

## Other commands

| Command | What |
| --- | --- |
| `ros2 launch bracketbot_moveit_config demo.launch.py` | RViz teleop |
| `ros2 launch bracketbot_moveit_config mediapipe_live.launch.py camera:=<url>` | mirror without the webpage |
| `ros2 launch bracketbot_moveit_config mediapipe_replay.launch.py recording:=latest` | replay a recording |
| `python -m therapy.coach therapy/demos/throw_keyframes.csv --child fake --robot stub-fast --no-window` | coaching, no camera or robot |
| `ros2 launch chopped_urdf_v2_gazebo sim.launch.py` | Gazebo |
| `./view_urdf.sh --urdf chopped_urdf_v2/urdf/chopped_urdf_v2.urdf` | URDF in Rerun |

## Robot model

- 7 joints per arm: `rj0`/`lj0` slides along the mast (−1.03..0 m); `rj1`–`rj6` revolute (±2.09 rad).
- Elbows (`rj3`/`lj3`) bend one way only.
- Grippers: one joint each (0..1 rad).
- IK: base `arm_base`, end effectors `right_eef`/`left_eef`. Mirroring excludes the mast joint.
