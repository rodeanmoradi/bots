# therapy — the Action Observation Therapy loop

The robot has been taught a movement (a joint-angle CSV from the RViz teleop in
`bracketbot_moveit_config`). This package closes the loop around it:

    demonstrate (sim robot)  ->  child imitates (webcam)  ->  score  ->  adapt  ->  repeat

No ROS on this side. The robot is reached over a TCP socket, so the sim can run
on another machine.

## How robot and child are compared

They have different bodies, so they are never compared joint-to-joint. Both are
reduced to the same thing — **where the hand goes, relative to the shoulder, in
units of arm length** (`motion.py`):

| | robot | child |
| --- | --- | --- |
| source | taught CSV | webcam → MediaPipe Pose |
| converted by | `robot_path.py` (forward kinematics on the URDF) | `observer.py` |
| shoulder→hand vector in | robot body frame (x fwd, y left, z up) | torso frame from shoulders + hips |
| divided by | 0.725 m (arm hanging straight) | upper arm + forearm, measured in a 2 s calibration |
| axes | forward / **outward** / up | same |

"Outward" (away from the midline) instead of left/right lets a robot right-arm
demo be imitated mirror-wise with the child's left arm. The child's arm defaults
to the mirror of the robot's; override with `--arm`.

## Scoring (`scorer.py`)

Sub-scores, each 0..1, each meaning something a therapist would say:

| sub-score | question | how |
| --- | --- | --- |
| shape | did the hand travel the same way? | DTW distance between the two paths |
| reach | did they go as far? | peak distance from where the hand started, vs demo |
| height | did the hand get up high enough? | peak rise vs demo |
| direction | did it go the right way? | angle between the two "start → farthest point" vectors |
| elbow | did the elbow bend and straighten as much? | elbow angle range vs demo |
| smoothness | one clean movement or several? | number of speed peaks |
| tempo | same pace? | duration ratio |

Composite 0–100 is for the screen. Forward (depth) is weighted 0.4 because a
front-facing webcam measures it poorly; height and sideways are weighted 1.

## Adapting (`coach.py`)

A rule table over three knobs of the demonstration (`demo.py: Demo.adapt`):
`speed`, `size` (shrink every joint's motion toward the start pose), and
`segment` / `hold` (play only the extension part; pause at the top).

| weakest sub-score | next demonstration |
| --- | --- |
| reach | smaller and slower |
| height | 1 s hold at the top of the motion, "reach up high" |
| elbow | replay only the extension part, slower |
| smoothness / shape / direction | slower |
| tempo | match the child's pace |
| 3 good reps in a row | speed and size step back up toward the taught motion |
| 3 reps getting worse | short rest, restart smaller |

Every rep is logged to `local/sessions/<time>/session.jsonl` (git-ignored) with both hand paths.

## Run

    python3 -m venv .venv-therapy
    .venv-therapy/bin/pip install -r therapy/requirements.txt
    source .venv-therapy/bin/activate

    # 1. whole loop, no camera, no robot — proves the pipeline
    python -m therapy.coach therapy/demos/throw_keyframes.csv --child fake --robot stub-fast --no-window

    # 2. webcam, stub robot — tune the observer and the scores on yourself
    python -m therapy.observer            # live features; c = calibrate, r = record, q = quit
    python -m therapy.coach therapy/demos/throw_keyframes.csv

    # 3. the real thing: sim on the other machine
    #    there:  ros2 launch bracketbot_moveit_config demo.launch.py
    #            ros2 run bracketbot_moveit_config play_demo.py --serve 5555
    python -m therapy.coach therapy/demos/throw_keyframes.csv --robot <sim-machine-ip>:5555

`q` in the window quits. The demo CSV is cleaned first (pauses between the
teleop's Plan & Execute moves are cut to 0.3 s); `--raw` keeps it as recorded.

Other useful commands:

    python -m therapy.robot_path therapy/demos/throw_keyframes.csv --plot     # what the robot's hand path looks like
    ros2 run bracketbot_moveit_config play_demo.py therapy/demos/throw_keyframes.csv --speed 0.5 --size 0.5

## Authoring a throw instead of recording one

Pose-to-pose teleop can't capture timing. `Demo.from_keyframes` takes poses
(from the teleop, or `ros2 topic echo /joint_states`) and a duration for each
transition, and fills in minimum-jerk motion:

```python
from therapy.demo import Demo, RIGHT_ARM
ready = [0, 0, 0, 0, 0, 0, 0]
back  = [0, -0.6, 0, -1.2, 0, 0, 0]
throw = [0, 1.4, 0.3, -0.4, 0, 0, 0]
Demo.from_keyframes(RIGHT_ARM, [ready, back, throw, ready], [0.8, 0.6, 1.2]).save_csv("demos/throw.csv")
```

`therapy/demos/throw_keyframes.csv` is exactly this — a placeholder until a
real one is recorded with the teleop.

## Known limits

- Depth from one webcam is poor. If the movement is mostly forward, put the
  camera at the child's side (then it's in the image plane) — everything else
  stays the same.
- MediaPipe's left/right labels assume a person facing the camera. If the wrong
  arm gets highlighted, pass `--arm`.
- The robot's "elbow" is the `forearm` link origin; its angle is a reasonable
  analogue (89°–174° over rj3's range) but not identical to a human elbow.
