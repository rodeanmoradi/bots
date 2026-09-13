# therapy

The coaching loop: **demonstrate → person copies → score → adapt → repeat**.

No ROS: the robot is reached over TCP (`play_demo.py --serve`). Used by the
web interface's Coach tab, or run on its own. Overview diagram in the
[top-level README](../README.md#adaptive-coaching).

## Modules

| Module | Role |
| --- | --- |
| `coach.py` | loop and adaptation rules |
| `demo.py` | joint-angle demo: CSV, cleaning, speed/size/segment/hold |
| `motion.py` | shared hand-path format |
| `observer.py` | webcam → person's hand path (MediaPipe) |
| `robot_path.py` | robot demo → robot's hand path (forward kinematics) |
| `scorer.py` | compares the two paths |
| `robot_link.py` | stub robot and TCP client |
| `retarget.py` | human landmarks → robot targets for IK |
| `mjpeg.py` | phone camera stream reader |

## Comparing robot and person

Both become **the hand's path relative to the shoulder, in arm lengths**, on
axes forward / outward / up.

- Robot: forward kinematics, divided by its 0.725 m arm.
- Person: MediaPipe, divided by their arm length (2 s calibration).
- "Outward" lets a robot's right arm be copied mirror-wise with the left.

## Scoring

Each score is 0–1. Composite = weighted sum × 100; **75+ is a good rep**.

| Score | Checks | Weight |
| --- | --- | --- |
| shape | path similarity (DTW) | 0.25 |
| reach | max distance from start vs robot | 0.20 |
| height | max rise vs robot | 0.15 |
| direction | angle to the farthest point | 0.10 |
| elbow | elbow angle range vs robot | 0.10 |
| smoothness | number of speed peaks | 0.10 |
| tempo | duration ratio | 0.10 |

Depth counts 0.4× (a webcam measures it poorly).

## Adapting

Four knobs on the demo: **speed**, **size**, **segment** (part of the motion),
**hold** (pause). First matching rule wins:

| Situation | Next demo |
| --- | --- |
| no movement | same |
| 3 worsening reps | rest, size −0.2 |
| 3 good reps | remove segment/hold, else size and speed +0.2 |
| good rep | same |
| weakest = reach | size −0.2, speed −0.2 |
| weakest = height | hold 1 s at the top |
| weakest = elbow | only the straightening part, slower |
| weakest = tempo | speed toward the person's pace |
| weakest = other | speed −0.2 |

Size and speed stay ≥ 0.4. Reps are logged to `local/sessions/<time>/`.

## Run on its own

    python -m therapy.coach therapy/demos/throw_keyframes.csv --child fake --robot stub-fast --no-window   # no camera, no robot
    python -m therapy.coach therapy/demos/throw_keyframes.csv --camera <url>                              # webcam, stub robot
    python -m therapy.coach therapy/demos/throw_keyframes.csv --robot 127.0.0.1:5555                      # sim robot (play_demo.py --serve 5555)
    python -m therapy.observer --camera <url>                                                              # test tracking: c calibrate, r record, q quit

## Keyframed demos

```python
from therapy.demo import Demo, RIGHT_ARM
ready, back, throw = [0]*7, [0, -0.6, 0, -1.2, 0, 0, 0], [0, 1.4, 0.3, -0.4, 0, 0, 0]
Demo.from_keyframes(RIGHT_ARM, [ready, back, throw, ready], [0.8, 0.6, 1.2]).save_csv("throw.csv")
```

## Known limits

- **Tempo rule is inverted**: a faster person gets a slower demo.
- **Size scales joint angles**, not reach (size 0.6 ≠ 60 % reach).
- **Depth is poor** from one webcam: film from the side for forward movements.
- Wrong arm tracked? Pass `--arm`.
