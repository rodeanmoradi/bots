"""
Arm position from a webcam frame, as plain xy numbers.

One function, call it per frame:

    from arm_xy import arm_xy

    pts = arm_xy(frame)          # (4, 2) array, or None if no one is tracked
    # pts[0] = shoulder (x, y)
    # pts[1] = elbow    (x, y)
    # pts[2] = wrist    (x, y)
    # pts[3] = hand     (x, y)   palm centre

Origin is the hip centre. Units are roughly metres. Nothing is scaled or
normalised - scale it however the robot needs on your side.

To build a trajectory, keep the frames you get back and stack them:

    traj = []
    while capturing:
        pts = arm_xy(frame)
        if pts is not None:
            traj.append(pts)
    traj = np.stack(traj)        # (n_frames, 4, 2)

A camera only measures one plane properly, so `plane` picks which two body
axes become x and y:

    "side"   x = forward,  y = up        camera at the subject's side
    "front"  x = sideways, y = up        camera facing the subject
    "top"    x = sideways, y = forward   camera above, looking down

There is also a claw reading if you want it:

    pts, claw = arm_xy(frame, with_claw=True)   # claw is True/False/None
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# The HTTP stream reader lives in the therapy package so both pipelines share one copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from therapy.mjpeg import MjpegStream  # noqa: E402

# Pose landmarks
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24

# Hand landmarks, for the claw
THUMB_TIP, INDEX_TIP, MIDDLE_MCP, HAND_WRIST = 4, 8, 9, 0

# Which torso axes become x and y. 0 = sideways, 1 = up, 2 = forward.
PLANES = {"side": (2, 1), "front": (0, 1), "top": (0, 2)}

CLAW_OPEN_ABOVE = 0.60   # aperture over this reads as open
CLAW_SHUT_BELOW = 0.40   # under this reads as shut, between the two we hold

_pose = None
_hands = None
_claw_open = True

# Image-space landmarks from the most recent call, for anything that wants to
# draw the skeleton. Not needed to use arm_xy itself.
last_pose = None
last_hands = None


def _models():
    """Build the MediaPipe models once and reuse them.

    They are stateful trackers and slow to construct, so they must not be
    rebuilt per frame.
    """
    global _pose, _hands
    if _pose is None:
        _pose = mp.solutions.pose.Pose(model_complexity=1,
                                       min_detection_confidence=0.6,
                                       min_tracking_confidence=0.6,
                                       smooth_landmarks=True)
        _hands = mp.solutions.hands.Hands(max_num_hands=1,
                                          model_complexity=0,
                                          min_detection_confidence=0.6,
                                          min_tracking_confidence=0.6)
    return _pose, _hands


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def arm_xy(frame, side="right", plane="side", with_claw=False,
           min_visibility=0.5):
    """Shoulder, elbow, wrist and hand as a (4, 2) array. None if not tracked.

    frame           BGR image, straight from cv2
    side            which arm, "left" or "right"
    plane           "side", "front" or "top" - see PLANES
    with_claw       also return True/False for the claw being open
    min_visibility  how confident MediaPipe must be before we return anything
    """
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {sorted(PLANES)}")

    global _claw_open, last_pose, last_hands
    pose, hands = _models()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_res = pose.process(rgb)
    hand_res = hands.process(rgb) if with_claw else None
    last_pose, last_hands = pose_res, hand_res

    if not pose_res.pose_world_landmarks:
        return (None, None) if with_claw else None

    lm = np.array([[p.x, p.y, p.z]
                   for p in pose_res.pose_world_landmarks.landmark])
    # MediaPipe gives +x right, +y DOWN, +z AWAY from camera, which is
    # right-handed. We want +y up and +z towards the camera, so flip BOTH:
    # flipping only y would leave a left-handed frame and the cross product
    # below would then point backwards, inverting the forward axis.
    lm[:, 1:3] *= -1.0

    if side == "left":
        sh, el, wr = lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]
        idx, pky = lm[L_INDEX], lm[L_PINKY]
        arm = (L_ELBOW, L_WRIST)
    else:
        sh, el, wr = lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]
        idx, pky = lm[R_INDEX], lm[R_PINKY]
        arm = (R_ELBOW, R_WRIST)

    # Bail out rather than return a guess when the body is half out of shot.
    checks = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) + arm
    if min(pose_res.pose_landmarks.landmark[i].visibility
           for i in checks) < min_visibility:
        return (None, None) if with_claw else None

    # Body axes, so the numbers do not swing around when the person turns.
    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    hip_mid = 0.5 * (lm[L_HIP] + lm[R_HIP])
    x_ax = _unit(ls - rs)                                  # sideways
    y_ax = _unit((0.5 * (ls + rs) - hip_mid)
                 - np.dot(0.5 * (ls + rs) - hip_mid, x_ax) * x_ax)   # up
    z_ax = np.cross(x_ax, y_ax)                            # forward
    R = np.stack([x_ax, y_ax, z_ax])

    # Palm centre, steadier than the bare wrist landmark.
    hand = (wr + idx + pky) / 3.0

    ax, ay = PLANES[plane]
    pts = np.array([[(R @ (p - hip_mid))[ax], (R @ (p - hip_mid))[ay]]
                    for p in (sh, el, wr, hand)])

    if not with_claw:
        return pts

    if hand_res and hand_res.multi_hand_landmarks:
        h = np.array([[p.x, p.y, p.z]
                      for p in hand_res.multi_hand_landmarks[0].landmark])
        palm = np.linalg.norm(h[MIDDLE_MCP] - h[HAND_WRIST])
        if palm > 1e-6:
            span = np.linalg.norm(h[THUMB_TIP] - h[INDEX_TIP])
            ap = (span / palm - 0.25) / (1.6 - 0.25)
            # Only flip on a decisive reading, so the claw cannot chatter.
            if ap > CLAW_OPEN_ABOVE:
                _claw_open = True
            elif ap < CLAW_SHUT_BELOW:
                _claw_open = False
    return pts, _claw_open


def reset():
    """Drop the models, e.g. to switch camera or free memory."""
    global _pose, _hands
    if _pose is not None:
        _pose.close()
        _hands.close()
    _pose = _hands = None


def open_camera(spec="iriun", width=1280, height=720, fps=None):
    """Open a webcam by index, /dev/video path or stream URL; on Windows also by part of its name such as "iriun"."""
    spec = str(spec)
    if spec.startswith(("http://", "https://")):
        index = spec
        cap = MjpegStream(spec)
        no_frame_hint = "is the phone app streaming, with no other viewer connected?"
    elif spec.startswith("rtsp://"):
        index = spec
        cap = cv2.VideoCapture(spec, cv2.CAP_FFMPEG)
        no_frame_hint = "is the RTSP stream reachable from WSL?"
    elif sys.platform.startswith("linux"):
        index = int(spec) if spec.isdigit() else spec
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # USB 2 webcams only reach 720p at a usable frame rate in MJPG.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        no_frame_hint = "is it attached to WSL with `usbipd attach --wsl`?"
    else:
        if spec.isdigit():
            index = int(spec)
        else:
            try:
                from pygrabber.dshow_graph import FilterGraph
                names = FilterGraph().get_input_devices()
            except ImportError:
                raise SystemExit("opening a camera by name needs: pip install pygrabber")
            hits = [i for i, n in enumerate(names) if spec.lower() in n.lower()]
            if not hits:
                avail = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
                raise SystemExit(f"no camera matching {spec!r}. available:\n{avail}")
            index = hits[0]
        # DSHOW, because the default Windows backend is slow to open virtual
        # cameras and often ignores the resolution request.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        no_frame_hint = "is Iriun connected?"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        error = getattr(cap, "last_error", None)
        raise SystemExit(f"camera {index} gave no frame - {no_frame_hint}"
                         + (f" (last error: {error})" if error else ""))
    return cap


if __name__ == "__main__":
    # Demo: print the four points, and show them, until Esc.
    cap = open_camera(sys.argv[1] if len(sys.argv) > 1 else "iriun")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            pts = arm_xy(frame)
            if pts is not None:
                print(" ".join(f"{n}({x:+.2f},{y:+.2f})"
                               for n, (x, y) in zip(("sh", "el", "wr", "hand"), pts)),
                      end="\r")
            cv2.imshow("arm_xy", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        reset()
  