"""
Shared arm-tracking core.

Turns webcam frames into joint angles in a torso-fixed frame. Used by
arm_tracker.py (live UDP stream) and record.py (reference capture), so the
two can never drift apart in how they measure an angle.

Joint channels (all radians except gripper):
  shoulder_abduction  0 = arm hanging down, +ve = away from body
  shoulder_flexion    +ve = arm forward of the torso plane
  elbow_flexion       0 = straight, +ve = bent
  gripper             0 = closed pinch, 1 = fully open

Coordinate frame (built from the torso, so it moves with the person):
  x = lateral, pointing towards the subject's LEFT
  y = up, from hip midpoint towards shoulder midpoint
  z = forward, out of the chest (x cross y)
"""

import math

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# MediaPipe Pose landmark indices we care about
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24

# MediaPipe Hands landmark indices
THUMB_TIP, INDEX_TIP, MIDDLE_MCP, HAND_WRIST = 4, 8, 9, 0

# The three arm points we record, in order out from the body. Inverse
# kinematics downstream turns these back into robot joint commands.
POINTS = ("shoulder", "elbow", "hand")

# Legacy angle channels. Not recorded any more (the pipeline is Cartesian),
# kept because arm_angles is still handy for debugging a bad take.
JOINTS = ("shoulder_abduction", "shoulder_flexion", "elbow_flexion", "gripper")

# Camera placements. A single camera only measures one plane well, because
# monocular depth (z) is by far the noisiest axis. Each view names which two
# torso axes become the recorded x and y, so the motion you care about lands
# in the plane the camera can actually see.
#
#   x_axis/y_axis are indices into the torso frame: 0 = lateral (signed so
#   +ve is towards the tracked arm), 1 = up, 2 = forward out of the chest.
VIEWS = {
    "side":  {"x_axis": 2, "y_axis": 1,
              "help": "Camera at shoulder height, square to the subject's SIDE. "
                      "x = forward, y = up. Best for throwing."},
    "front": {"x_axis": 0, "y_axis": 1,
              "help": "Camera at shoulder height, facing the subject head on. "
                      "x = sideways, y = up. Best for reaching out to the side."},
    "top":   {"x_axis": 0, "y_axis": 2,
              "help": "Camera ABOVE the subject looking down. "
                      "x = sideways, y = forward. Best for reaching across a table."},
}

# Claw thresholds on the normalised finger aperture, with a dead band between
# them so the claw does not chatter open/closed on borderline frames.
CLAW_OPEN_ABOVE = 0.60
CLAW_SHUT_BELOW = 0.40


# ---------------------------------------------------------------- filtering

class OneEuro:
    """One Euro filter. Low lag at speed, heavy smoothing when nearly still.

    Tune: raise min_cutoff to reduce lag, raise beta to reduce overshoot on
    fast motion. Defaults are a reasonable starting point for limb tracking.
    """

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ------------------------------------------------------------------- geometry

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def torso_frame(lm):
    """Build an orthonormal torso frame from pose world landmarks.

    Returns a 3x3 matrix whose ROWS are the x, y, z basis vectors, so
    `R @ v` expresses a world vector v in torso coordinates.
    """
    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    lh, rh = lm[L_HIP], lm[R_HIP]

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)

    x = unit(ls - rs)                     # towards subject's left
    y_raw = shoulder_mid - hip_mid        # up along the spine
    # Gram-Schmidt: strip any x component out of y so the frame is orthogonal
    y = unit(y_raw - np.dot(y_raw, x) * x)
    z = np.cross(x, y)                    # forward, out of the chest
    return np.stack([x, y, z]), shoulder_mid, hip_mid


def arm_angles(lm, side):
    """Shoulder abduction, shoulder flexion and elbow flexion, in radians.

    `side` is "left" or "right" and refers to the SUBJECT's own side.
    """
    if side == "left":
        sh, el, wr = lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]
        lateral_sign = 1.0    # subject's left is +x in the torso frame
    else:
        sh, el, wr = lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]
        lateral_sign = -1.0

    R, _, _ = torso_frame(lm)

    upper = R @ (el - sh)     # shoulder to elbow, in torso coordinates
    fore = R @ (wr - el)      # elbow to wrist, in torso coordinates

    u = unit(upper)
    # Abduction: angle between the upper arm and straight down (-y),
    # measured in the frontal (x-y) plane.
    abduction = math.atan2(lateral_sign * u[0], -u[1])
    # Flexion: how far forward the upper arm is out of the frontal plane.
    flexion = math.asin(float(np.clip(u[2], -1.0, 1.0)))
    # Elbow: angle between upper arm and forearm. 0 when the arm is straight.
    cos_e = float(np.clip(np.dot(unit(upper), unit(fore)), -1.0, 1.0))
    elbow = math.acos(cos_e)

    return abduction, flexion, elbow


def arm_points(lm, side, view):
    """2D shoulder, elbow and hand positions for one arm.

    Origin is the hip midpoint. Units are TORSO LENGTHS: everything is
    divided by the hip-to-shoulder distance, so a tall and a short person
    performing the same motion produce the same numbers. Downstream, one
    scale factor converts torso lengths to robot metres.

    The plane is chosen by `view` (see VIEWS), because a single camera only
    measures one plane reliably. The lateral axis is signed so +ve always
    points towards the tracked arm, which keeps left and right takes
    directly comparable.

    Returns {"shoulder": (x, y), "elbow": (x, y), "hand": (x, y)}.
    """
    if side == "left":
        sh, el, wr = lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]
        idx, pky = lm[L_INDEX], lm[L_PINKY]
        lateral_sign = 1.0    # subject's left is +x in the torso frame
    else:
        sh, el, wr = lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]
        idx, pky = lm[R_INDEX], lm[R_PINKY]
        lateral_sign = -1.0

    R, shoulder_mid, hip_mid = torso_frame(lm)

    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_len < 1e-6:
        return None

    # Palm centre rather than the wrist landmark: averaging the wrist with
    # the index and pinky knuckles is steadier and sits where a claw would.
    hand = (wr + idx + pky) / 3.0

    ax, ay = VIEWS[view]["x_axis"], VIEWS[view]["y_axis"]
    out = {}
    for name, p in zip(POINTS, (sh, el, hand)):
        v = R @ (p - hip_mid) / torso_len      # torso coords, hip origin
        v = v * np.array([lateral_sign, 1.0, 1.0])   # +x towards tracked arm
        out[name] = (float(v[ax]), float(v[ay]))
    return out


def finger_aperture(hand_lm):
    """Thumb-to-index distance, normalised by hand size, clamped to 0..1.

    Normalising by the wrist-to-middle-MCP length makes this roughly
    invariant to how far the hand is from the camera.
    """
    pts = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark])
    span = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
    palm = np.linalg.norm(pts[MIDDLE_MCP] - pts[HAND_WRIST])
    if palm < 1e-6:
        return None
    # ~0.25 of palm length reads as a closed pinch, ~1.6 as wide open.
    raw = (span / palm - 0.25) / (1.6 - 0.25)
    return float(np.clip(raw, 0.0, 1.0))


# ------------------------------------------------------------------- tracker

class ArmTracker:
    """Frames in, Cartesian arm points out.

    `process` returns (sample, pose_res, hand_res). `sample` is None when
    the arm is not confidently visible; otherwise:

        {"shoulder": (x, y), "elbow": (x, y), "hand": (x, y),
         "claw_open": bool, "aperture": 0.0-1.0, "grip_valid": bool}

    Coordinates are in torso lengths from the hip midpoint, in the plane
    named by `view`. See arm_points.
    """

    def __init__(self, side="right", view="side", pose_complexity=1,
                 min_visibility=0.5):
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}, expected one of {sorted(VIEWS)}")
        self.side = side
        self.view = view
        self.min_visibility = min_visibility
        self.pose = mp_pose.Pose(model_complexity=pose_complexity,
                                 min_detection_confidence=0.6,
                                 min_tracking_confidence=0.6,
                                 smooth_landmarks=True)
        self.hands = mp_hands.Hands(max_num_hands=1,
                                    model_complexity=0,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)
        # One filter per scalar: x and y of each point, plus the aperture.
        self.filters = {f"{p}_{ax}": OneEuro(min_cutoff=1.2, beta=0.06)
                        for p in POINTS for ax in "xy"}
        self.filters["aperture"] = OneEuro(min_cutoff=1.2, beta=0.06)
        # Held so a brief hand dropout does not slam the claw shut. The
        # sample carries grip_valid so a consumer can tell a real hold from
        # a stale one.
        self.last_grip = 0.5
        self.claw_open = True

    def process(self, frame, t):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_res = self.pose.process(rgb)
        hand_res = self.hands.process(rgb)

        sample = None
        if pose_res.pose_world_landmarks:
            lm = np.array([[p.x, p.y, p.z]
                           for p in pose_res.pose_world_landmarks.landmark])
            # MediaPipe gives +x right, +y DOWN, +z AWAY from camera, which
            # is right-handed. torso_frame wants +y up and +z towards the
            # camera, so flip BOTH: flipping only y would leave a left-handed
            # frame and torso_frame's cross product would point backwards,
            # inverting the forward axis.
            lm[:, 1:3] *= -1.0

            keys = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
            keys += ([L_ELBOW, L_WRIST] if self.side == "left"
                     else [R_ELBOW, R_WRIST])
            vis = [pose_res.pose_landmarks.landmark[i].visibility for i in keys]

            if min(vis) > self.min_visibility:
                pts = arm_points(lm, self.side, self.view)
                if pts is not None:
                    grip_valid = False
                    if hand_res.multi_hand_landmarks:
                        g = finger_aperture(hand_res.multi_hand_landmarks[0])
                        if g is not None:
                            self.last_grip, grip_valid = g, True

                    f = self.filters
                    sample = {p: (f[f"{p}_x"](pts[p][0], t),
                                  f[f"{p}_y"](pts[p][1], t))
                              for p in POINTS}

                    ap = f["aperture"](self.last_grip, t)
                    # Hysteresis: only flip on a decisive reading, otherwise
                    # hold, so a borderline aperture cannot chatter the claw.
                    if ap > CLAW_OPEN_ABOVE:
                        self.claw_open = True
                    elif ap < CLAW_SHUT_BELOW:
                        self.claw_open = False
                    sample.update(claw_open=self.claw_open,
                                  aperture=ap,
                                  grip_valid=grip_valid)

        return sample, pose_res, hand_res

    def close(self):
        self.pose.close()
        self.hands.close()


# ------------------------------------------------------------------- cameras

def list_cameras():
    """DirectShow capture devices, in the same order cv2 indexes them.

    Returns a list of names, or None if pygrabber is not installed.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    return FilterGraph().get_input_devices()


def resolve_camera(spec):
    """Turn a --camera value into a cv2 index.

    A bare number is used as-is. Anything else is matched case-insensitively
    against the device names, so "iriun" keeps working after a reboot
    reshuffles the indices.
    """
    spec = str(spec)
    if spec.isdigit():
        return int(spec)

    names = list_cameras()
    if names is None:
        raise SystemExit("--camera by name needs pygrabber: pip install pygrabber")

    hits = [i for i, n in enumerate(names) if spec.lower() in n.lower()]
    if not hits:
        avail = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        raise SystemExit(f"No camera matching {spec!r}. Available:\n{avail}")
    if len(hits) > 1:
        amb = ", ".join(f"{i}:{names[i]}" for i in hits)
        raise SystemExit(f"{spec!r} is ambiguous, matches {amb}")
    print(f"camera {hits[0]}: {names[hits[0]]}")
    return hits[0]


def open_camera(spec, width=1280, height=720):
    """Open a capture device and prove it actually yields a frame."""
    index = resolve_camera(spec)
    # DSHOW rather than the MSMF default: MSMF is slow to open virtual
    # cameras on Windows and often refuses to set the frame size.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # drop stale frames, keeps latency down
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}")

    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        raise SystemExit(
            f"Camera {index} opened but produced no frame. If this is Iriun, "
            "start the app on the phone and make sure it says connected.")
    print(f"capturing {probe.shape[1]}x{probe.shape[0]}")
    return cap


# ------------------------------------------------------------------- overlay

def draw_overlay(frame, pose_res, hand_res, side):
    """Full-body skeleton, with the tracked arm highlighted on top."""
    if pose_res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

        # Redraw the tracked shoulder-elbow-wrist chain thick and bright so
        # it reads clearly against the rest of the skeleton.
        chain = ((L_SHOULDER, L_ELBOW, L_WRIST) if side == "left"
                 else (R_SHOULDER, R_ELBOW, R_WRIST))
        h, w = frame.shape[:2]
        pts = []
        for i in chain:
            p = pose_res.pose_landmarks.landmark[i]
            pts.append((int(p.x * w), int(p.y * h)))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 200, 255), 6, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 9, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 9, (20, 20, 20), 2, cv2.LINE_AA)

    if hand_res.multi_hand_landmarks:
        for hlm in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hlm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())


def draw_panel(frame, lines, colour=(0, 255, 0), width=560):
    """Text on a darkened plate, so it stays readable on a bright feed."""
    lh, pad = 26, 10
    box = min(lh * len(lines) + pad * 2, frame.shape[0])
    width = min(width, frame.shape[1])
    plate = frame[0:box, 0:width]
    cv2.addWeighted(plate, 0.35, np.zeros_like(plate), 0.65, 0, dst=plate)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (12, pad + 18 + lh * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def sample_lines(sample):
    """The recorded channels, formatted for the on-screen panel."""
    lines = [f"{p:9s} x{sample[p][0]:+6.2f}  y{sample[p][1]:+6.2f}"
             for p in POINTS]
    lines.append(
        f"claw      {'OPEN' if sample['claw_open'] else 'SHUT'}"
        f"  (ap {sample['aperture']:.2f})"
        + ("" if sample["grip_valid"] else "  hand lost, holding"))
    return lines
