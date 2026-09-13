"""Webcam -> the child's hand path, via MediaPipe Pose.

Per frame: shoulder, elbow, wrist of the arm in use, plus the other shoulder
and both hips to build a torso frame. The wrist relative to the shoulder in
that frame, divided by the child's arm length, is the same quantity
`robot_path.py` computes for the robot.

    python -m therapy.observer                 # live view: features drawn on the video
    python -m therapy.observer --arm right
    python -m therapy.observer --camera clip.mp4

Keys in the live view: c = calibrate arm length (hold the arm out straight,
any direction, for two seconds), r = record 5 s and print the Motion, q = quit.
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .motion import Motion, body_frame, elbow_angle_deg, to_hand_vector

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"

# MediaPipe Pose landmark indices
LM = {"l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
      "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24}
FALLBACK_ARM_LENGTH_M = 0.55   # adult-ish; replaced by calibration


@dataclass
class Sample:
    t: float
    hand: np.ndarray        # (3,) forward/outward/up in arm lengths
    elbow: float            # degrees
    visibility: float       # min visibility of the arm's three landmarks
    px: dict                # landmark name -> (x, y) pixel coords, for drawing
    arm_len_raw: float      # upper arm + forearm, metres (MediaPipe's scale)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """The model ships in the repo; this only runs if it has gone missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading pose model -> {path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:      # python.org builds on macOS often lack CA certs; curl has them
        import subprocess
        if subprocess.run(["curl", "-fsSL", "-o", str(path), MODEL_URL]).returncode != 0:
            raise RuntimeError(f"could not download the pose model ({e}); fetch {MODEL_URL} to {path} by hand")
    return path


class Observer:
    """Opens a camera (index or video path) and turns frames into Samples."""

    def __init__(self, camera=0, arm: str = "left", mirror: bool = True,
                 model_path: Path = MODEL_PATH, smoothing: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp, self.vision = mp, vision
        self.arm = arm
        self.mirror = mirror and not isinstance(camera, str)   # mirror live video only
        self.smoothing = smoothing
        self.arm_length = FALLBACK_ARM_LENGTH_M
        self.calibrated = False
        self._hand_ema: Optional[np.ndarray] = None
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path)),
                                               delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera!r}")
        self.is_file = isinstance(camera, str)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ---- per frame --------------------------------------------------------

    def read(self):
        """-> (frame_bgr, Sample or None). frame is None at end of a video file."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_file:
            t = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
        else:
            t = time.monotonic() - self._t0
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)   # MediaPipe needs strictly increasing
        self._last_ts_ms = ts_ms
        self.last_t = t

        # Detect on the raw frame: MediaPipe's left/right labels assume an
        # un-mirrored image. Mirror only what is shown, so the child sees
        # themselves as in a mirror (which is how they imitate the robot).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if not res.pose_world_landmarks:
            self._hand_ema = None
            return frame, None
        return frame, self._to_sample(t, res.pose_world_landmarks[0], res.pose_landmarks[0], frame.shape)

    def _to_sample(self, t, world, image, shape) -> Sample:
        p = {k: np.array([world[i].x, world[i].y, world[i].z]) for k, i in LM.items()}
        a = self.arm[0]          # 'l' or 'r'
        sh, el, wr = p[f"{a}_shoulder"], p[f"{a}_elbow"], p[f"{a}_wrist"]
        frame = body_frame(p["l_shoulder"], p["r_shoulder"], p["l_hip"], p["r_hip"])
        hand = to_hand_vector(wr, sh, frame, self.arm, self.arm_length)
        if self.smoothing > 0 and self._hand_ema is not None:
            hand = self.smoothing * self._hand_ema + (1 - self.smoothing) * hand
        self._hand_ema = hand
        h, w = shape[:2]
        px = {k: (int((1 - image[i].x) * w) if self.mirror else int(image[i].x * w), int(image[i].y * h))
              for k, i in LM.items()}
        vis = min(world[LM[f"{a}_shoulder"]].visibility, world[LM[f"{a}_elbow"]].visibility,
                  world[LM[f"{a}_wrist"]].visibility)
        return Sample(t, hand, elbow_angle_deg(sh, el, wr), vis, px,
                      float(np.linalg.norm(el - sh) + np.linalg.norm(wr - el)))

    # ---- higher level -----------------------------------------------------

    def calibrate(self, seconds: float = 2.0, on_frame=None) -> float:
        """Measure arm length (upper arm + forearm) as the median over `seconds`."""
        lens, t_start = [], None
        while True:
            frame, s = self.read()
            if frame is None:
                break
            t_start = self.last_t if t_start is None else t_start
            if self.last_t - t_start >= seconds:
                break
            if s is not None and s.visibility > 0.5:
                lens.append(s.arm_len_raw)
            if on_frame:
                on_frame(frame, s, "Hold your arm out straight...")
        if len(lens) >= 5:
            self.arm_length = float(np.median(lens))
            self.calibrated = True
        return self.arm_length

    def record(self, seconds: float = 5.0, on_frame=None, stop_when_still: bool = True) -> Motion:
        """Record a Motion. Stops early once the child has moved and gone still again."""
        ts, hands, elbows = [], [], []
        t_start = None
        moved_at = None
        self._hand_ema = None
        while True:
            frame, s = self.read()
            if frame is None:
                break
            t_start = self.last_t if t_start is None else t_start
            elapsed = self.last_t - t_start
            if s is not None and s.visibility > 0.3:
                ts.append(s.t); hands.append(s.hand); elbows.append(s.elbow)
            if on_frame:
                on_frame(frame, s, f"Your turn!  {max(0.0, seconds - elapsed):.1f}")
            if elapsed >= seconds:
                break
            if stop_when_still and len(ts) > 10:
                v = np.linalg.norm(np.gradient(np.array(hands[-10:]), np.array(ts[-10:]), axis=0), axis=1)
                if v.max() > 0.3 and moved_at is None:
                    moved_at = elapsed
                if moved_at is not None and elapsed - moved_at > 1.0 and v.max() < 0.1:
                    break
        if len(ts) < 2:
            return Motion(np.zeros(0), np.zeros((0, 3)), None, "webcam", {"moved": False})
        m = Motion(np.array(ts), np.array(hands), np.array(elbows), "webcam",
                   {"arm": self.arm, "arm_length_m": self.arm_length, "calibrated": self.calibrated})
        return m.trimmed_to_movement()

    def close(self):
        self.cap.release()
        self.landmarker.close()


# ---- drawing -----------------------------------------------------------------

def draw_sample(frame, s: Optional[Sample], arm: str, message: str = ""):
    """Skeleton of the arm in use + the numbers, drawn onto the BGR frame."""
    if s is not None:
        a = arm[0]
        pts = [s.px[f"{a}_shoulder"], s.px[f"{a}_elbow"], s.px[f"{a}_wrist"]]
        col = (0, 220, 0) if s.visibility > 0.5 else (0, 165, 255)
        cv2.line(frame, pts[0], pts[1], col, 3)
        cv2.line(frame, pts[1], pts[2], col, 3)
        for p in pts:
            cv2.circle(frame, p, 6, col, -1)
        cv2.line(frame, s.px["l_shoulder"], s.px["r_shoulder"], (200, 200, 200), 2)
        cv2.line(frame, s.px["l_hip"], s.px["r_hip"], (200, 200, 200), 2)
        f, o, u = s.hand
        txt = f"fwd {f:+.2f}  out {o:+.2f}  up {u:+.2f}  reach {np.linalg.norm(s.hand):.2f}  elbow {s.elbow:3.0f}"
    else:
        txt = "no person detected"
    cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, txt, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255) if s is not None else (0, 0, 255), 1, cv2.LINE_AA)
    if message:
        cv2.putText(frame, message, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index or a video file path")
    ap.add_argument("--arm", default="left", choices=["left", "right"], help="the child's arm to watch")
    args = ap.parse_args()
    cam = int(args.camera) if args.camera.isdigit() else args.camera
    obs = Observer(cam, arm=args.arm)
    msg = "c = calibrate   r = record 5 s   q = quit"

    def show(frame, s, m=msg):
        cv2.imshow("observer", draw_sample(frame, s, args.arm, m))
        cv2.waitKey(1)

    try:
        while True:
            frame, s = obs.read()
            if frame is None:
                break
            show(frame, s)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                print(f"arm length: {obs.calibrate(on_frame=show):.3f} m")
            if k == ord("r"):
                m = obs.record(on_frame=show)
                print(m.describe())
    finally:
        obs.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
