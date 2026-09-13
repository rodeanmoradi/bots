"""The shared representation of an arm movement — robot or human.

A `Motion` is the hand's position relative to the shoulder over time, in a
body-fixed frame and in units of arm length:

    hand[:, 0]  forward   (away from the chest)
    hand[:, 1]  outward   (away from the body's midline, for the arm in use)
    hand[:, 2]  up

Using "outward" instead of left/right makes a robot right-arm demonstration
directly comparable with a child imitating it in the mirror with their left
arm. Dividing by arm length makes a 0.73 m robot arm comparable with a
0.45 m child's arm: a reach of 0.9 means "90 % of a straight arm" for both.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

AXES = ("forward", "outward", "up")


@dataclass
class Motion:
    t: np.ndarray                      # (N,) seconds, starts at 0
    hand: np.ndarray                   # (N, 3) forward/outward/up, arm lengths
    elbow: Optional[np.ndarray] = None  # (N,) elbow angle, degrees (180 = straight)
    source: str = ""                   # "robot", "webcam", "fake" — for display only
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.hand = np.asarray(self.hand, dtype=float).reshape(-1, 3)
        if self.elbow is not None:
            self.elbow = np.asarray(self.elbow, dtype=float)
        if len(self.t) and self.t[0] != 0:
            self.t = self.t - self.t[0]

    # ---- basic properties -------------------------------------------------

    def __len__(self):
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def reach(self) -> np.ndarray:
        """Distance of the hand from the shoulder, in arm lengths, per frame."""
        return np.linalg.norm(self.hand, axis=1)

    @property
    def speed(self) -> np.ndarray:
        """Hand speed in arm lengths / s, per frame (same length as t)."""
        if len(self.t) < 2:
            return np.zeros(len(self.t))
        d = np.gradient(self.hand, self.t, axis=0)
        return np.linalg.norm(d, axis=1)

    @property
    def displacement(self) -> np.ndarray:
        """Where the hand ended relative to where it started (3,)."""
        return self.hand[-1] - self.hand[0]

    def path_length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.hand, axis=0), axis=1)))

    # ---- transforms -------------------------------------------------------

    def resample(self, n: int) -> "Motion":
        """Uniformly resample to n frames (linear interpolation)."""
        if len(self.t) < 2:
            return self
        tn = np.linspace(self.t[0], self.t[-1], n)
        hand = np.column_stack([np.interp(tn, self.t, self.hand[:, k]) for k in range(3)])
        elbow = np.interp(tn, self.t, self.elbow) if self.elbow is not None else None
        return Motion(tn, hand, elbow, self.source, dict(self.meta))

    def smoothed(self, window: int = 5) -> "Motion":
        """Moving-average smoothing (odd window, in frames)."""
        if window < 3 or len(self.t) < window:
            return self
        k = np.ones(window) / window
        pad = window // 2

        def sm(x):
            xp = np.pad(x, pad, mode="edge")
            return np.convolve(xp, k, mode="valid")

        hand = np.column_stack([sm(self.hand[:, i]) for i in range(3)])
        elbow = sm(self.elbow) if self.elbow is not None else None
        return Motion(self.t, hand, elbow, self.source, dict(self.meta))

    def trimmed_to_movement(self, speed_thresh: float = 0.15, pad_s: float = 0.15) -> "Motion":
        """Cut the still time before the hand starts moving and after it stops.

        speed_thresh is in arm lengths / s. Returns an empty Motion (len 0) if
        the hand never moved faster than the threshold.
        """
        if len(self.t) < 3:
            return self
        moving = np.where(self.speed > speed_thresh)[0]
        if len(moving) == 0:
            return Motion(np.zeros(0), np.zeros((0, 3)), None, self.source, {**self.meta, "moved": False})
        t0 = self.t[moving[0]] - pad_s
        t1 = self.t[moving[-1]] + pad_s
        keep = (self.t >= t0) & (self.t <= t1)
        return Motion(self.t[keep], self.hand[keep],
                      self.elbow[keep] if self.elbow is not None else None,
                      self.source, {**self.meta, "moved": True})

    def slice(self, a: float, b: float) -> "Motion":
        """Keep frames with a <= t <= b (seconds)."""
        keep = (self.t >= a) & (self.t <= b)
        return Motion(self.t[keep], self.hand[keep],
                      self.elbow[keep] if self.elbow is not None else None,
                      self.source, dict(self.meta))

    # ---- persistence ------------------------------------------------------

    def save(self, path: str):
        np.savez(path, t=self.t, hand=self.hand,
                 elbow=self.elbow if self.elbow is not None else np.zeros(0),
                 source=self.source)

    @classmethod
    def load(cls, path: str) -> "Motion":
        z = np.load(path, allow_pickle=False)
        elbow = z["elbow"] if z["elbow"].size else None
        return cls(z["t"], z["hand"], elbow, str(z["source"]))

    def describe(self) -> str:
        if len(self) == 0:
            return f"{self.source or 'motion'}: (no movement)"
        r = self.reach
        return (f"{self.source or 'motion'}: {len(self)} frames, {self.duration:.2f} s, "
                f"reach {r.min():.2f}->{r.max():.2f}, peak up {self.hand[:, 2].max():+.2f}, "
                f"peak forward {self.hand[:, 0].max():+.2f}")


def body_frame(shoulder_l: np.ndarray, shoulder_r: np.ndarray,
               hip_l: np.ndarray, hip_r: np.ndarray) -> np.ndarray:
    """Rows = forward, left, up unit vectors of a torso, from four landmarks.

    Works in any right-handed world frame (MediaPipe's or the robot's): up is
    hips->shoulders, left is right shoulder->left shoulder, forward = left x up.
    """
    up = (shoulder_l + shoulder_r) / 2 - (hip_l + hip_r) / 2
    up = up / (np.linalg.norm(up) + 1e-9)
    left = shoulder_l - shoulder_r
    left = left - up * np.dot(left, up)
    left = left / (np.linalg.norm(left) + 1e-9)
    forward = np.cross(left, up)
    return np.vstack([forward, left, up])


def to_hand_vector(wrist: np.ndarray, shoulder: np.ndarray, frame: np.ndarray,
                   arm: str, arm_length: float) -> np.ndarray:
    """(wrist - shoulder) expressed as [forward, outward, up] / arm_length."""
    v = frame @ (wrist - shoulder)
    if arm == "right":
        v[1] = -v[1]        # left component -> outward for a right arm
    return v / arm_length


def elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    a = shoulder - elbow
    b = wrist - elbow
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
