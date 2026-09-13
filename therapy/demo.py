"""A taught demonstration: joint angles over time, and the knobs the coach turns.

The file format is the CSV written by `bag_to_csv.py`:

    t, rj0, rj1, ..., rj6
    0.000, 0.0, 0.1, ...

`Demo.clean()` turns a raw teleop recording (long pauses between Plan &
Execute moves) into a continuous motion. `Demo.adapt()` applies the coach's
knobs — speed, size, segment, hold — and returns a new Demo ready to play.

Keyframe authoring is also here (`from_keyframes`) for motions like a throw
where the timing matters and pose-to-pose teleop can't produce it.
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

RIGHT_ARM = ["rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6"]
LEFT_ARM = ["lj0", "lj1", "lj2", "lj3", "lj4", "lj5", "lj6"]


@dataclass
class Demo:
    joints: list                  # J joint names
    t: np.ndarray                 # (N,) seconds from 0
    q: np.ndarray                 # (N, J) positions (rad, or m for rj0/lj0)
    name: str = "demo"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.q = np.asarray(self.q, dtype=float).reshape(len(self.t), -1)
        if len(self.t) and self.t[0] != 0:
            self.t = self.t - self.t[0]

    def __len__(self):
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def arm(self) -> str:
        return "left" if any(j.startswith("l") for j in self.joints) else "right"

    def copy(self, **changes) -> "Demo":
        d = Demo(list(self.joints), self.t.copy(), self.q.copy(), self.name, dict(self.meta))
        for k, v in changes.items():
            setattr(d, k, v)
        return d

    # ---- I/O --------------------------------------------------------------

    @classmethod
    def load_csv(cls, path: str, joints: Optional[Sequence[str]] = None) -> "Demo":
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        header, data = rows[0], np.array(rows[1:], dtype=float)
        if header[0] != "t":
            raise ValueError(f"{path}: first column must be 't', got {header[0]!r}")
        names = header[1:]
        if joints is None:
            # default: the right arm if present, else the left, else everything
            joints = [j for j in RIGHT_ARM if j in names] or [j for j in LEFT_ARM if j in names] or names
        cols = [names.index(j) + 1 for j in joints]
        return cls(list(joints), data[:, 0], data[:, cols], name=path.rsplit("/", 1)[-1].rsplit(".", 1)[0])

    def save_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + self.joints)
            for t, q in zip(self.t, self.q):
                w.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in q])

    def to_json(self) -> str:
        return json.dumps({"joints": self.joints, "t": self.t.round(4).tolist(),
                           "q": self.q.round(6).tolist(), "name": self.name})

    @classmethod
    def from_json(cls, s: str) -> "Demo":
        d = json.loads(s)
        return cls(d["joints"], d["t"], d["q"], d.get("name", "demo"))

    # ---- cleaning a raw teleop recording ---------------------------------

    def clean(self, still_speed: float = 0.02, max_pause: float = 0.3, dt: float = 0.02) -> "Demo":
        """Resample to `dt`, drop still time at both ends, shorten pauses.

        A joint moving slower than `still_speed` (rad/s) counts as still; any
        still stretch longer than `max_pause` seconds is cut down to it.
        """
        d = self.resample(dt)
        if len(d) < 3:
            return d
        speed = np.max(np.abs(np.gradient(d.q, d.t, axis=0)), axis=1)
        moving = speed > still_speed
        if not moving.any():
            return d
        first, last = np.argmax(moving), len(moving) - np.argmax(moving[::-1]) - 1
        keep = np.zeros(len(d), dtype=bool)
        keep[first:last + 1] = True
        # shorten interior pauses
        i = first
        while i <= last:
            if not moving[i]:
                j = i
                while j <= last and not moving[j]:
                    j += 1
                n_keep = int(round(max_pause / dt))
                keep[i + n_keep:j] = False
                i = j
            else:
                i += 1
        t = np.arange(keep.sum()) * dt
        return Demo(self.joints, t, d.q[keep], self.name, {**self.meta, "cleaned": True})

    def resample(self, dt: float) -> "Demo":
        if len(self.t) < 2:
            return self.copy()
        n = int(round(self.duration / dt)) + 1
        tn = np.linspace(0.0, self.duration, n)
        qn = np.column_stack([np.interp(tn, self.t, self.q[:, k]) for k in range(self.q.shape[1])])
        return Demo(self.joints, tn, qn, self.name, dict(self.meta))

    # ---- the coach's knobs -----------------------------------------------

    def adapt(self, speed: float = 1.0, size: float = 1.0,
              segment: Optional[Tuple[float, float]] = None,
              hold: Optional[Tuple[float, float]] = None) -> "Demo":
        """Return a modified copy.

        speed   1.0 = as taught, 0.5 = twice as slow.
        size    1.0 = as taught, 0.5 = every joint moves half as far from the
                start pose (a smaller version of the same motion).
        segment (a, b) fractions of the duration to keep, e.g. (0.4, 1.0) for
                the second half only. The arm still starts from the pose at a.
        hold    (fraction, seconds): pause at that point of the motion, e.g.
                (0.7, 1.0) freezes for one second at 70 % through.
        """
        d = self.copy()
        if segment is not None:
            a, b = (np.clip(segment, 0.0, 1.0) * d.duration).tolist()
            keep = (d.t >= a - 1e-9) & (d.t <= b + 1e-9)
            d = Demo(d.joints, d.t[keep], d.q[keep], d.name, d.meta)
        if size != 1.0 and len(d):
            q0 = d.q[0]
            d.q = q0 + size * (d.q - q0)
        if hold is not None and len(d) > 1:
            frac, secs = hold
            i = int(round(frac * (len(d) - 1)))
            dt = np.median(np.diff(d.t))
            n = max(1, int(round(secs / dt)))
            d.t = np.concatenate([d.t[:i + 1], d.t[i] + dt * np.arange(1, n + 1), d.t[i + 1:] + n * dt])
            d.q = np.vstack([d.q[:i + 1], np.repeat(d.q[i:i + 1], n, axis=0), d.q[i + 1:]])
        if speed != 1.0:
            d.t = d.t / speed
        d.meta = {**d.meta, "speed": speed, "size": size, "segment": segment, "hold": hold}
        return d

    # ---- keyframe authoring ----------------------------------------------

    @classmethod
    def from_keyframes(cls, joints: Sequence[str], poses: Sequence[Sequence[float]],
                       durations: Sequence[float], dt: float = 0.02, name: str = "keyframes") -> "Demo":
        """Minimum-jerk interpolation between poses.

        poses[i] -> poses[i+1] takes durations[i] seconds. Minimum jerk is
        what human reaching looks like: smooth start, one speed peak, smooth
        stop — and it's what a therapist wants the child to copy.
        """
        poses = np.asarray(poses, dtype=float)
        if len(poses) != len(durations) + 1:
            raise ValueError("need one duration per pair of consecutive poses")
        ts, qs = [0.0], [poses[0]]
        t_acc = 0.0
        for p0, p1, T in zip(poses[:-1], poses[1:], durations):
            n = max(2, int(round(T / dt)))
            s = np.linspace(0, 1, n + 1)[1:]
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            for b, si in zip(blend, s):
                ts.append(t_acc + si * T)
                qs.append(p0 + b * (p1 - p0))
            t_acc += T
        return cls(list(joints), np.array(ts), np.vstack(qs), name)
