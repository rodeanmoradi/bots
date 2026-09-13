"""Score a child's attempt against the robot's demonstration.

Both are `Motion`s in the same units, so this is plain curve comparison.
The composite 0-100 is for the screen; the sub-scores (each 0..1) are what
the coach acts on, and each one maps to something a therapist says:

    shape       "did the hand travel the same way?"      DTW distance
    reach       "did they extend as far?"                peak reach vs demo
    height      "did they get the hand up high enough?"  peak up vs demo
    direction   "did they go the right way?"             angle between net displacements
    elbow       "did the elbow straighten?"              peak elbow angle vs demo
    smoothness  "one clean movement, or several?"        number of speed peaks
    tempo       "same speed?"                            duration ratio

Forward (depth) is weighted low by default because a single front-facing
webcam measures it poorly.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .motion import Motion

# axis weights for distance-based scores: forward, outward, up
AXIS_WEIGHTS = np.array([0.4, 1.0, 1.0])

COMPOSITE_WEIGHTS = {
    "shape": 0.25, "reach": 0.2, "height": 0.15, "direction": 0.1,
    "elbow": 0.1, "smoothness": 0.1, "tempo": 0.1,
}


def dtw_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray = AXIS_WEIGHTS,
                 band: Optional[int] = None) -> float:
    """Dynamic time warping distance between two (N,3) and (M,3) paths.

    Returns the average per-step weighted Euclidean distance along the best
    alignment, so the number is in arm lengths and doesn't grow with N.
    """
    a = np.asarray(a) * weights
    b = np.asarray(b) * weights
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    band = band or max(n, m)
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        for j in range(j0, j1 + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def excursion(m: Motion) -> np.ndarray:
    """How far the hand is from where it started, per frame (arm lengths)."""
    return np.linalg.norm(m.hand - m.hand[0], axis=1)


def count_speed_peaks(m: Motion, min_frac: float = 0.2) -> int:
    """Number of distinct speed peaks above min_frac of the max (1 = one clean move)."""
    v = m.smoothed(5).speed if len(m) >= 5 else m.speed
    if len(v) < 3 or v.max() <= 0:
        return 0
    thresh = min_frac * v.max()
    above = v > thresh
    # count rising edges
    return int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))


@dataclass
class Score:
    composite: float                  # 0..100
    sub: Dict[str, float]             # each 0..1
    moved: bool = True
    notes: dict = field(default_factory=dict)

    def weakest(self) -> str:
        """Lowest sub-score; near-ties (within 0.05) go to the one that is
        easiest to act on — size before elbow tricks before pacing."""
        lo = min(self.sub.values())
        for k in ("reach", "height", "elbow", "smoothness", "tempo", "direction", "shape"):
            if self.sub[k] <= lo + 0.05:
                return k
        return min(self.sub, key=self.sub.get)

    def good(self, thresh: float = 0.75) -> bool:
        return self.moved and self.composite >= 100 * thresh

    def summary(self) -> str:
        if not self.moved:
            return "no movement detected"
        parts = ", ".join(f"{k} {v:.2f}" for k, v in self.sub.items())
        return f"{self.composite:.0f}/100  ({parts})"


def _ratio(child: float, ref: float, floor: float = 0.05) -> float:
    """child/ref clipped to [0, 1]; both measured from the same baseline."""
    if ref < floor:
        return 1.0
    return float(np.clip(child / ref, 0.0, 1.0))


def score_attempt(child: Motion, ref: Motion, n: int = 100) -> Score:
    """Compare a child's Motion with the robot's demonstration Motion."""
    if len(child) < 5 or child.meta.get("moved") is False:
        return Score(0.0, {k: 0.0 for k in COMPOSITE_WEIGHTS}, moved=False)

    c = child.smoothed(5).resample(n)
    r = ref.resample(n)

    # shape: DTW on paths re-based to their own start, so a different standing
    # posture at the start doesn't count against the child.
    d = dtw_distance(c.hand - c.hand[0], r.hand - r.hand[0])
    shape = float(np.exp(-d / 0.12))          # 0.12 arm-length average error -> 0.37

    # amplitudes are measured from where the hand started (the "excursion"),
    # not from the shoulder: an arm hanging straight down is already at full
    # distance from the shoulder, so distance alone says nothing about reaching.
    exc_c, exc_r = excursion(c), excursion(r)
    reach = _ratio(exc_c.max(), exc_r.max())
    height = _ratio(c.hand[:, 2].max() - c.hand[0, 2], r.hand[:, 2].max() - r.hand[0, 2])

    # direction: the way the hand went to its farthest point
    dc = (c.hand[np.argmax(exc_c)] - c.hand[0]) * AXIS_WEIGHTS
    dr = (r.hand[np.argmax(exc_r)] - r.hand[0]) * AXIS_WEIGHTS
    cosang = np.dot(dc, dr) / (np.linalg.norm(dc) * np.linalg.norm(dr) + 1e-9)
    direction = float(np.clip((cosang + 1) / 2, 0, 1) ** 2)   # 1 = same way, 0.25 = 90° off

    # elbow: how much the elbow angle changed (bend + straighten), child vs demo
    if c.elbow is not None and r.elbow is not None:
        elbow = _ratio(np.ptp(c.elbow), np.ptp(r.elbow), floor=5.0)
    else:
        elbow = reach

    peaks_c, peaks_r = count_speed_peaks(c), max(1, count_speed_peaks(r))
    smoothness = float(np.clip(peaks_r / max(peaks_c, 1), 0, 1)) if peaks_c else 0.0

    dur_c, dur_r = max(c.duration, 1e-3), max(r.duration, 1e-3)
    tempo = float(min(dur_c, dur_r) / max(dur_c, dur_r))

    sub = {"shape": shape, "reach": reach, "height": height, "direction": direction,
           "elbow": elbow, "smoothness": smoothness, "tempo": tempo}
    composite = 100 * sum(COMPOSITE_WEIGHTS[k] * v for k, v in sub.items())
    return Score(round(composite, 1), {k: round(v, 3) for k, v in sub.items()}, True,
                 notes={"dtw": round(d, 4), "child_dur": round(dur_c, 2), "ref_dur": round(dur_r, 2),
                        "speed_peaks": peaks_c})
