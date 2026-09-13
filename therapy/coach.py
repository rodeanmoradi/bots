"""The therapy loop: demonstrate -> watch the child -> score -> adapt -> repeat.

    python -m therapy.coach reach_01.csv                        # stub robot, webcam
    python -m therapy.coach reach_01.csv --robot 192.168.1.20:5555   # sim on another machine
    python -m therapy.coach reach_01.csv --child fake --robot stub-fast --no-window   # end-to-end, no hardware

The adaptation is a rule table (`adapt`), not learning, so a therapist can
read exactly why the robot changed its demonstration:

    weakest sub-score   next demonstration
    reach               smaller and slower; grows back as the child succeeds
    height              pause at the top of the motion; cue "reach up high"
    elbow               replay only the extension part, slowly
    smoothness          slower whole motion
    tempo               match the child's pace
    3 good reps         step speed and size back up toward the taught motion
    3 reps getting worse   short rest, restart smaller
"""

import argparse
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .demo import Demo
from .motion import Motion
from .robot_link import Robot, make_robot
from .robot_path import DEFAULT_URDF, RobotKinematics
from .scorer import Score, score_attempt

GOOD = 0.75           # composite fraction that counts as a good rep
SPEED_MIN, SIZE_MIN = 0.4, 0.4
SPEED_STEP, SIZE_STEP = 0.2, 0.2


@dataclass
class Plan:
    speed: float = 1.0
    size: float = 1.0
    segment: Optional[Tuple[float, float]] = None
    hold: Optional[Tuple[float, float]] = None

    def knobs(self) -> dict:
        return {"speed": self.speed, "size": self.size, "segment": self.segment, "hold": self.hold}

    def describe(self) -> str:
        s = f"speed {self.speed:.1f}  size {self.size:.1f}"
        if self.segment:
            s += f"  part {self.segment[0]:.0%}-{self.segment[1]:.0%}"
        if self.hold:
            s += f"  hold {self.hold[1]:.0f}s"
        return s


@dataclass
class Rep:
    n: int
    plan: Plan
    score: Score
    child: Motion
    ref: Motion
    message: str


# ---- the rule table ----------------------------------------------------------

def _extension_segment(ref: Motion) -> Tuple[float, float]:
    """The part of the demo where the elbow goes from most bent to straightest."""
    e = ref.elbow if ref.elbow is not None else -np.linalg.norm(ref.hand - ref.hand[0], axis=1)
    i_min = int(np.argmin(e))
    i_max = i_min + int(np.argmax(e[i_min:]))
    if i_max <= i_min:
        return (0.0, 1.0)
    return (ref.t[i_min] / ref.duration, min(1.0, ref.t[i_max] / ref.duration + 0.05))


def _top_fraction(ref: Motion) -> float:
    return float(ref.t[int(np.argmax(ref.hand[:, 2]))] / ref.duration)


def adapt(plan: Plan, history: List[Rep], full_ref: Motion) -> Tuple[Plan, str]:
    """Decide the next demonstration from the scores so far. Returns (plan, message)."""
    last = history[-1]
    p = Plan(**asdict(plan))
    sc = last.score

    if not sc.moved:
        return p, "I didn't see you move. Let's try again!"

    recent = [r.score.composite for r in history[-3:]]
    if len(recent) == 3 and recent[0] > recent[1] > recent[2] and recent[2] < 100 * GOOD:
        p.size = max(SIZE_MIN, p.size - SIZE_STEP)
        p.segment, p.hold = None, None
        return p, "Let's take a short break, then try a smaller one."

    if len(history) >= 3 and all(r.score.good(GOOD) for r in history[-3:]):
        if p.segment or p.hold:
            p.segment, p.hold = None, None
            return p, "Great! Now let's do the whole movement."
        if p.size < 1.0 or p.speed < 1.0:
            p.size = min(1.0, p.size + SIZE_STEP)
            p.speed = min(1.0, p.speed + SPEED_STEP)
            return p, "You're doing great! A bit bigger this time."
        return p, "Perfect! Do it just like that."

    if sc.good(GOOD):
        return p, "Nice one! Again."

    weakest = sc.weakest()
    if weakest == "reach":
        p.size = max(SIZE_MIN, p.size - SIZE_STEP)
        p.speed = max(SPEED_MIN, p.speed - SPEED_STEP)
        return p, "Let's make it a little smaller. Watch the robot's hand."
    if weakest == "height":
        p.hold = (_top_fraction(full_ref), 1.0)
        return p, "Reach up high, like the robot!"
    if weakest == "elbow":
        p.segment = _extension_segment(full_ref)
        p.speed = max(SPEED_MIN, p.speed - SPEED_STEP)
        return p, "Watch how the arm stretches out straight."
    if weakest == "smoothness":
        p.speed = max(SPEED_MIN, p.speed - SPEED_STEP)
        return p, "Nice and smooth, one movement."
    if weakest == "tempo":
        ratio = sc.notes.get("ref_dur", 1) / max(sc.notes.get("child_dur", 1), 1e-3)
        if ratio > 1:          # the child was slower: meet them there
            p.speed = float(np.clip(p.speed / ratio, SPEED_MIN, 1.0))
            return p, "Good. Let's go at your speed."
        return p, "Not so fast! Follow the robot."
    if weakest == "direction":
        p.speed = max(SPEED_MIN, p.speed - SPEED_STEP)
        return p, "Watch which way the robot's hand goes."
    # shape
    p.speed = max(SPEED_MIN, p.speed - SPEED_STEP)
    return p, "Watch closely and copy the robot."


# ---- helpers -----------------------------------------------------------------

class Voice:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.cmd = None
        if enabled:
            if platform.system() == "Darwin":
                self.cmd = ["say"]
            elif shutil.which("espeak"):
                self.cmd = ["espeak"]
        self.proc = None

    def say(self, text: str):
        print(f"[robot says] {text}")
        if not self.enabled or not self.cmd:
            return
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = subprocess.Popen(self.cmd + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class FakeChild:
    """Stands in for the webcam: a scaled, noisy, slower copy of the robot's
    motion that improves every rep. Lets the whole loop run with no camera."""

    arm = "left"
    calibrated = True

    def __init__(self, reach: float = 0.5, improve: float = 0.12, noise: float = 0.02,
                 slow: float = 1.4, seed: int = 0):
        self.reach, self.improve, self.noise, self.slow = reach, improve, noise, slow
        self.rng = np.random.default_rng(seed)
        self.ref: Optional[Motion] = None

    def read(self):
        return None, None

    def record(self, seconds=5.0, on_frame=None, **_) -> Motion:
        r = self.ref
        hand = r.hand[0] + self.reach * (r.hand - r.hand[0]) + self.rng.normal(0, self.noise, r.hand.shape)
        elbow = r.elbow[0] + self.reach * (r.elbow - r.elbow[0]) if r.elbow is not None else None
        m = Motion(r.t * self.slow, hand, elbow, "fake", {"moved": True})
        self.reach = min(1.0, self.reach + self.improve)
        self.slow = max(1.0, self.slow - 0.1)
        return m

    def close(self):
        pass


def draw_panel(frame, rep: Optional[Rep], plan: Plan, ref: Motion, live_msg: str, n: int, width=340):
    """Webcam frame on the left, status panel on the right. Returns a new image."""
    import cv2
    h = 480 if frame is None else frame.shape[0]
    left = np.zeros((h, 640, 3), np.uint8) if frame is None else frame
    panel = np.full((h, width, 3), 30, np.uint8)
    y = 34
    cv2.putText(panel, f"Rep {n}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA); y += 28
    cv2.putText(panel, plan.describe(), (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA); y += 30
    if rep is not None:
        sc = rep.score
        col = (0, 220, 0) if sc.good(GOOD) else (0, 200, 255) if sc.moved else (0, 0, 255)
        cv2.putText(panel, f"{sc.composite:.0f}" if sc.moved else "--", (16, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3, cv2.LINE_AA)
        y += 50
        for k, v in sc.sub.items():
            cv2.putText(panel, k, (16, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (110, y), (110 + int(200 * v), y + 10), (0, 220, 0) if v >= 0.7 else (0, 165, 255), -1)
            y += 18
        y += 8
    # front-view hand path: robot (grey) vs child (green); outward -> right, up -> up
    box = (16, y, width - 16, h - 40)
    cv2.rectangle(panel, box[:2], box[2:], (70, 70, 70), 1)

    pts = [m.hand[:, 1:3] for m in (ref, rep.child if rep is not None else None) if m is not None and len(m) > 1]
    allp = np.vstack(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(0) - 0.1, allp.max(0) + 0.1
    scale = min((box[2] - box[0]) / (hi[0] - lo[0]), (box[3] - box[1]) / (hi[1] - lo[1]))
    cx, cy = (box[0] + box[2]) / 2 - (lo[0] + hi[0]) / 2 * scale, (box[1] + box[3]) / 2 + (lo[1] + hi[1]) / 2 * scale

    def to_px(m: Motion):
        return np.column_stack([cx + m.hand[:, 1] * scale, cy - m.hand[:, 2] * scale]).astype(np.int32)

    if len(ref):
        cv2.polylines(panel, [to_px(ref)], False, (160, 160, 160), 2)
    if rep is not None and len(rep.child) > 1:
        cv2.polylines(panel, [to_px(rep.child.smoothed(5))], False, (0, 220, 0), 2)
    cv2.putText(panel, "robot", (box[0] + 6, box[3] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "you", (box[0] + 66, box[3] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, (rep.message if rep else "")[:34], (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if live_msg:
        cv2.putText(left, live_msg, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([left, panel])


# ---- the loop ----------------------------------------------------------------

class Coach:
    def __init__(self, demo: Demo, robot: Robot, observer, kin: RobotKinematics,
                 voice: Voice, window: bool = True, record_s: float = 5.0,
                 session_dir: Optional[Path] = None):
        self.demo, self.robot, self.obs, self.kin, self.voice = demo, robot, observer, kin, voice
        self.window, self.record_s = window, record_s
        self.full_ref = kin.hand_path(demo)
        self.plan = Plan()
        self.history: List[Rep] = []
        self.session_dir = session_dir
        if session_dir:
            session_dir.mkdir(parents=True, exist_ok=True)
        self._last_rep: Optional[Rep] = None
        self._ref = self.full_ref

    def show(self, frame, sample, msg: str):
        if not self.window:
            return
        import cv2
        from .observer import draw_sample
        if frame is not None and hasattr(self.obs, "arm"):
            frame = draw_sample(frame, sample, self.obs.arm)
        cv2.imshow("coach", draw_panel(frame, self._last_rep, self.plan, self._ref, msg, len(self.history) + 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def idle(self, seconds: float, msg: str):
        """Keep the camera/window alive for `seconds`."""
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            frame, s = self.obs.read()
            self.show(frame, s, msg)
            if frame is None:
                time.sleep(0.03)

    def calibrate(self):
        if getattr(self.obs, "calibrated", True):
            return
        self.voice.say("Hold your arm out straight.")
        self.idle(1.5, "Hold your arm out straight")
        L = self.obs.calibrate(2.0, on_frame=lambda f, s, m: self.show(f, s, m))
        print(f"arm length {L:.3f} m ({'calibrated' if self.obs.calibrated else 'fallback'})")

    def rep(self) -> Rep:
        n = len(self.history) + 1
        demo_i = self.demo.adapt(**self.plan.knobs())
        ref = self.kin.hand_path(demo_i)
        self._ref = ref
        if isinstance(self.obs, FakeChild):
            self.obs.ref = ref

        self.voice.say("Watch the robot.")
        done = self.robot.play_async(demo_i)
        while not done.is_set():
            frame, s = self.obs.read()
            self.show(frame, s, "Watch the robot")
            if frame is None:
                done.wait(0.03)

        self.voice.say("Your turn!")
        child = self.obs.record(self.record_s, on_frame=lambda f, s, m: self.show(f, s, m))
        score = score_attempt(child, ref)
        print(f"rep {n}: {score.summary()}")

        r = Rep(n, Plan(**asdict(self.plan)), score, child, ref, "")
        self.history.append(r)
        self.plan, r.message = adapt(self.plan, self.history, self.full_ref)
        self._last_rep = r
        self.voice.say(r.message)
        self.log(r)
        return r

    def log(self, r: Rep):
        if not self.session_dir:
            return
        with open(self.session_dir / "session.jsonl", "a") as f:
            f.write(json.dumps({"rep": r.n, "plan": asdict(r.plan), "score": r.score.composite,
                                "sub": r.score.sub, "moved": r.score.moved, "message": r.message,
                                "next_plan": asdict(self.plan)}) + "\n")
        if len(r.child):
            r.child.save(self.session_dir / f"rep{r.n:02d}_child.npz")
        r.ref.save(self.session_dir / f"rep{r.n:02d}_robot.npz")

    def run(self, reps: int, pause_s: float = 3.0):
        self.calibrate()
        try:
            for _ in range(reps):
                r = self.rep()
                if isinstance(self.obs, FakeChild):
                    continue
                is_rest = r.message.startswith("Let's take a short break")
                self.idle(10.0 if is_rest else pause_s, r.message)
        except KeyboardInterrupt:
            pass
        finally:
            self.obs.close()
            self.robot.close()
        if self.history:
            print("\nsession: " + "  ".join(f"{r.score.composite:.0f}" for r in self.history))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", help="taught demonstration CSV (from bag_to_csv.py)")
    ap.add_argument("--robot", default="stub", help="'stub', 'stub-fast', or host:port of play_demo.py --serve")
    ap.add_argument("--child", default="0", help="camera index, video file, or 'fake'")
    ap.add_argument("--arm", default=None, choices=["left", "right"],
                    help="the child's arm (default: mirror of the robot's)")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--record-seconds", type=float, default=5.0)
    ap.add_argument("--raw", action="store_true", help="don't clean pauses out of the recording")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no speech")
    ap.add_argument("--session-dir", default=None, help="where to log reps (default: sessions/<time>)")
    args = ap.parse_args()

    demo = Demo.load_csv(args.demo)
    if not args.raw:
        demo = demo.clean()
    if len(demo) < 2:
        raise SystemExit("demonstration has no movement in it")
    kin = RobotKinematics(args.urdf)
    arm = args.arm or ("left" if demo.arm == "right" else "right")
    print(f"demo '{demo.name}': {demo.duration:.1f} s, robot {demo.arm} arm -> child {arm} arm")

    if args.child == "fake":
        obs = FakeChild()
        realtime = False
    else:
        from .observer import Observer
        cam = int(args.child) if args.child.isdigit() else args.child
        obs = Observer(cam, arm=arm)
        realtime = True
    robot = make_robot(args.robot, realtime=realtime)
    session = Path(args.session_dir) if args.session_dir else Path("sessions") / time.strftime("%Y%m%d-%H%M%S")
    coach = Coach(demo, robot, obs, kin, Voice(not args.quiet and args.child != "fake"),
                  window=not args.no_window, record_s=args.record_seconds, session_dir=session)
    coach.run(args.reps, pause_s=0.0 if args.child == "fake" else 3.0)


if __name__ == "__main__":
    main()
