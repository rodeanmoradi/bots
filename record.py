"""
Record a session of patient attempts.

The correct motion is NOT captured here - it lives in Gazebo. This script
watches what the human actually does and records their takes in order: the
first uncorrected attempt, then each subsequent attempt as the robot guides
them at a higher correction level. The point of a session is the
progression across takes, so takes are numbered and kept together.

Each take holds the shoulder, elbow and hand as (x, y) in TORSO LENGTHS from
the hip midpoint, plus a claw open/shut flag. Normalising by torso length
means a take is comparable across people; one scale factor converts torso
lengths to robot metres.

Layout:
    recordings/<session>/session.json      manifest, takes in order
    recordings/<session>/take_01_a000.json attempt 1, 0% correction
    recordings/<session>/take_02_a025.json attempt 2, 25% correction

Controls:
    SPACE   start recording (after a 3 second countdown) / stop recording
    - / =   lower / raise the correction level for the NEXT take
    D       discard the take you just recorded
    P       plot the take you just recorded
    C       compare every take in this session
    Esc     quit

Usage:
    python record.py --subject p1 --action throw_ball --side right --view side
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

import tracking
from tracking import POINTS, VIEWS, ArmTracker, draw_overlay, draw_panel

RECORDINGS = Path(__file__).parent / "recordings"
COUNTDOWN = 3.0
ALPHA_STEP = 25          # correction increments the interface offers


# ------------------------------------------------------------------ session

class Session:
    """A run of attempts by one person at one action, kept in one folder."""

    def __init__(self, args, root=RECORDINGS):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = args.session or f"{args.subject}_{args.action}_{stamp}"
        self.dir = root / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.takes = []

    @property
    def next_attempt(self):
        return len(self.takes) + 1

    def save(self, samples, alpha, dropped):
        """Write one take plus the refreshed manifest. Returns its path."""
        a = self.args
        t0 = samples[0]["t"]
        attempt = self.next_attempt
        path = self.dir / f"take_{attempt:02d}_a{alpha:03d}.json"

        take = {
            "session": self.name,
            "subject": a.subject,
            "action": a.action,
            "side": a.side,
            "view": a.view,
            "attempt": attempt,
            # How much the robot was correcting the person on this take.
            # 0 = the person unaided, 100 = the robot driving the ideal motion.
            "correction_pct": alpha,
            "recorded": datetime.now().isoformat(timespec="seconds"),
            "duration": round(samples[-1]["t"] - t0, 3),
            "n_samples": len(samples),
            "points": list(POINTS),
            "units": "torso lengths (hip midpoint to shoulder midpoint = 1.0), "
                     "origin at hip midpoint",
            "axes": {"x": VIEWS[a.view]["x_axis"], "y": VIEWS[a.view]["y_axis"],
                     "note": "torso axis indices: 0 = lateral (+ve towards the "
                             "tracked arm), 1 = up, 2 = forward"},
            "samples": [
                {"t": round(s["t"] - t0, 4),
                 **{p: [round(s[p][0], 5), round(s[p][1], 5)] for p in POINTS},
                 "claw_open": s["claw_open"],
                 "aperture": round(s["aperture"], 4),
                 "grip_valid": s["grip_valid"]}
                for s in samples
            ],
        }
        path.write_text(json.dumps(take, indent=2))

        self.takes.append({
            "file": path.name,
            "attempt": attempt,
            "correction_pct": alpha,
            "duration": take["duration"],
            "n_samples": take["n_samples"],
            "warnings": take_warnings(samples, dropped),
        })
        self.write_manifest()
        return path

    def drop_last(self):
        if not self.takes:
            return None
        entry = self.takes.pop()
        (self.dir / entry["file"]).unlink(missing_ok=True)
        self.write_manifest()
        return entry["file"]

    def write_manifest(self):
        a = self.args
        (self.dir / "session.json").write_text(json.dumps({
            "session": self.name,
            "subject": a.subject,
            "action": a.action,
            "side": a.side,
            "view": a.view,
            "created": datetime.now().isoformat(timespec="seconds"),
            "note": "Reference motion comes from Gazebo, not from this file. "
                    "These are the human attempts, in order.",
            "takes": self.takes,
        }, indent=2))

    def take_paths(self):
        return [self.dir / t["file"] for t in self.takes]


# ------------------------------------------------------------------ checking

def take_warnings(samples, dropped):
    """Sanity checks worth seeing before you trust a take."""
    out = []
    dur = samples[-1]["t"] - samples[0]["t"]
    fps = len(samples) / dur if dur > 0 else 0

    if dur < 0.5:
        out.append(f"very short take ({dur:.2f}s) - probably a misfire")
    if dropped:
        pct = 100 * dropped / (len(samples) + dropped)
        out.append(f"tracking lost on {dropped} frames ({pct:.0f}%)")
    if fps < 8:
        out.append(f"low capture rate ({fps:.1f} fps) - motion may be undersampled")

    grip_lost = sum(1 for s in samples if not s["grip_valid"])
    if grip_lost > len(samples) * 0.3:
        pct = 100 * grip_lost / len(samples)
        out.append(f"hand not seen on {pct:.0f}% of frames - claw is unreliable")

    if all(s["claw_open"] for s in samples) or not any(s["claw_open"] for s in samples):
        out.append("claw never changed state - no release captured")

    span = max(abs(s["hand"][0] - samples[0]["hand"][0]) for s in samples)
    if span < 0.15:
        out.append(f"hand barely moved along x ({span:.2f} torso lengths) - "
                   "wrong --view for this motion?")
    return out


def resample(take, n=100):
    """Resample a take onto n evenly spaced points over 0-100% of its span.

    Attempts differ in duration - a guided take is usually slower than an
    unaided one - so takes can only be compared on a common phase axis.
    """
    import numpy as np

    s = take["samples"]
    t = np.array([x["t"] for x in s])
    if t[-1] <= t[0]:
        return None
    phase = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, n)

    out = {"phase": grid}
    for p in POINTS:
        out[p] = np.stack([
            np.interp(grid, phase, [x[p][0] for x in s]),
            np.interp(grid, phase, [x[p][1] for x in s]),
        ], axis=1)
    out["aperture"] = np.interp(grid, phase, [x["aperture"] for x in s])
    return out


# ------------------------------------------------------------------ plotting

def plot_take(path):
    """Plot one take. Blocking, so only call between takes."""
    import matplotlib.pyplot as plt

    take = json.loads(Path(path).read_text())
    s = take["samples"]
    t = [x["t"] for x in s]

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(f"{take['action']}  |  attempt {take['attempt']}  |  "
                 f"{take['correction_pct']}% correction  |  {take['duration']}s")

    ax0 = fig.add_subplot(1, 2, 1)
    for p, style in zip(POINTS, ("o-", "s-", "^-")):
        ax0.plot([x[p][0] for x in s], [x[p][1] for x in s], style,
                 ms=2.5, lw=1.2, label=p, alpha=0.8)
    for i in range(0, len(s), max(1, len(s) // 12)):
        ax0.plot([s[i][p][0] for p in POINTS], [s[i][p][1] for p in POINTS],
                 "-", color="0.6", lw=0.8, zorder=0)
    ax0.plot(0, 0, "k+", ms=12, label="hip origin")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (torso lengths)")
    ax0.set_ylabel("y (torso lengths)")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(2, 2, 2)
    for p in POINTS:
        ax1.plot(t, [x[p][0] for x in s], lw=1.4, label=f"{p} x")
        ax1.plot(t, [x[p][1] for x in s], lw=1.0, ls="--", label=f"{p} y")
    ax1.set_ylabel("position")
    ax1.legend(fontsize=6, ncol=3)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)
    ax2.plot(t, [x["aperture"] for x in s], lw=1.4, label="aperture")
    ax2.step(t, [1 if x["claw_open"] else 0 for x in s], lw=1.6,
             where="post", label="claw open")
    for x in s:
        if not x["grip_valid"]:
            ax2.axvspan(x["t"], x["t"] + 0.02, color="red", alpha=0.15, lw=0)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("claw")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_session(paths):
    """Overlay every take in the session, to show the progression."""
    import matplotlib.pyplot as plt

    takes = [json.loads(Path(p).read_text()) for p in paths]
    takes.sort(key=lambda d: d["attempt"])
    if not takes:
        print("no takes to compare yet")
        return

    cmap = plt.get_cmap("viridis")
    colours = [cmap(i / max(1, len(takes) - 1)) for i in range(len(takes))]

    fig = plt.figure(figsize=(13, 6))
    head = takes[0]
    fig.suptitle(f"{head['subject']}  |  {head['action']}  |  "
                 f"{head['side']} arm  |  {len(takes)} attempts")

    # Hand path per attempt: the clearest read on whether guiding is working.
    ax0 = fig.add_subplot(1, 2, 1)
    for tk, c in zip(takes, colours):
        s = tk["samples"]
        ax0.plot([x["hand"][0] for x in s], [x["hand"][1] for x in s],
                 lw=1.8, color=c, alpha=0.9,
                 label=f"#{tk['attempt']}  {tk['correction_pct']}%")
    ax0.plot(0, 0, "k+", ms=12)
    ax0.set_aspect("equal")
    ax0.set_title("hand path", fontsize=10)
    ax0.set_xlabel("x (torso lengths)")
    ax0.set_ylabel("y (torso lengths)")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # Same takes on a common phase axis, since durations differ.
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)
    for tk, c in zip(takes, colours):
        r = resample(tk)
        if r is None:
            continue
        pct = r["phase"] * 100
        ax1.plot(pct, r["hand"][:, 0], lw=1.6, color=c)
        ax2.plot(pct, r["aperture"], lw=1.6, color=c)
    ax1.set_ylabel("hand x")
    ax1.set_title("normalised to 0-100% of each attempt", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("aperture")
    ax2.set_xlabel("percent through the motion")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="p1",
                    help="Who is performing. Used to name the session folder.")
    ap.add_argument("--action", default="throw_ball",
                    help="Name of the motion being attempted.")
    ap.add_argument("--side", choices=["left", "right"], default="right",
                    help="The arm performing the motion, i.e. the AFFECTED one.")
    ap.add_argument("--view", choices=sorted(VIEWS), default="side",
                    help="Where the camera is placed. Sets which plane x and "
                         "y are measured in, and is stored in every take.")
    ap.add_argument("--session", default=None,
                    help="Resume or name a session folder. Default is a new "
                         "one stamped with the time.")
    ap.add_argument("--camera", default="iriun")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--flip", action="store_true",
                    help="Horizontally flip the frame if the feed is not "
                         "already mirrored.")
    args = ap.parse_args()

    if args.list_cameras:
        names = tracking.list_cameras()
        if names is None:
            raise SystemExit("pip install pygrabber to list device names")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        return

    print(f"view '{args.view}': {VIEWS[args.view]['help']}")
    session = Session(args)
    print(f"session: {session.dir}")

    cap = tracking.open_camera(args.camera)
    tracker = ArmTracker(side=args.side, view=args.view)

    state = "idle"          # idle -> countdown -> recording -> idle
    samples, dropped = [], 0
    countdown_end = 0.0
    alpha = 0               # correction level for the next take
    last_path, last_msg = None, []
    fps_t, fps_n, fps = time.time(), 0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.time()
            if args.flip:
                frame = cv2.flip(frame, 1)

            sample, pose_res, hand_res = tracker.process(frame, t)

            if state == "countdown" and t >= countdown_end:
                state, samples, dropped = "recording", [], 0

            if state == "recording":
                if sample is None:
                    dropped += 1
                else:
                    samples.append({"t": t, **sample})

            draw_overlay(frame, pose_res, hand_res, args.side)

            # --- on-screen panel ---
            head = (f"{args.subject}  {args.action}  |  attempt "
                    f"{session.next_attempt}  |  {alpha}% correction")
            if state == "countdown":
                lines = [f"GET READY  {countdown_end - t:.0f}", head]
                colour = (0, 215, 255)
            elif state == "recording":
                dur = (samples[-1]["t"] - samples[0]["t"]) if samples else 0.0
                lines = [f"REC  {dur:5.2f}s   {len(samples)} samples", head,
                         "SPACE to stop"]
                lines += tracking.sample_lines(sample) if sample else ["tracking lost"]
                colour = (60, 60, 255)
            else:
                lines = [head,
                         "SPACE rec   -/= correction   D discard   "
                         "P plot   C compare   Esc quit"]
                if last_msg:
                    lines += last_msg
                lines += tracking.sample_lines(sample) if sample else ["tracking lost"]
                colour = (0, 255, 0) if sample else (0, 165, 255)

            lines.append(f"{fps:.1f} fps")
            draw_panel(frame, lines, colour, width=620)

            if state == "recording":
                cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (60, 60, 255), -1)

            cv2.imshow("record attempts", frame)

            fps_n += 1
            if t - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (t - fps_t), 0, t

            # --- keys ---
            k = cv2.waitKey(1) & 0xFF
            if k == 27:                                  # Esc
                break
            elif k == 32:                                # SPACE
                if state == "idle":
                    state, countdown_end = "countdown", t + COUNTDOWN
                    last_path, last_msg = None, []
                elif state == "countdown":
                    state = "idle"
                elif state == "recording":
                    state = "idle"
                    if len(samples) < 2:
                        last_msg = ["nothing captured - was the arm in frame?"]
                    else:
                        last_path = session.save(samples, alpha, dropped)
                        warn = session.takes[-1]["warnings"]
                        last_msg = [f"saved {last_path.name}"] + warn
                        print(f"saved {last_path}")
                        for w in warn:
                            print(f"  warning: {w}")
                        # Step the correction up for the next attempt; the
                        # operator can still override with - and =.
                        alpha = min(100, alpha + ALPHA_STEP)
            elif state == "idle" and k in (ord("-"), ord("_")):
                alpha = max(0, alpha - ALPHA_STEP)
            elif state == "idle" and k in (ord("="), ord("+")):
                alpha = min(100, alpha + ALPHA_STEP)
            elif k in (ord("d"), ord("D")) and state == "idle" and last_path:
                name = session.drop_last()
                print(f"discarded {name}")
                last_msg, last_path = ["discarded"], None
                alpha = max(0, alpha - ALPHA_STEP)
            elif k in (ord("p"), ord("P")) and state == "idle" and last_path:
                # Close the capture window first: the plot blocks the loop.
                cv2.destroyAllWindows()
                plot_take(last_path)
            elif k in (ord("c"), ord("C")) and state == "idle":
                cv2.destroyAllWindows()
                plot_session(session.take_paths())
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
