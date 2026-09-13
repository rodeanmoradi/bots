"""
Run arm_xy.py against a live camera.

A test harness, not part of the library. It calls arm_xy once per frame and
shows three things at once, so you can see whether the numbers are sane:

  - the MediaPipe skeleton drawn on the video
  - a small plot of the (4, 2) array arm_xy actually returned, in body
    coordinates with the hip origin marked
  - the raw numbers

Controls:
    SPACE   start / stop collecting frames into a trajectory
    S       save the collected trajectory
    C       clear the collected trajectory
    Esc     quit

Usage:
    python run_arm_xy.py
    python run_arm_xy.py --side right --plane side --claw
    python run_arm_xy.py --list-cameras
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import arm_xy
from arm_xy import PLANES, arm_xy as get_arm_xy, open_camera

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LABELS = ("shoulder", "elbow", "wrist", "hand")
POINT_COLOURS = ((255, 180, 0), (0, 255, 120), (200, 120, 255), (0, 200, 255))


def draw_skeleton(frame):
    """Draw whatever arm_xy saw on its last call."""
    if arm_xy.last_pose and arm_xy.last_pose.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, arm_xy.last_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    if arm_xy.last_hands and arm_xy.last_hands.multi_hand_landmarks:
        for h in arm_xy.last_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, h, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())


def draw_points_plot(frame, pts, trail, size=240, margin=12):
    """Bottom-right panel plotting the returned points in body coordinates.

    This is the useful one: it shows the array arm_xy handed back, not the
    camera image, so a wrong plane or a bad origin is obvious.
    """
    h, w = frame.shape[:2]
    x0, y0 = w - size - margin, h - size - margin
    panel = frame[y0:y0 + size, x0:x0 + size]
    cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0, dst=panel)
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (90, 90, 90), 1)

    # Body coords are roughly -0.6..1.2 in both axes. Fix the window rather
    # than autoscaling, so the arm visibly moves instead of the axes moving.
    lo, hi = -0.7, 1.3

    def to_px(p):
        u = (p[0] - lo) / (hi - lo)
        v = (p[1] - lo) / (hi - lo)
        return int(x0 + u * size), int(y0 + (1 - v) * size)   # y up on screen

    ox, oy = to_px((0.0, 0.0))
    cv2.line(frame, (x0, oy), (x0 + size, oy), (70, 70, 70), 1)
    cv2.line(frame, (ox, y0), (ox, y0 + size), (70, 70, 70), 1)
    cv2.putText(frame, "hip", (ox + 4, oy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (150, 150, 150), 1, cv2.LINE_AA)

    for old in trail:
        cv2.circle(frame, to_px(old[-1]), 1, (120, 120, 120), -1)

    if pts is not None:
        px = [to_px(p) for p in pts]
        for a, b in zip(px, px[1:]):
            cv2.line(frame, a, b, (0, 200, 255), 3, cv2.LINE_AA)
        for p, c in zip(px, POINT_COLOURS):
            cv2.circle(frame, p, 6, c, -1, cv2.LINE_AA)


def draw_panel(frame, lines, colour, width=560):
    lh, pad = 26, 10
    box = min(lh * len(lines) + pad * 2, frame.shape[0])
    plate = frame[0:box, 0:min(width, frame.shape[1])]
    cv2.addWeighted(plate, 0.35, np.zeros_like(plate), 0.65, 0, dst=plate)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (12, pad + 18 + lh * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def save_trajectory(traj, args, out_dir=Path("trajectories")):
    """Write the collected frames as .npy plus a readable .json."""
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    arr = np.stack(traj)                       # (n_frames, 4, 2)
    npy = out_dir / f"{args.side}_{args.plane}_{stamp}.npy"
    js = npy.with_suffix(".json")
    np.save(npy, arr)
    js.write_text(json.dumps({
        "side": args.side,
        "plane": args.plane,
        "shape": list(arr.shape),
        "order": list(LABELS),
        "units": "raw, roughly metres, origin at hip centre, unscaled",
        "frames": arr.round(5).tolist(),
    }, indent=2))
    return npy, js, arr.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0",
                    help="Camera index, /dev/video path, or stream URL such as "
                         "http://192.168.1.50:8080/video")
    ap.add_argument("--fps", type=int, default=30,
                    help="Frame rate to request from the camera. Lower it if "
                         "frames arrive corrupt over usbipd.")
    ap.add_argument("--width", type=int, default=640,
                    help="Frame size to request. MediaPipe pose does not need "
                         "more, and smaller frames survive usbipd intact.")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--plane", choices=sorted(PLANES), default="side")
    ap.add_argument("--claw", action="store_true",
                    help="Also read the claw open/shut state (a bit slower).")
    ap.add_argument("--flip", action="store_true",
                    help="Mirror the frame if the feed is not already mirrored.")
    ap.add_argument("--list-cameras", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        from pygrabber.dshow_graph import FilterGraph
        for i, n in enumerate(FilterGraph().get_input_devices()):
            print(f"{i}: {n}")
        return

    cap = open_camera(args.camera, args.width, args.height, fps=args.fps)
    print(f"requested {args.width}x{args.height}@{args.fps}, got "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"@{cap.get(cv2.CAP_PROP_FPS):g}")
    print(f"side={args.side}  plane={args.plane}  claw={args.claw}")
    print("SPACE collect   S save   C clear   Esc quit")

    traj, collecting = [], False
    trail = []                      # recent hand positions, for the mini plot
    msg = ""
    fps_t, fps_n, fps = time.time(), 0, 0.0
    last_frame_t, last_warn_t = time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Without this a dead stream just spins silently and no window ever opens.
                now = time.time()
                if now - last_frame_t > 3.0 and now - last_warn_t > 5.0:
                    error = getattr(cap, "last_error", None)
                    print(f"no frames from camera for {now - last_frame_t:.0f} s"
                          + (f", last error: {error}" if error else ""))
                    last_warn_t = now
                continue
            last_frame_t = time.time()
            if args.flip:
                frame = cv2.flip(frame, 1)

            result = get_arm_xy(frame, side=args.side, plane=args.plane,
                                with_claw=args.claw)
            pts, claw = result if args.claw else (result, None)

            if pts is not None:
                trail.append(pts)
                trail = trail[-120:]
                if collecting:
                    traj.append(pts)

            draw_skeleton(frame)
            draw_points_plot(frame, pts, trail)

            if pts is not None:
                lines = [f"{n:9s} x{p[0]:+7.3f}  y{p[1]:+7.3f}"
                         for n, p in zip(LABELS, pts)]
                if args.claw:
                    lines.append(f"claw      {'OPEN' if claw else 'SHUT'}")
                colour = (0, 255, 0)
            else:
                lines = ["arm_xy returned None",
                         "get your hips, shoulders and arm in frame"]
                colour = (0, 165, 255)

            lines.insert(0, f"side={args.side}  plane={args.plane}  "
                            f"{'COLLECTING' if collecting else 'idle'}  "
                            f"{len(traj)} frames")
            if msg:
                lines.append(msg)
            lines.append(f"{fps:.1f} fps   SPACE collect  S save  C clear  Esc quit")
            draw_panel(frame, lines, colour)

            if collecting:
                cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (60, 60, 255), -1)

            cv2.imshow("run_arm_xy", frame)

            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == 32:
                collecting = not collecting
                msg = "collecting..." if collecting else f"stopped at {len(traj)} frames"
            elif k in (ord("s"), ord("S")):
                if traj:
                    npy, js, shape = save_trajectory(traj, args)
                    msg = f"saved {npy.name}  {shape}"
                    print(f"saved {npy}  shape={shape}")
                    print(f"      {js}")
                else:
                    msg = "nothing collected yet"
            elif k in (ord("c"), ord("C")):
                traj, msg = [], "cleared"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm_xy.reset()
        if traj:
            print(f"exited with {len(traj)} uncollected frames "
                  f"(press S before Esc to keep them)")


if __name__ == "__main__":
    main()
