"""
Mirror-therapy arm tracker.

Tracks one arm from a webcam feed (e.g. Iriun virtual camera), converts to
joint angles in a torso-fixed frame, mirrors them to the opposite side, and
streams JSON over UDP. The measurement itself lives in tracking.py.

Points are (x, y) in TORSO LENGTHS from the hip midpoint, in the plane named
by --view. Inverse kinematics downstream turns them into robot joint
commands; a single scale factor converts torso lengths to robot metres.

Output packet (one JSON object per frame, newline-free):
{
  "t": 1757712345.123,          # unix time at capture
  "valid": true,                # false when tracking is lost
  "shoulder": [0.02, 1.00],
  "elbow":    [0.31, 0.62],
  "hand":     [0.68, 0.55],
  "claw_open": true,
  "aperture": 0.85,             # 0 = closed pinch, 1 = fully open
  "grip_valid": true            # false = hand not seen, claw is a held value
}

Usage:
    python arm_tracker.py --list-cameras
    python arm_tracker.py --camera iriun --source-side right --view side
"""

import argparse
import json
import socket
import time

import cv2

import tracking
from tracking import POINTS, VIEWS, ArmTracker, draw_overlay, draw_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="iriun",
                    help="Capture index, or part of a device name such as "
                         "'iriun'. Names survive an index reshuffle on reboot.")
    ap.add_argument("--list-cameras", action="store_true",
                    help="Print the available capture devices and exit.")
    ap.add_argument("--source-side", choices=["left", "right"], default="right",
                    help="The subject's UNAFFECTED arm, i.e. the one being tracked.")
    ap.add_argument("--view", choices=sorted(VIEWS), default="side",
                    help="Camera placement. Sets which plane x and y are "
                         "measured in. Must match the reference take.")
    ap.add_argument("--udp", default="127.0.0.1:5005")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--flip", action="store_true",
                    help="Horizontally flip the frame. Use if the feed is not "
                         "already mirrored, so left/right labels match reality.")
    args = ap.parse_args()

    if args.list_cameras:
        names = tracking.list_cameras()
        if names is None:
            raise SystemExit("pip install pygrabber to list device names")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        return

    host, port = args.udp.split(":")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (host, int(port))

    print(f"view '{args.view}': {VIEWS[args.view]['help']}")
    cap = tracking.open_camera(args.camera)
    tracker = ArmTracker(side=args.source_side, view=args.view)
    print(f"streaming to {host}:{port}")

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

            packet = {"t": t, "valid": sample is not None}
            if sample is not None:
                # Mirroring is already handled in arm_points: the lateral
                # axis is signed towards the tracked arm, so a left-arm and a
                # right-arm take produce the same numbers. If the robot moves
                # the wrong way in sim, negate x here and nowhere else.
                packet.update({p: [round(sample[p][0], 4), round(sample[p][1], 4)]
                               for p in POINTS})
                packet.update(claw_open=sample["claw_open"],
                              aperture=round(sample["aperture"], 4),
                              grip_valid=sample["grip_valid"])

            sock.sendto(json.dumps(packet).encode(), addr)

            fps_n += 1
            if t - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (t - fps_t), 0, t

            if not args.no_preview:
                draw_overlay(frame, pose_res, hand_res, args.source_side)
                if sample is not None:
                    lines = tracking.sample_lines(sample)
                    colour = (0, 255, 0)
                else:
                    lines = ["TRACKING LOST - get your torso and arm in frame"]
                    colour = (0, 165, 255)
                lines.append(f"{fps:.1f} fps   {args.source_side} arm   Esc to quit")
                draw_panel(frame, lines, colour)

                cv2.imshow("arm tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:   # Esc
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        sock.close()


if __name__ == "__main__":
    main()
