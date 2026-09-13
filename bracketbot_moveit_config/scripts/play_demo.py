#!/usr/bin/env python3
"""Play a demonstration on the (sim) arm, or serve trajectories to the coach.

    # play a CSV once, as taught / half speed / half size
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv --speed 0.5 --size 0.5

    # serve: the therapy coach (on this or another machine) sends trajectories
    ros2 run bracketbot_moveit_config play_demo.py --serve 5555

The arm first glides from wherever it is to the demo's start pose over
--approach seconds, plays the demo, then glides back to the start pose over
--return-time seconds, so every demonstration starts and ends in the same place.

Wire protocol for --serve: one JSON line per request,
  {"joints": [...], "t": [...], "q": [[...], ...]}
      ->  {"event": "accepted", "approach": s, "duration": s, "return": s}   as soon as the controller accepts
      ->  {"ok": true, "duration": s}                                        when the arm has finished
  {"cmd": "ping"}  ->  {"ok": true}
Requests are handled one at a time.
"""

import argparse
import json
import signal
import socketserver
import sys
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

CONTROLLERS = {"r": "right_arm_controller", "l": "left_arm_controller"}


def _import_therapy():
    """The therapy package lives at the repo root; find it from this file (symlink-install) or PYTHONPATH."""
    try:
        import therapy  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from therapy.demo import Demo
    return Demo


def _point(seconds, positions):
    p = JointTrajectoryPoint()
    p.positions = [float(v) for v in positions]
    p.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
    return p


class DemoPlayer(Node):
    def __init__(self, approach_s: float, return_s: float):
        super().__init__("play_demo")
        self.approach_s = approach_s
        self.return_s = return_s
        self.clients_ = {}

    def client(self, controller: str) -> ActionClient:
        if controller not in self.clients_:
            self.clients_[controller] = ActionClient(self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        return self.clients_[controller]

    def play(self, joints, t, q, on_accepted=None) -> float:
        """Blocking. Returns the total duration (approach + demo + return) or raises.

        on_accepted(approach_s, demo_s, return_s) is called once the controller has taken the goal.
        """
        t = np.asarray(t, float)
        q = np.asarray(q, float).reshape(len(t), -1)
        controller = CONTROLLERS.get(joints[0][0], "right_arm_controller")
        ac = self.client(controller)
        if not ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{controller} action server not available")

        traj = JointTrajectory()
        traj.joint_names = list(joints)
        traj.points = [_point(ti, qi) for ti, qi in zip(t + self.approach_s, q)]
        return_s = self.return_s if not np.allclose(q[-1], q[0], atol=1e-3) else 0.0
        if return_s > 0:
            traj.points.append(_point(t[-1] + self.approach_s + return_s, q[0]))

        send = ac.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj))
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if not handle.accepted:
            raise RuntimeError("trajectory rejected by controller")
        if on_accepted:
            on_accepted(self.approach_s, float(t[-1]), return_s)
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"controller error {code}: {res.result().result.error_string}")
        return float(self.approach_s + t[-1] + return_s)


def serve(player: DemoPlayer, port: int):
    class Handler(socketserver.StreamRequestHandler):
        def send(self, message):
            self.wfile.write((json.dumps(message) + "\n").encode())
            self.wfile.flush()

        def handle(self):
            line = self.rfile.readline()
            if not line:
                return
            try:
                req = json.loads(line)
                if req.get("cmd") == "ping":
                    reply = {"ok": True}
                else:
                    player.get_logger().info(f"playing '{req.get('name', '?')}' ({len(req['t'])} pts, {req['t'][-1]:.1f} s)")
                    dur = player.play(req["joints"], req["t"], req["q"],
                                      on_accepted=lambda a, d, r: self.send(
                                          {"event": "accepted", "approach": a, "duration": d, "return": r}))
                    reply = {"ok": True, "duration": dur}
            except Exception as e:  # report to the client instead of dying
                player.get_logger().error(str(e))
                reply = {"ok": False, "error": str(e)}
            self.send(reply)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as srv:
        player.get_logger().info(f"serving trajectories on port {port}")
        srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="demonstration CSV from bag_to_csv.py")
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve trajectories on this TCP port")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="don't clean pauses out of the recording")
    ap.add_argument("--approach", type=float, default=2.0, help="seconds to glide to the start pose")
    ap.add_argument("--return-time", type=float, default=2.0,
                    help="seconds to glide back to the start pose afterwards (0 = stay at the end pose)")
    args, ros_args = ap.parse_known_args()
    if not args.csv and args.serve is None:
        ap.error("give a CSV to play, or --serve PORT")

    # rclpy's own SIGINT/SIGTERM handlers only shut its context down, which serve_forever() never notices,
    # so the server outlived every stop. Let both signals end the program instead (SIGINT explicitly too:
    # a process started in the background inherits it ignored).
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    player = DemoPlayer(args.approach, args.return_time)
    try:
        if args.serve is not None:
            serve(player, args.serve)
        else:
            Demo = _import_therapy()
            demo = Demo.load_csv(args.csv)
            if not args.raw:
                demo = demo.clean()
            demo = demo.adapt(speed=args.speed, size=args.size)
            print(f"playing {demo.name}: {len(demo)} points, {args.approach:.1f} s approach + "
                  f"{demo.duration:.1f} s + {args.return_time:.1f} s return")
            player.play(demo.joints, demo.t, demo.q)
            print("done")
    except KeyboardInterrupt:
        pass
    finally:
        player.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
