#!/usr/bin/env python3
"""Play a demonstration on the (sim) arm, or serve trajectories to the coach.

    # play a CSV once, as taught / half speed / half size
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv
    ros2 run bracketbot_moveit_config play_demo.py reach_01.csv --speed 0.5 --size 0.5

    # serve: the therapy coach (on this or another machine) sends trajectories
    ros2 run bracketbot_moveit_config play_demo.py --serve 5555

The arm first glides from wherever it is to the demo's start pose over
--approach seconds, then plays the demo, so the first point never jumps.

Wire protocol for --serve: one JSON line per request,
  {"joints": [...], "t": [...], "q": [[...], ...]}   ->   {"ok": true, "duration": s}
  {"cmd": "ping"}                                    ->   {"ok": true}
Requests are handled one at a time; the reply is sent when the arm has finished.
"""

import argparse
import json
import socketserver
import sys
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
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


class DemoPlayer(Node):
    def __init__(self, approach_s: float):
        super().__init__("play_demo")
        self.approach_s = approach_s
        self.clients_ = {}

    def client(self, controller: str) -> ActionClient:
        if controller not in self.clients_:
            self.clients_[controller] = ActionClient(self, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        return self.clients_[controller]

    def play(self, joints, t, q) -> float:
        """Blocking. Returns the trajectory duration (incl. approach) or raises."""
        t = np.asarray(t, float)
        q = np.asarray(q, float).reshape(len(t), -1)
        controller = CONTROLLERS.get(joints[0][0], "right_arm_controller")
        ac = self.client(controller)
        if not ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{controller} action server not available")

        traj = JointTrajectory()
        traj.joint_names = list(joints)
        for ti, qi in zip(t + self.approach_s, q):
            p = JointTrajectoryPoint()
            p.positions = qi.tolist()
            p.time_from_start = Duration(sec=int(ti), nanosec=int((ti % 1) * 1e9))
            traj.points.append(p)

        goal = FollowJointTrajectory.Goal(trajectory=traj)
        send = ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if not handle.accepted:
            raise RuntimeError("trajectory rejected by controller")
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"controller error {code}: {res.result().result.error_string}")
        return float(t[-1] + self.approach_s)


def serve(player: DemoPlayer, port: int):
    class Handler(socketserver.StreamRequestHandler):
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
                    dur = player.play(req["joints"], req["t"], req["q"])
                    reply = {"ok": True, "duration": dur}
            except Exception as e:  # report to the client instead of dying
                player.get_logger().error(str(e))
                reply = {"ok": False, "error": str(e)}
            self.wfile.write((json.dumps(reply) + "\n").encode())

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
    args, ros_args = ap.parse_known_args()
    if not args.csv and args.serve is None:
        ap.error("give a CSV to play, or --serve PORT")

    rclpy.init(args=ros_args)
    player = DemoPlayer(args.approach)
    try:
        if args.serve is not None:
            serve(player, args.serve)
        else:
            Demo = _import_therapy()
            demo = Demo.load_csv(args.csv)
            if not args.raw:
                demo = demo.clean()
            demo = demo.adapt(speed=args.speed, size=args.size)
            print(f"playing {demo.name}: {len(demo)} points, {demo.duration:.1f} s + {args.approach:.1f} s approach")
            player.play(demo.joints, demo.t, demo.q)
            print("done")
    except KeyboardInterrupt:
        pass
    finally:
        player.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
