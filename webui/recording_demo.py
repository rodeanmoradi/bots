"""Turn a recorded arm movement (local/recordings/*.json) into a robot demonstration, via MoveIt IK.

Uses the live mirror's retargeting (therapy/retarget.py) and MoveIt's /compute_ik, so move_group must be
running, e.g. the simulated robot the web interface starts for coaching. Each pose is solved starting from the
previous solution, so the arm moves continuously; joint limits (such as the one-way elbow) apply.
"""

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from therapy.demo import Demo
from therapy.retarget import BASE_FRAME, Retargeter, RobotModel

RATE_HZ = 20.0      # IK poses per second of recording; also the assumed frame rate of takes saved without timestamps
IK_TIMEOUT = Duration(sec=0, nanosec=50_000_000)


def load_recording(path):
    """-> (metadata, (N, 3 or 4, 2) frames, (N,) seconds from the first frame)."""
    meta = json.loads(Path(path).read_text())
    frames = np.asarray(meta["frames"], dtype=float)
    t = np.asarray(meta["t"], dtype=float) if "t" in meta else np.arange(len(frames)) / RATE_HZ
    return meta, frames, t - t[0]


def _robot_description(node, timeout=10.0):
    received = []
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(String, "/robot_description", lambda msg: received.append(msg.data), latched)
    try:
        deadline = time.monotonic() + timeout
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
    if not received:
        raise RuntimeError("no /robot_description: is the simulated robot running?")
    return received[0]


def _solve(node, client, retarget, positions, target):
    request = GetPositionIK.Request()
    ik = request.ik_request
    ik.group_name = retarget.arm["group"]
    ik.ik_link_name = retarget.arm["eef"]
    ik.avoid_collisions = False
    ik.timeout = IK_TIMEOUT
    ik.robot_state.joint_state = JointState(name=list(positions), position=[float(v) for v in positions.values()])
    ik.pose_stamped.header.frame_id = BASE_FRAME
    ik.pose_stamped.pose.position = Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))
    ik.pose_stamped.pose.orientation.w = 1.0
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
    response = future.result()
    if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
        return False
    positions.update(zip(response.solution.joint_state.name, response.solution.joint_state.position))
    return True


def recording_to_demo(path, node, smoothing=0.5, on_progress=None):
    """Recording -> Demo of the robot arm on the recording's side, sampled at RATE_HZ.

    on_progress(done, total, failed) is called after every pose; an unreachable pose holds the previous one.
    """
    meta, frames, t = load_recording(path)
    client = node.create_client(GetPositionIK, "/compute_ik")
    try:
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("MoveIt's /compute_ik isn't available: coaching from a recording needs the "
                               "simulated robot")
        model = RobotModel(_robot_description(node))
        retarget = Retargeter(model, meta["side"], meta["plane"], bool(meta.get("mirrored", False)))
        positions = {name: 0.0 for name in model.movable}

        keep, last = [], -np.inf
        for i, ti in enumerate(t):
            if ti - last >= 1.0 / RATE_HZ - 1e-6:
                keep.append(i)
                last = ti

        rows, times, failed, target = [], [], 0, None
        for done, i in enumerate(keep, 1):
            tip = retarget.targets(frames[i])[-1]
            target = tip if target is None or smoothing <= 0 else smoothing * target + (1 - smoothing) * tip
            if not _solve(node, client, retarget, positions, target):
                failed += 1
            rows.append([positions[j] for j in retarget.joints])
            times.append(t[i])
            if on_progress:
                on_progress(done, len(keep), failed)
    finally:
        node.destroy_client(client)
    return Demo(retarget.joints, times, rows, name=Path(path).stem,
                meta={"source": str(path), "ik_failed": failed, "ik_total": len(keep)})
