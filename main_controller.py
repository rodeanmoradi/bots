#!/usr/bin/env python3
"""Drive the robot arm from MediaPipe arm landmarks through MoveIt's IK.

    ros2 launch bracketbot_moveit_config mediapipe_replay.launch.py   # source:=recording
    ros2 launch bracketbot_moveit_config mediapipe_live.launch.py     # source:=live

Landmarks (shoulder, elbow, wrist, hand in arm_xy.py plane coordinates; older
recordings have no wrist) arrive on /mediapipe/arm_landmarks, from
RecordingPlayer here or MediaPipe/arm_xy_node.py. ArmController retargets them
onto the robot's arm, asks move_group's /compute_ik for joint angles and
publishes /joint_states.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseArray
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

REPO_ROOT = Path(__file__).resolve().parent
LANDMARKS_TOPIC = "/mediapipe/arm_landmarks"
TARGETS_TOPIC = "/mediapipe/arm_targets"
BASE_FRAME = "arm_base"
IK_TIMEOUT = Duration(sec=0, nanosec=50_000_000)
LANDMARK_ORDERS = (["shoulder", "elbow", "wrist", "hand"], ["shoulder", "elbow", "hand"])

ARMS = {
    # *_no_mast groups leave rj0/lj0 out of IK, so the j1 anchor never moves.
    "right": {"group": "right_arm_no_mast", "eef": "right_eef", "prefix": "rj"},
    "left": {"group": "left_arm_no_mast", "eef": "left_eef", "prefix": "lj"},
}

# arm_xy.py plane axes (x, y) as directions in arm_base (x forward, y left, z up).
PLANE_AXES = {
    "side": (np.array([1.0, 0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "front": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "top": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
}

# Launch arguments arrive as "15" or "15.0"; accept either.
DYNAMIC = ParameterDescriptor(dynamic_typing=True)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _origin_transform(origin):
    T = np.eye(4)
    if origin is None:
        return T
    r, p, y = (float(v) for v in origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return T


class RobotModel:
    """Joint names, mimic rules and zero-pose link positions from the URDF."""

    def __init__(self, urdf_xml):
        root = ET.fromstring(urdf_xml)
        self.parent = {}
        self.joint_child = {}
        self.movable = []
        self.mimic = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            child = joint.find("child").get("link")
            self.parent[child] = (joint.find("parent").get("link"),
                                  _origin_transform(joint.find("origin")))
            self.joint_child[name] = child
            if joint.get("type") != "fixed":
                self.movable.append(name)
                mimic = joint.find("mimic")
                if mimic is not None:
                    self.mimic[name] = (mimic.get("joint"),
                                        float(mimic.get("multiplier", 1.0)),
                                        float(mimic.get("offset", 0.0)))

    def position(self, link, ref=BASE_FRAME):
        T = np.eye(4)
        while link != ref:
            parent, T_joint = self.parent[link]
            T = T_joint @ T
            link = parent
        return T[:3, 3]


class ArmController(Node):
    """Retargets landmarks onto the robot arm, solves IK, publishes /joint_states."""

    def __init__(self):
        super().__init__("arm_controller")
        self.declare_parameter("source", "recording")
        self.declare_parameter("side", "right")
        self.declare_parameter("plane", "side")
        self.declare_parameter("smoothing", 0.5, DYNAMIC)
        self.declare_parameter("publish_rate_hz", 30.0, DYNAMIC)
        self.declare_parameter("mirrored", True)
        self._model = None
        self._positions = {}
        self._target = None
        self._queued = None
        self._ik_pending = False
        self._ik_ok = True
        self._stats = [0, 0]
        self.set_arm(self.get_parameter("side").value, self.get_parameter("plane").value)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._on_robot_description, latched)
        self.create_subscription(PoseArray, LANDMARKS_TOPIC, self._on_landmarks, 10)
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._marker_pub = self.create_publisher(MarkerArray, TARGETS_TOPIC, 10)
        self._ik = self.create_client(GetPositionIK, "/compute_ik")
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate, self._publish_joint_states)
        self.create_timer(2.0, self._log_stats)

    def set_arm(self, side, plane):
        """Pick the robot arm and landmark plane. Call before spinning."""
        self._arm = ARMS[side]
        self._axes = PLANE_AXES[plane]
        if self.get_parameter("mirrored").value:
            # A mirrored camera feed makes arm_xy's forward and sideways axes point backwards/right.
            self._axes = tuple(axis * np.array([-1.0, -1.0, 1.0]) for axis in self._axes)

    def is_ready(self):
        return self._model is not None and self._ik.service_is_ready()

    def _on_robot_description(self, msg):
        if self._model is not None:
            return
        self._model = RobotModel(msg.data)
        self._positions = {name: 0.0 for name in self._model.movable}
        prefix = self._arm["prefix"]
        self._shoulder = self._model.position(self._model.joint_child[f"{prefix}1"])
        elbow = self._model.position(self._model.joint_child[f"{prefix}3"])
        wrist = self._model.position(self._model.joint_child[f"{prefix}5"])
        eef = self._model.position(self._arm["eef"])
        self._upper_len = float(np.linalg.norm(elbow - self._shoulder))
        self._forearm_len = float(np.linalg.norm(wrist - elbow))
        self._hand_len = float(np.linalg.norm(eef - wrist))
        self._lower_len = float(np.linalg.norm(eef - elbow))
        self.get_logger().info(
            f"{self._arm['group']}: shoulder {np.round(self._shoulder, 3)} in {BASE_FRAME}, "
            f"upper arm {self._upper_len:.3f} m, forearm {self._forearm_len:.3f} m, "
            f"wrist to gripper tip {self._hand_len:.3f} m")

    def _on_landmarks(self, msg):
        if self._model is None or len(msg.poses) not in (3, 4):
            return
        ax, ay = self._axes
        human = [p.position.x * ax + p.position.y * ay for p in msg.poses]
        if len(human) == 4:
            lengths = (self._upper_len, self._forearm_len, self._hand_len)
        else:
            # Older recordings have no wrist point: one segment from elbow to gripper tip.
            lengths = (self._upper_len, self._lower_len)
        # Keep the human's bone directions, use the robot's bone lengths.
        targets = []
        joint = self._shoulder
        for start, end, length in zip(human, human[1:], lengths):
            joint = joint + _unit(end - start) * length
            targets.append(joint)
        alpha = float(self.get_parameter("smoothing").value)
        if self._target is not None and len(self._target) == len(targets) and alpha > 0.0:
            targets = [alpha * old + (1.0 - alpha) * new for old, new in zip(self._target, targets)]
        self._target = targets
        self._queued = targets[-1]
        self._send_ik()
        self._publish_markers()

    def _send_ik(self):
        # One request in flight at a time; frames arriving meanwhile collapse to the newest.
        if self._ik_pending or self._queued is None or not self._ik.service_is_ready():
            return
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self._arm["group"]
        ik.ik_link_name = self._arm["eef"]
        ik.avoid_collisions = False
        ik.timeout = IK_TIMEOUT
        ik.robot_state.joint_state = self._joint_state_msg()
        ik.pose_stamped.header.frame_id = BASE_FRAME
        x, y, z = (float(v) for v in self._queued)
        ik.pose_stamped.pose.position = Point(x=x, y=y, z=z)
        ik.pose_stamped.pose.orientation.w = 1.0
        self._queued = None
        self._ik_pending = True
        self._ik.call_async(request).add_done_callback(self._on_ik_result)

    def _on_ik_result(self, future):
        self._ik_pending = False
        response = future.result()
        self._ik_ok = response is not None and response.error_code.val == MoveItErrorCodes.SUCCESS
        if self._ik_ok:
            solution = response.solution.joint_state
            self._positions.update(zip(solution.name, solution.position))
        self._stats[0 if self._ik_ok else 1] += 1
        self._send_ik()

    def _joint_state_msg(self):
        for follower, (leader, multiplier, offset) in self._model.mimic.items():
            self._positions[follower] = multiplier * self._positions[leader] + offset
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._positions)
        msg.position = [float(v) for v in self._positions.values()]
        return msg

    def _publish_joint_states(self):
        if self._model is not None:
            self._joint_pub.publish(self._joint_state_msg())

    def _publish_markers(self):
        points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                  for p in [self._shoulder, *self._target]]
        bones = Marker(type=Marker.LINE_STRIP, ns="retarget", id=0, points=points)
        bones.scale.x = 0.01
        bones.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)
        joints = Marker(type=Marker.SPHERE_LIST, ns="retarget", id=1, points=points)
        joints.scale.x = joints.scale.y = joints.scale.z = 0.035
        grey = ColorRGBA(r=0.8, g=0.8, b=0.8, a=1.0)
        status = (ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0) if self._ik_ok
                  else ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0))
        joints.colors = [grey] * (len(points) - 1) + [status]
        for marker in (bones, joints):
            marker.header.frame_id = BASE_FRAME
        self._marker_pub.publish(MarkerArray(markers=[bones, joints]))

    def _log_stats(self):
        ok, failed = self._stats
        if ok or failed:
            self.get_logger().info(f"IK: {ok} solved, {failed} failed (last 2 s)")
        self._stats = [0, 0]


class RecordingPlayer(Node):
    """Replays a MediaPipe/run_arm_xy.py recording as landmark messages."""

    def __init__(self):
        super().__init__("recording_player")
        self.declare_parameter("recording", "latest")
        self.declare_parameter("rate_hz", 20.0, DYNAMIC)
        self.declare_parameter("loop", True)
        path = self._resolve(self.get_parameter("recording").value)
        meta = json.loads(path.read_text())
        if meta["order"] not in LANDMARK_ORDERS:
            raise ValueError(f"{path}: unexpected landmark order {meta['order']}")
        self.side, self.plane = meta["side"], meta["plane"]
        self.ready = None
        self._frames = np.asarray(meta["frames"], dtype=float)
        self._index = 0
        self._loop = bool(self.get_parameter("loop").value)
        self._pub = self.create_publisher(PoseArray, LANDMARKS_TOPIC, 10)
        rate = float(self.get_parameter("rate_hz").value)
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"{path.name}: {len(self._frames)} frames, "
                               f"side={self.side}, plane={self.plane}, {rate:g} Hz")

    @staticmethod
    def _resolve(spec):
        if spec == "latest":
            recordings = sorted((REPO_ROOT / "trajectories").glob("*.json"),
                                key=lambda p: p.stem.rsplit("_", 1)[-1])
            if not recordings:
                raise FileNotFoundError(f"no recordings in {REPO_ROOT / 'trajectories'}")
            return recordings[-1]
        path = Path(spec).expanduser()
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / path
        return path.with_suffix(".json")

    def _tick(self):
        if self.ready is not None and not self.ready():
            self.get_logger().info("waiting for /robot_description and move_group's /compute_ik",
                                   throttle_duration_sec=5.0)
            return
        if self._index == len(self._frames):
            if not self._loop:
                self._timer.cancel()
                self.get_logger().info("recording finished")
                return
            self._index = 0
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"mediapipe_{self.plane}"
        msg.poses = [Pose(position=Point(x=float(x), y=float(y)))
                     for x, y in self._frames[self._index]]
        self._pub.publish(msg)
        self._index += 1


def main():
    rclpy.init(args=sys.argv)
    controller = ArmController()
    nodes = [controller]
    source = controller.get_parameter("source").value
    if source == "recording":
        player = RecordingPlayer()
        controller.set_arm(player.side, player.plane)
        player.ready = controller.is_ready
        nodes.append(player)
    elif source != "live":
        raise ValueError(f"source must be 'recording' or 'live', got {source!r}")
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
