#!/usr/bin/env python3
"""Live arm tracking: webcam -> arm_xy() -> /mediapipe/arm_landmarks.

    ros2 launch bracketbot_moveit_config mediapipe_live.launch.py

Publishes the same PoseArray as main_controller.py's RecordingPlayer (shoulder,
elbow, wrist, hand; x/y from arm_xy, z = 0), plus /mediapipe/claw_open when claw:=true.
"""

import time

import cv2
import mediapipe as mp
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Bool

import arm_xy

LANDMARKS_TOPIC = "/mediapipe/arm_landmarks"
CLAW_TOPIC = "/mediapipe/claw_open"
# Launch arguments arrive as "0" or "/dev/video0"; accept either.
DYNAMIC = ParameterDescriptor(dynamic_typing=True)


class ArmXYNode(Node):
    def __init__(self):
        super().__init__("arm_xy")
        self.declare_parameter("camera", "0", DYNAMIC)
        self.declare_parameter("side", "right")
        self.declare_parameter("plane", "side")
        self.declare_parameter("flip", False)
        self.declare_parameter("claw", False)
        self.declare_parameter("preview", True)
        self.declare_parameter("width", 1280, DYNAMIC)
        self.declare_parameter("height", 720, DYNAMIC)
        param = lambda name: self.get_parameter(name).value
        self._side = param("side")
        self._plane = param("plane")
        self._flip = bool(param("flip"))
        self._claw = bool(param("claw"))
        self._preview = bool(param("preview"))

        self._cap = arm_xy.open_camera(str(param("camera")), int(param("width")), int(param("height")))
        self._landmarks_pub = self.create_publisher(PoseArray, LANDMARKS_TOPIC, 10)
        self._claw_pub = self.create_publisher(Bool, CLAW_TOPIC, 10)
        self._frames = self._tracked = 0
        self._stats_since = time.monotonic()
        self.create_timer(2.0, self._log_stats)
        self.get_logger().info(f"camera {param('camera')}: side={self._side}, plane={self._plane}, "
                               f"flip={self._flip}, claw={self._claw}")

    def step(self):
        """Process one camera frame. Returns False when the preview window asks to quit."""
        ok, frame = self._cap.read()
        if not ok:
            return True
        if self._flip:
            frame = cv2.flip(frame, 1)
        result = arm_xy.arm_xy(frame, side=self._side, plane=self._plane, with_claw=self._claw)
        pts, claw = result if self._claw else (result, None)
        self._frames += 1

        if pts is not None:
            self._tracked += 1
            msg = PoseArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"mediapipe_{self._plane}"
            msg.poses = [Pose(position=Point(x=float(x), y=float(y))) for x, y in pts]
            self._landmarks_pub.publish(msg)
        if claw is not None:
            self._claw_pub.publish(Bool(data=bool(claw)))

        if self._preview:
            if arm_xy.last_pose is not None and arm_xy.last_pose.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, arm_xy.last_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            cv2.imshow("arm_xy (Esc to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False
        return True

    def _log_stats(self):
        elapsed = time.monotonic() - self._stats_since
        self.get_logger().info(f"{self._frames / elapsed:.1f} fps, "
                               f"arm tracked in {self._tracked}/{self._frames} frames")
        self._frames = self._tracked = 0
        self._stats_since = time.monotonic()

    def close(self):
        self._cap.release()
        cv2.destroyAllWindows()
        arm_xy.reset()


def main():
    rclpy.init()
    node = ArmXYNode()
    try:
        # The camera read blocks, so it drives the loop; spin_once just services the stats timer.
        while rclpy.ok() and node.step():
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
