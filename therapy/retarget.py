"""Human arm landmarks -> robot arm target points. Shared by the live controller and the recording converter.

Numpy only. Landmarks are arm_xy.py plane coordinates (shoulder, elbow, [wrist,] hand); targets are points in
the robot's arm_base frame (x forward, y left, z up). The human's bone directions are kept and the robot's bone
lengths are used, anchored at the robot's j1 shoulder pivot.
"""

import xml.etree.ElementTree as ET

import numpy as np

BASE_FRAME = "arm_base"

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


class Retargeter:
    """Maps one arm's landmarks onto the robot arm on the same side."""

    def __init__(self, model, side, plane, mirrored):
        self.arm = ARMS[side]
        axes = PLANE_AXES[plane]
        if mirrored:
            # A mirrored camera feed makes arm_xy's forward and sideways axes point backwards/right.
            axes = tuple(axis * np.array([-1.0, -1.0, 1.0]) for axis in axes)
        self.axes = axes
        prefix = self.arm["prefix"]
        self.joints = [f"{prefix}{i}" for i in range(7)]
        self.shoulder = model.position(model.joint_child[f"{prefix}1"])
        elbow = model.position(model.joint_child[f"{prefix}3"])
        wrist = model.position(model.joint_child[f"{prefix}5"])
        eef = model.position(self.arm["eef"])
        self.upper_len = float(np.linalg.norm(elbow - self.shoulder))
        self.forearm_len = float(np.linalg.norm(wrist - elbow))
        self.hand_len = float(np.linalg.norm(eef - wrist))
        self.lower_len = float(np.linalg.norm(eef - elbow))

    def targets(self, points):
        """3 or 4 (x, y) landmark points -> [elbow, (wrist,) gripper tip] targets in arm_base; the last is for IK."""
        ax, ay = self.axes
        human = [x * ax + y * ay for x, y in points]
        if len(human) == 4:
            lengths = (self.upper_len, self.forearm_len, self.hand_len)
        else:
            # Older recordings have no wrist point: one segment from elbow to gripper tip.
            lengths = (self.upper_len, self.lower_len)
        targets, joint = [], self.shoulder
        for start, end, length in zip(human, human[1:], lengths):
            joint = joint + _unit(end - start) * length
            targets.append(joint)
        return targets
