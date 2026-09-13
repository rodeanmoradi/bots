"""Human arm landmarks -> robot arm target points. Shared by the live controller and the recording converter.

Numpy only. Landmarks are arm_xy.py plane coordinates (shoulder, elbow, [wrist,] hand); targets are points in
the robot's arm_base frame (x forward, y left, z up). The human's bone directions are kept and the robot's bone
lengths are used, anchored at the robot's j1 shoulder pivot.

The hand target is where the gripper's fingertips should be. IK solves for the eef frame, which sits a little
short of the fingertips, so `ik_target` moves the target back along the forearm-to-hand direction.

Position-only IK leaves the arm free to reach the hand target with the elbow anywhere. `elbow_seed` fits the
two shoulder joints so the elbow lands on its target, for use as an IK seed, and `pick` chooses between IK
solutions by elbow error and joint change.
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

REACH_MARGIN = 0.98     # keep hand targets this fraction of the arm's full reach: a straight arm is a singularity
FINGERTIP_BEYOND_EEF = 0.014    # the gripper mesh ends this far past the eef frame (measured from the STL meshes)
JUMP_WEIGHT = 0.05      # in `pick`: metres of elbow error one radian of joint change is worth


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rotation(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin_transform(origin):
    T = np.eye(4)
    if origin is None:
        return T
    T[:3, :3] = _rotation(*(float(v) for v in origin.get("rpy", "0 0 0").split()))
    T[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return T


def _axis_angle(axis, angle):
    k = _unit(np.asarray(axis, float))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class RobotModel:
    """Joint names, limits, mimic rules and forward kinematics from the URDF."""

    def __init__(self, urdf_xml):
        root = ET.fromstring(urdf_xml)
        self.parent = {}        # child link -> (parent link, joint origin transform)
        self.joint_child = {}   # joint name -> child link
        self.child_joint = {}   # child link -> (joint name, type, axis)
        self.limits = {}        # joint name -> (lower, upper)
        self.movable = []
        self.mimic = {}
        for joint in root.findall("joint"):
            name, kind = joint.get("name"), joint.get("type")
            child = joint.find("child").get("link")
            self.parent[child] = (joint.find("parent").get("link"), _origin_transform(joint.find("origin")))
            self.joint_child[name] = child
            axis = joint.find("axis")
            self.child_joint[child] = (name, kind, np.array([float(v) for v in axis.get("xyz").split()])
                                       if axis is not None else np.array([1.0, 0.0, 0.0]))
            if kind != "fixed":
                self.movable.append(name)
                limit = joint.find("limit")
                if limit is not None:
                    self.limits[name] = (float(limit.get("lower", "-inf")), float(limit.get("upper", "inf")))
                mimic = joint.find("mimic")
                if mimic is not None:
                    self.mimic[name] = (mimic.get("joint"),
                                        float(mimic.get("multiplier", 1.0)),
                                        float(mimic.get("offset", 0.0)))

    def position(self, link, q=None, ref=BASE_FRAME):
        """Origin of `link` in `ref` for joint positions q (default: all zero)."""
        q = q or {}
        T = np.eye(4)
        while link != ref:
            parent, T_origin = self.parent[link]
            name, kind, axis = self.child_joint[link]
            value = q.get(name, 0.0)
            T_motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                T_motion[:3, :3] = _axis_angle(axis, value)
            elif kind == "prismatic":
                T_motion[:3, 3] = axis * value
            T = T_origin @ T_motion @ T
            link = parent
        return T[:3, 3]


class Retargeter:
    """Maps one arm's landmarks onto the robot arm on the same side."""

    def __init__(self, model, side, plane, mirrored):
        self.model = model
        self.arm = ARMS[side]
        axes = PLANE_AXES[plane]
        if mirrored:
            # A mirrored camera feed makes arm_xy's forward and sideways axes point backwards/right.
            axes = tuple(axis * np.array([-1.0, -1.0, 1.0]) for axis in axes)
        self.axes = axes
        prefix = self.arm["prefix"]
        self.joints = [f"{prefix}{i}" for i in range(7)]
        self.shoulder_joints = (f"{prefix}1", f"{prefix}2")   # the only joints that move the elbow
        self.elbow_link = model.joint_child[f"{prefix}3"]
        self.shoulder = model.position(model.joint_child[f"{prefix}1"])
        elbow = model.position(self.elbow_link)
        wrist = model.position(model.joint_child[f"{prefix}5"])
        eef = model.position(self.arm["eef"])
        # The arm hangs straight at the zero pose, so shoulder -> eef there is its full reach (checked against a
        # random search over the joint limits). Joint origins aren't all on the arm's centre line, so the
        # straight-line bone lengths add up to slightly more than that: scale them down to match, or a straight
        # human arm would put the hand target out of reach.
        self.reach = float(np.linalg.norm(eef - self.shoulder)) + FINGERTIP_BEYOND_EEF
        chords = np.array([np.linalg.norm(elbow - self.shoulder), np.linalg.norm(wrist - elbow),
                           np.linalg.norm(eef - wrist) + FINGERTIP_BEYOND_EEF])
        chords *= self.reach / chords.sum()
        self.upper_len, self.forearm_len, self.hand_len = (float(v) for v in chords)
        self.lower_len = self.forearm_len + self.hand_len

    def targets(self, points):
        """3 or 4 (x, y) landmark points -> [elbow, (wrist,) fingertip] targets in arm_base."""
        ax, ay = self.axes
        human = [x * ax + y * ay for x, y in points]
        if len(human) == 4:
            lengths = (self.upper_len, self.forearm_len, self.hand_len)
        else:
            # Older recordings have no wrist point: one segment from elbow to fingertip.
            lengths = (self.upper_len, self.lower_len)
        targets, joint = [], self.shoulder
        for start, end, length in zip(human, human[1:], lengths):
            joint = joint + _unit(end - start) * length
            targets.append(joint)
        # keep the hand target inside the reachable sphere
        offset = targets[-1] - self.shoulder
        limit = REACH_MARGIN * self.reach
        if np.linalg.norm(offset) > limit:
            targets[-1] = self.shoulder + _unit(offset) * limit
        return targets

    @staticmethod
    def ik_target(targets):
        """The point to send to IK for the eef frame: the fingertip target moved back to where the eef sits."""
        tip = targets[-1]
        return tip - _unit(tip - targets[-2]) * FINGERTIP_BEYOND_EEF

    def elbow_error(self, q, elbow_target):
        return float(np.linalg.norm(self.model.position(self.elbow_link, q) - elbow_target))

    def elbow_seed(self, q, elbow_target, iterations=20, damping=1e-3):
        """A copy of joint positions q with the two shoulder joints fitted so the elbow reaches elbow_target
        (damped least squares), for seeding IK. The other joints are unchanged."""
        seed = dict(q)
        names = self.shoulder_joints
        lower = np.array([self.model.limits.get(n, (-np.pi, np.pi))[0] for n in names])
        upper = np.array([self.model.limits.get(n, (-np.pi, np.pi))[1] for n in names])
        x = np.array([seed.get(n, 0.0) for n in names], float)
        for _ in range(iterations):
            seed.update(zip(names, x))
            p = self.model.position(self.elbow_link, seed)
            J = np.zeros((3, 2))
            for k, name in enumerate(names):
                nudged = dict(seed)
                nudged[name] += 1e-5
                J[:, k] = (self.model.position(self.elbow_link, nudged) - p) / 1e-5
            step = -np.linalg.solve(J.T @ J + damping * np.eye(2), J.T @ (p - elbow_target))
            x = np.clip(x + step, lower, upper)
            if np.linalg.norm(step) < 1e-6:
                break
        seed.update(zip(names, x))
        return seed

    def pick(self, candidates, previous, elbow_target):
        """The IK solution (dicts of joint positions) with the best elbow error plus a penalty for the largest
        joint change from `previous`; None if there are none."""
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None

        def cost(q):
            jump = max(abs(q.get(n, 0.0) - previous.get(n, 0.0)) for n in self.joints)
            return self.elbow_error(q, elbow_target) + JUMP_WEIGHT * jump
        return min(candidates, key=cost)
