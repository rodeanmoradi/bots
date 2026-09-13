"""Robot joint angles -> hand path, via forward kinematics on the URDF.

Only needs numpy and the URDF file; no ROS. The robot's `root` frame is
x-forward, y-left, z-up (checked numerically: the shoulders sit at y = ∓0.027
and the arm hangs straight down at the zero pose), so its body frame is the
identity and the shoulder link is the analogue of the human shoulder.

    python -m therapy.robot_path reach_01.csv          # print the hand path summary
    python -m therapy.robot_path reach_01.csv --plot   # and draw it
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np

from .demo import Demo
from .motion import Motion, elbow_angle_deg

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "chopped_urdf_v2" / "urdf" / "chopped_urdf_v2.urdf"

ARM_LINKS = {
    "right": {"shoulder": "shoulder__shoulder", "elbow": "forearm__forearm", "hand": "right_eef"},
    "left": {"shoulder": "l_shoulder__shoulder", "elbow": "l_forearm__forearm", "hand": "left_eef"},
}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rodrigues(axis, angle):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class RobotKinematics:
    """Forward kinematics for any link of a URDF tree."""

    def __init__(self, urdf_path=DEFAULT_URDF):
        root = ET.parse(str(urdf_path)).getroot()
        self.joint_to_child: Dict[str, dict] = {}   # child link -> joint spec
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])
            self.joint_to_child[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "xyz": xyz, "R": _rpy(*rpy), "axis": axis,
            }
        self._chains: Dict[str, list] = {}

    def chain(self, link: str) -> list:
        if link not in self._chains:
            c, cur = [], link
            while cur in self.joint_to_child:
                c.append(self.joint_to_child[cur])
                cur = self.joint_to_child[cur]["parent"]
            self._chains[link] = c[::-1]
        return self._chains[link]

    def position(self, link: str, q: Dict[str, float]) -> np.ndarray:
        """Position of `link`'s origin in the root frame for joint values q."""
        T = np.eye(4)
        for j in self.chain(link):
            A = np.eye(4)
            A[:3, 3] = j["xyz"]
            v = q.get(j["name"], 0.0)
            if j["type"] == "revolute" or j["type"] == "continuous":
                A[:3, :3] = j["R"] @ _rodrigues(j["axis"], v)
            elif j["type"] == "prismatic":
                A[:3, :3] = j["R"]
                A[:3, 3] = j["xyz"] + j["R"] @ (j["axis"] * v)
            else:
                A[:3, :3] = j["R"]
            T = T @ A
        return T[:3, 3]

    def arm_length(self, arm: str) -> float:
        """Shoulder-to-hand distance at the zero pose, where the arm hangs straight."""
        L = ARM_LINKS[arm]
        return float(np.linalg.norm(self.position(L["hand"], {}) - self.position(L["shoulder"], {})))

    def hand_path(self, demo: Demo, arm: str = None) -> Motion:
        """Run the demonstration through FK -> Motion in the shared representation."""
        arm = arm or demo.arm
        L = ARM_LINKS[arm]
        length = self.arm_length(arm)
        hand = np.zeros((len(demo), 3))
        elbow = np.zeros(len(demo))
        for i, row in enumerate(demo.q):
            q = dict(zip(demo.joints, row))
            s = self.position(L["shoulder"], q)
            e = self.position(L["elbow"], q)
            h = self.position(L["hand"], q)
            v = h - s                          # root frame: x forward, y left, z up
            outward = -v[1] if arm == "right" else v[1]
            hand[i] = [v[0], outward, v[2]]
            elbow[i] = elbow_angle_deg(s, e, h)
        return Motion(demo.t, hand / length, elbow, source="robot",
                      meta={"arm": arm, "arm_length_m": length, "demo": demo.name})


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="demonstration CSV from bag_to_csv.py")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--raw", action="store_true", help="don't clean pauses out of the recording first")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", help="write the Motion as .npz")
    args = ap.parse_args()

    demo = Demo.load_csv(args.csv)
    if not args.raw:
        demo = demo.clean()
    kin = RobotKinematics(args.urdf)
    m = kin.hand_path(demo)
    print(f"arm: {demo.arm}, arm length {m.meta['arm_length_m']:.3f} m, joints {demo.joints}")
    print(m.describe())
    if args.save:
        m.save(args.save)
        print("saved", args.save)
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for k, name in enumerate(("forward", "outward", "up")):
            ax[0].plot(m.t, m.hand[:, k], label=name)
        ax[0].plot(m.t, m.reach, "k--", label="reach")
        ax[0].set_xlabel("s"); ax[0].set_ylabel("arm lengths"); ax[0].legend()
        ax[1].plot(m.hand[:, 1], m.hand[:, 2]); ax[1].set_xlabel("outward"); ax[1].set_ylabel("up")
        ax[1].set_aspect("equal"); ax[1].set_title("hand path (front view)")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
