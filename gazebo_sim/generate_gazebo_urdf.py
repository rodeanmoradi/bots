#!/usr/bin/env python3
"""Generate a Gazebo-ready URDF from chopped_urdf_v2's source URDF.

The source URDF (chopped_urdf_v2/urdf/chopped_urdf_v2.urdf) has visual +
inertial geometry but no <collision> elements, no fixed-to-world anchor, and
no ros2_control block — fine for RViz/Rerun, not enough for physics sim.

This script is the one source of truth for the delta; re-run it any time the
source URDF changes instead of hand-editing the generated file.

    python3 gazebo_sim/generate_gazebo_urdf.py

Output: chopped_urdf_v2/urdf/chopped_urdf_v2.gazebo.urdf
  - every <visual> gets a matching <collision> (same origin + mesh)
  - a fixed `world` -> `root` joint anchors the base (base is not drivable:
    the wheel joints in this export are fixed, see README)
  - a <ros2_control> block + <gazebo> gz_ros2_control plugin exposing the 16
    independently-actuated joints (mimic joints excluded) as position
    command / position+velocity state interfaces
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_URDF = ROOT_DIR / "chopped_urdf_v2" / "urdf" / "chopped_urdf_v2.urdf"
OUT_URDF = ROOT_DIR / "chopped_urdf_v2" / "urdf" / "chopped_urdf_v2.gazebo.urdf.xacro"
CONTROLLERS_YAML = ROOT_DIR / "chopped_urdf_v2" / "config" / "controllers.yaml"

XACRO_NS = "http://www.ros.org/wiki/xacro"
ET.register_namespace("xacro", XACRO_NS)

# Independently-actuated joints, grouped the way the controllers.yaml groups
# them. The URDF's <mimic> tags say right_right_gripper/left_right_gripper
# should track their _left_gripper sibling, but gz-sim's default dartsim
# physics engine does not implement mimic constraints (confirmed by a
# smoke-test run: "chosen physics engine does not support mimic constraints,
# so no constraint will be created") — so both finger joints are commanded
# directly here instead of relying on the constraint.
RIGHT_ARM_JOINTS = [
    "rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6",
    "right_left_gripper", "right_right_gripper",
]
LEFT_ARM_JOINTS = [
    "lj0", "lj1", "lj2", "lj3", "lj4", "lj5", "lj6",
    "left_left_gripper", "left_right_gripper",
]
ACTUATED_JOINTS = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS


def strip_mimic(root: ET.Element) -> int:
    # ros2_control refuses to give a command interface to a joint that still
    # carries <mimic> ("Activated mimic joints cannot have command
    # interfaces") — and gz-sim's dartsim doesn't implement the mimic
    # constraint anyway (see ACTUATED_JOINTS comment above), so there's
    # nothing for the tag to do here except block loading.
    count = 0
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None and joint.get("name") in ACTUATED_JOINTS:
            joint.remove(mimic)
            count += 1
    return count


def add_collisions(root: ET.Element) -> int:
    count = 0
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            collision = ET.Element("collision")
            origin = visual.find("origin")
            if origin is not None:
                collision.append(copy.deepcopy(origin))
            geometry = visual.find("geometry")
            if geometry is not None:
                collision.append(copy.deepcopy(geometry))
            link.append(collision)
            count += 1
    return count


def anchor_to_world(root: ET.Element) -> None:
    world_link = ET.SubElement(root, "link", {"name": "world"})
    joint = ET.SubElement(root, "joint", {"name": "world_to_root", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "world"})
    ET.SubElement(joint, "child", {"link": "root"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    # Move it to the front so `world` reads as the root link.
    root.remove(world_link)
    root.insert(0, world_link)


def add_ros2_control(root: ET.Element) -> None:
    xacro_arg = ET.Element(f"{{{XACRO_NS}}}arg", {"name": "controllers_config", "default": "config/controllers.yaml"})
    root.insert(0, xacro_arg)

    ros2_control = ET.SubElement(root, "ros2_control", {"name": "ChoppedBotSystem", "type": "system"})
    hardware = ET.SubElement(ros2_control, "hardware")
    ET.SubElement(hardware, "plugin").text = "gz_ros2_control/GazeboSimSystem"
    for joint_name in ACTUATED_JOINTS:
        joint_el = ET.SubElement(ros2_control, "joint", {"name": joint_name})
        ET.SubElement(joint_el, "command_interface", {"name": "position"})
        state_pos = ET.SubElement(joint_el, "state_interface", {"name": "position"})
        state_vel = ET.SubElement(joint_el, "state_interface", {"name": "velocity"})
        del state_pos, state_vel

    gazebo = ET.SubElement(root, "gazebo")
    plugin = ET.SubElement(
        gazebo,
        "plugin",
        {
            "filename": "gz_ros2_control-system",
            "name": "gz_ros2_control::GazeboSimROS2ControlPlugin",
        },
    )
    ET.SubElement(plugin, "parameters").text = "$(arg controllers_config)"


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "    "
        for child in elem:
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "    "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def write_controllers_yaml() -> None:
    # Each controller's own params must be a *top-level* key (a sibling of
    # controller_manager, matching its node name) — nesting them inside
    # controller_manager.ros__parameters creates a second "right_arm_controller"
    # key in that same mapping (the first holds `type:`), which is a duplicate
    # YAML key. yaml.safe_load silently keeps only the last one, so the `type:`
    # line vanishes and — since the top-level entry moved the joints list one
    # level too deep — controller_manager ends up seeing an empty joints list.
    def joint_group(name: str, joints: list[str]) -> str:
        joints_yaml = "\n".join(f"    - {j}" for j in joints)
        return (
            f"{name}:\n"
            f"  ros__parameters:\n"
            f"    joints:\n{joints_yaml}\n"
            f"    command_interfaces:\n"
            f"      - position\n"
            f"    state_interfaces:\n"
            f"      - position\n"
            f"      - velocity\n"
        )

    content = (
        "controller_manager:\n"
        "  ros__parameters:\n"
        "    update_rate: 100  # Hz\n"
        "\n"
        "    joint_state_broadcaster:\n"
        "      type: joint_state_broadcaster/JointStateBroadcaster\n"
        "\n"
        "    right_arm_controller:\n"
        "      type: joint_trajectory_controller/JointTrajectoryController\n"
        "\n"
        "    left_arm_controller:\n"
        "      type: joint_trajectory_controller/JointTrajectoryController\n"
        "\n"
        + joint_group("right_arm_controller", RIGHT_ARM_JOINTS)
        + "\n"
        + joint_group("left_arm_controller", LEFT_ARM_JOINTS)
    )
    CONTROLLERS_YAML.parent.mkdir(parents=True, exist_ok=True)
    CONTROLLERS_YAML.write_text(content)
    print(f"wrote {CONTROLLERS_YAML.relative_to(ROOT_DIR)}")


def main() -> None:
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()

    n_mimic = strip_mimic(root)
    n_collisions = add_collisions(root)
    anchor_to_world(root)
    add_ros2_control(root)

    indent(root)
    tree.write(OUT_URDF, encoding="unicode", xml_declaration=False)
    OUT_URDF.write_text('<?xml version="1.0"?>\n' + OUT_URDF.read_text())

    print(
        f"wrote {OUT_URDF.relative_to(ROOT_DIR)} "
        f"({n_collisions} collisions added, {n_mimic} mimic tags stripped)"
    )
    write_controllers_yaml()


if __name__ == "__main__":
    main()
