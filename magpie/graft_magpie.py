#!/usr/bin/env python3
"""
Build magpie/h1_2_magpie.urdf:
  - H1 body taken verbatim from the working handless URDF.
  - Each magpie gripper collapsed into a SINGLE rigid link, fixed-attached
    to its wrist_yaw_link. No joints, no actuation, no 4-bar mechanism.
  - Per-gripper mass set to USER_MASS_PER_GRIPPER (default 0.615 kg, measured
    from the physical hardware).
  - Lumped inertia computed by parallel-axis sum of all MJCF magpie body
    inertias, then mass-scaled so the total matches USER_MASS_PER_GRIPPER.
  - All 9 magpie meshes kept as visuals; collision is a single conservative box.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path("/home/niraj/h1_mujoco")
BASE_DIR = Path("/home/niraj/isaac_gym_projects/homie_h12/h1_2")
BASE_URDF = BASE_DIR / "h1_2_handless.urdf"
MJCF = REPO / "magpie" / "h1_2_magpie.xml"
# Sit next to the working handless URDF so the existing meshes/ folder resolves.
DST = BASE_DIR / "h1_2_magpie.urdf"

USER_MASS_PER_GRIPPER = 0.615  # kg, from physical scale measurement
ATTACH_XYZ = "0.07900000 0.00000000 -0.00312500"
ATTACH_RPY = "3.14159265 1.57079613 3.14159265"  # quat (0.7071,0,0.7071,0) → rpy

# Anchor body in MJCF whose subtree is the right / left gripper.
MAGPIE_ANCHORS = {"R": "gripper_attachment", "L": "gripper_attachment_L"}

# Mesh files (placed at link origin with the magpie xyaxes rotation, rpy=(0,0,-π/2))
MAGPIE_MESHES = [
    "mount.stl", "base_bot.stl", "base_top.stl",
    "left_crank.stl", "left_finger_combined.stl", "left_rocker.stl",
    "right_crank.stl", "right_finger_combined.stl", "right_rocker.stl",
]


def fix_h1_mesh_paths(text):
    # No path rewrite needed: the URDF lives next to the existing meshes/ folder.
    return text


def gripper_subtree(model, anchor_name):
    """Return list of body ids in the subtree rooted at anchor_name."""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_name)
    children = {aid}
    while True:
        added = False
        for i in range(model.nbody):
            if i in children:
                continue
            if int(model.body_parentid[i]) in children:
                children.add(i)
                added = True
        if not added:
            break
    return sorted(children)


def lumped_inertial(model, data, anchor_name, target_mass):
    """Sum every magpie body's inertia tensor into the anchor frame, mass-scale to target.

    Returns (com_in_anchor_frame, 3x3 inertia at that com)."""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_name)
    a_pos = data.xpos[aid].copy()
    a_R = data.xmat[aid].reshape(3, 3).copy()

    bids = gripper_subtree(model, anchor_name)

    # Stage 1: total mass + mass-weighted CoM in anchor frame
    M = 0.0
    weighted = np.zeros(3)
    for bid in bids:
        m = float(model.body_mass[bid])
        if m <= 0:
            continue
        com_world = data.xipos[bid]
        com_local = a_R.T @ (com_world - a_pos)
        M += m
        weighted += m * com_local
    com = weighted / M

    # Stage 2: parallel-axis sum of inertia tensors at `com` (in anchor frame)
    I_total = np.zeros((3, 3))
    for bid in bids:
        m = float(model.body_mass[bid])
        if m <= 0:
            continue
        com_world = data.xipos[bid]
        com_local = a_R.T @ (com_world - a_pos)
        # Body's principal-axis frame in world, then transform to anchor frame.
        Rb_world = data.ximat[bid].reshape(3, 3)
        Rb_local = a_R.T @ Rb_world
        I_principal = np.diag(model.body_inertia[bid])
        I_at_body = Rb_local @ I_principal @ Rb_local.T
        # Steiner shift to combined CoM
        r = com_local - com
        I_total += I_at_body + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    # Mass scaling: scale all masses → scales I linearly too
    scale = target_mass / M
    I_total *= scale
    return com, I_total


def fmt_diag(I):
    """Return ixx,iyy,izz from a 3x3 inertia matrix (we keep only diag for simplicity)."""
    return float(I[0, 0]), float(I[1, 1]), float(I[2, 2])


def build_gripper_link(side, anchor_name, com, I, mass):
    suffix = "" if side == "R" else "_L"
    parent = "right_wrist_yaw_link" if side == "R" else "left_wrist_yaw_link"
    link_name = f"magpie_gripper{suffix}"
    joint_name = f"magpie_gripper{suffix}_fixed"

    visuals = []
    for stl in MAGPIE_MESHES:
        visuals.append(
            f'    <visual>\n'
            f'      <origin xyz="0 0 0" rpy="0 0 -1.57079633"/>\n'
            f'      <geometry>\n'
            f'        <mesh filename="meshes/magpie/{stl}" scale="0.001 0.001 0.001"/>\n'
            f'      </geometry>\n'
            f'      <material name="dark_gray"><color rgba="0.1 0.1 0.1 1"/></material>\n'
            f'    </visual>'
        )
    visuals_xml = "\n".join(visuals)

    ixx, iyy, izz = fmt_diag(I)

    link_xml = (
        f'  <link name="{link_name}">\n'
        f'    <inertial>\n'
        f'      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n'
        f'      <mass value="{mass:.6f}"/>\n'
        f'      <inertia ixx="{ixx:.6e}" iyy="{iyy:.6e}" izz="{izz:.6e}"'
        f' ixy="{I[0,1]:.6e}" ixz="{I[0,2]:.6e}" iyz="{I[1,2]:.6e}"/>\n'
        f'    </inertial>\n'
        f'{visuals_xml}\n'
        f'    <collision>\n'
        f'      <origin xyz="0 0 0.09" rpy="0 0 0"/>\n'
        f'      <geometry>\n'
        f'        <box size="0.10 0.13 0.18"/>\n'
        f'      </geometry>\n'
        f'    </collision>\n'
        f'  </link>'
    )
    joint_xml = (
        f'  <joint name="{joint_name}" type="fixed">\n'
        f'    <origin xyz="{ATTACH_XYZ}" rpy="{ATTACH_RPY}"/>\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{link_name}"/>\n'
        f'  </joint>'
    )
    return link_xml + "\n" + joint_xml


H12_MOUNT_BLOCK = """    <visual>
      <origin xyz="0.05400000 0.00000000 0.00000000" rpy="3.14159265 1.57079613 3.14159265"/>
      <geometry>
        <mesh filename="meshes/magpie/magpie_h12.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="dark_gray"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <origin xyz="0.06650000 0.00000000 0.00000000" rpy="3.14159265 1.57079613 3.14159265"/>
      <geometry>
        <cylinder radius="0.03200000" length="0.02500000"/>
      </geometry>
    </collision>
"""


def inject_h12_mount(text):
    out = text
    for side in ("left", "right"):
        marker = f'<link name="{side}_wrist_yaw_link">'
        idx = out.index(marker)
        end = out.index("</link>", idx)
        out = out[:end] + H12_MOUNT_BLOCK + "  " + out[end:]
    return out


def main():
    base_text = fix_h1_mesh_paths(BASE_URDF.read_text())
    base_text = inject_h12_mount(base_text)
    base_root = ET.fromstring(base_text)
    base_links = {L.get("name") for L in base_root.findall("link")}
    for need in ("right_wrist_yaw_link", "left_wrist_yaw_link"):
        assert need in base_links, f"{need} missing from base URDF"

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    blocks = ["", "  <!-- ════ MAGPIE GRIPPERS (lumped rigid bodies, no joints) ════ -->"]
    for side, anchor in MAGPIE_ANCHORS.items():
        com, I = lumped_inertial(model, data, anchor, USER_MASS_PER_GRIPPER)
        blocks.append(build_gripper_link(side, anchor, com, I, USER_MASS_PER_GRIPPER))
        print(f"  {side} gripper:  CoM in attach frame = ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}) m")
        print(f"             diag inertia = ({I[0,0]:.3e}, {I[1,1]:.3e}, {I[2,2]:.3e}) kg·m²")

    insert = "\n".join(blocks) + "\n"
    out = base_text.rstrip()
    assert out.endswith("</robot>")
    out = out[: -len("</robot>")] + insert + "</robot>\n"
    DST.write_text(out)

    print(f"\nWrote {DST}")
    print(f"  Per-gripper mass: {USER_MASS_PER_GRIPPER:.3f} kg (matches measured 615 g)")
    print(f"  Bimanual magpie payload: {2*USER_MASS_PER_GRIPPER:.3f} kg")


if __name__ == "__main__":
    main()
