#!/usr/bin/env python3
"""
Convert an MJCF (compiled) into a URDF, preserving mesh-derived inertials.

Tailored for h1_2_magpie.xml (and siblings):
  - Skips the floating base; pelvis becomes URDF root.
  - Emits real mesh-based mass/inertia/CoM (from the MuJoCo compiler).
  - Visual + collision per mesh geom; box primitives become collision-only.
  - 4-bar loop closure expressed as <mimic> joints (URDF has no <connect>):
      *_hinge_2 mimics *_hinge_1 multiplier=-1
      *_hinge_3 mimics *_hinge_1 multiplier=+1

Usage:
  uv run python magpie/mjcf_to_urdf.py magpie/h1_2_magpie.xml magpie/h1_2_magpie.urdf
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

H1_MESH_DIR = "meshes/h1_2_meshes"
MAGPIE_MESH_DIR = "meshes/magpie"
DEFAULT_VEL = 10.0


def quat_wxyz_to_rpy(q):
    """MuJoCo quat is [w,x,y,z]. Return URDF roll-pitch-yaw (intrinsic XYZ)."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def fmt(x):
    return f"{x:.8f}"


def fmt_vec(v):
    return " ".join(fmt(x) for x in v)


def collect_mesh_files(mjcf_path):
    """Map mesh-name -> (urdf-relative-filename, scale-string)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    out = {}
    for mesh in root.iter("mesh"):
        name = mesh.attrib.get("name") or os.path.splitext(mesh.attrib["file"])[0]
        f = mesh.attrib["file"]
        scale = mesh.attrib.get("scale", "1 1 1")
        if f.endswith(".STL") or "h1_2_meshes/" in f:
            urdf_file = f"{H1_MESH_DIR}/{os.path.basename(f)}"
        else:
            urdf_file = f"{MAGPIE_MESH_DIR}/{os.path.basename(f)}"
        out[name] = (urdf_file, scale)
    return out


def collect_actuator_efforts(model):
    """For every actuated joint, return max(|forcerange|) as URDF effort."""
    eff = {}
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        if jid < 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if model.actuator_forcelimited[a]:
            lo, hi = model.actuator_forcerange[a]
            eff[jname] = max(abs(lo), abs(hi))
    return eff


def joints_for_body(model, body_id):
    """List of joint ids attached to this body (in declaration order)."""
    out = []
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            out.append(j)
    return out


def geoms_for_body(model, body_id):
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            out.append(g)
    return out


def write_inertial(out, model, body_id, indent=4):
    pad = " " * indent
    ipos = model.body_ipos[body_id]
    iquat = model.body_iquat[body_id]
    rpy = quat_wxyz_to_rpy(iquat)
    mass = float(model.body_mass[body_id])
    diag = model.body_inertia[body_id]
    out.append(f"{pad}<inertial>")
    out.append(f'{pad}  <origin xyz="{fmt_vec(ipos)}" rpy="{fmt_vec(rpy)}"/>')
    out.append(f'{pad}  <mass value="{fmt(mass)}"/>')
    out.append(
        f'{pad}  <inertia ixx="{fmt(diag[0])}" iyy="{fmt(diag[1])}" izz="{fmt(diag[2])}"'
        f' ixy="0.00000000" ixz="0.00000000" iyz="0.00000000"/>'
    )
    out.append(f"{pad}</inertial>")


def write_geom(out, model, geom_id, mesh_files, kind, indent=4):
    pad = " " * indent
    gtype = int(model.geom_type[geom_id])
    pos = model.geom_pos[geom_id]
    quat = model.geom_quat[geom_id]
    rpy = quat_wxyz_to_rpy(quat)
    out.append(f"{pad}<{kind}>")
    out.append(f'{pad}  <origin xyz="{fmt_vec(pos)}" rpy="{fmt_vec(rpy)}"/>')
    out.append(f"{pad}  <geometry>")
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[geom_id])
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        urdf_file, scale = mesh_files[mname]
        out.append(f'{pad}    <mesh filename="{urdf_file}" scale="{scale}"/>')
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        # MuJoCo stores half-sizes; URDF wants full extents.
        out.append(f'{pad}    <box size="{fmt_vec(2 * size)}"/>')
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = model.geom_size[geom_id, 0], model.geom_size[geom_id, 1]
        out.append(f'{pad}    <cylinder radius="{fmt(radius)}" length="{fmt(2*half)}"/>')
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        out.append(f'{pad}    <sphere radius="{fmt(model.geom_size[geom_id,0])}"/>')
    else:
        out.append(f"{pad}    <!-- unsupported geom type {gtype} -->")
    out.append(f"{pad}  </geometry>")
    if kind == "visual":
        out.append(f'{pad}  <material name="dark_gray"><color rgba="0.1 0.1 0.1 1"/></material>')
    out.append(f"{pad}</{kind}>")


def write_link(out, model, body_id, mesh_files):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    out.append(f'  <link name="{name}">')
    write_inertial(out, model, body_id, indent=4)

    geoms = geoms_for_body(model, body_id)
    for g in geoms:
        gtype = int(model.geom_type[g])
        is_pad_box = (gtype == mujoco.mjtGeom.mjGEOM_BOX)
        if not is_pad_box:
            write_geom(out, model, g, mesh_files, "visual", indent=4)
            write_geom(out, model, g, mesh_files, "collision", indent=4)
        else:
            # mass=0 contact pads → collision-only
            write_geom(out, model, g, mesh_files, "collision", indent=4)
    out.append("  </link>")


def mimic_relation(joint_name):
    """Map *_hinge_2 / *_hinge_3 → (parent_hinge_1, multiplier)."""
    if joint_name.endswith("_hinge_2"):
        return joint_name[: -len("_hinge_2")] + "_hinge_1", -1.0
    if joint_name.endswith("_hinge_3"):
        return joint_name[: -len("_hinge_3")] + "_hinge_1", +1.0
    return None


def write_joint(out, model, child_body_id, efforts):
    parent_id = int(model.body_parentid[child_body_id])
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
    child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_body_id)
    pos = model.body_pos[child_body_id]
    quat = model.body_quat[child_body_id]
    rpy = quat_wxyz_to_rpy(quat)

    js = joints_for_body(model, child_body_id)
    if len(js) == 0:
        # Fixed attachment.
        out.append(f'  <joint name="{child_name}_fixed" type="fixed">')
        out.append(f'    <parent link="{parent_name}"/>')
        out.append(f'    <child link="{child_name}"/>')
        out.append(f'    <origin xyz="{fmt_vec(pos)}" rpy="{fmt_vec(rpy)}"/>')
        out.append("  </joint>")
        return

    if len(js) > 1:
        out.append(f"  <!-- WARNING: body {child_name} has {len(js)} joints; only first emitted -->")

    j = js[0]
    jtype = int(model.jnt_type[j])
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)

    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        # No URDF equivalent: skip (caller treats as root).
        return

    if jtype == mujoco.mjtJoint.mjJNT_HINGE:
        utype = "revolute"
    elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
        utype = "prismatic"
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        out.append(f"  <!-- WARNING: ball joint {jname} has no URDF equivalent; emitted as fixed -->")
        utype = "fixed"
    else:
        utype = "fixed"

    if not np.allclose(model.jnt_pos[j], 0.0):
        out.append(f"  <!-- WARNING: jnt_pos != 0 for {jname}; URDF assumes joint at child origin -->")

    out.append(f'  <joint name="{jname}" type="{utype}">')
    out.append(f'    <parent link="{parent_name}"/>')
    out.append(f'    <child link="{child_name}"/>')
    out.append(f'    <origin xyz="{fmt_vec(pos)}" rpy="{fmt_vec(rpy)}"/>')
    if utype in ("revolute", "prismatic"):
        axis = model.jnt_axis[j]
        out.append(f'    <axis xyz="{fmt_vec(axis)}"/>')
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
        else:
            lo, hi = -np.pi, np.pi
        eff = efforts.get(jname, 100.0)
        out.append(
            f'    <limit lower="{fmt(lo)}" upper="{fmt(hi)}" '
            f'effort="{fmt(eff)}" velocity="{fmt(DEFAULT_VEL)}"/>'
        )
        damping = float(model.dof_damping[model.jnt_dofadr[j]])
        friction = float(model.dof_frictionloss[model.jnt_dofadr[j]])
        out.append(f'    <dynamics damping="{fmt(damping)}" friction="{fmt(friction)}"/>')
        rel = mimic_relation(jname)
        if rel is not None:
            parent_jnt, mult = rel
            out.append(
                f'    <mimic joint="{parent_jnt}" multiplier="{fmt(mult)}" offset="0.00000000"/>'
            )
    out.append("  </joint>")


def convert(mjcf_path, urdf_path):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    mesh_files = collect_mesh_files(mjcf_path)
    efforts = collect_actuator_efforts(model)

    robot_name = os.path.splitext(os.path.basename(mjcf_path))[0]
    out = []
    out.append('<?xml version="1.0" encoding="utf-8"?>')
    out.append(f"<!-- Auto-generated from {os.path.basename(mjcf_path)} by mjcf_to_urdf.py -->")
    out.append("<!-- Inertials are MuJoCo-compiled (mesh volume × default density 1000 kg/m³). -->")
    out.append("<!-- 4-bar loop closure approximated via <mimic> joints (URDF has no <connect>). -->")
    out.append(f'<robot name="{robot_name}">')

    # Write all links first (URDF parsers tolerate any order, but this matches the original).
    for b in range(1, model.nbody):  # skip world body 0
        write_link(out, model, b, mesh_files)

    # Then joints, depth-first by body order.
    for b in range(1, model.nbody):
        # Skip writing a joint for the root (free-base) body.
        js = joints_for_body(model, b)
        if any(int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE for j in js):
            continue
        write_joint(out, model, b, efforts)

    out.append("</robot>")
    with open(urdf_path, "w") as f:
        f.write("\n".join(out) + "\n")

    total_mass = float(np.sum(model.body_mass[1:]))
    print(f"Wrote {urdf_path}")
    print(f"  links={model.nbody-1}, joints emitted from {model.njnt} mjcf joints")
    print(f"  total mass = {total_mass:.4f} kg")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mjcf")
    p.add_argument("urdf")
    a = p.parse_args()
    convert(a.mjcf, a.urdf)
