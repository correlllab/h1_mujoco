#!/usr/bin/env python3
"""
Fingertip Polygon Control for Inspire Dexterous Hand.

Controls the polygon P formed by fingertip positions via:
  - Surface area (1 DOF)
  - 6D pose: position + orientation (6 DOF)
  - Total: 7 control variables

Computes inverse kinematics to achieve desired polygon configuration
by actuating the 6 DOF world frame and/or finger joints.

Usage:
    # Visualize current polygon
    python polygon_control.py

    # Control to desired area and pose
    python polygon_control.py --area 0.002 --pos 0.3 0.4 0.2

    # Use full optimization (vs fixed fingers)
    python polygon_control.py --optimize --area 0.0015
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time

try:
    from scipy.optimize import least_squares, minimize
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, IK optimization disabled")


@dataclass
class PolygonState:
    """State of the fingertip polygon."""
    area: float                      # Surface area (m²)
    centroid: np.ndarray            # (3,) centroid position in world frame
    normal: np.ndarray              # (3,) unit normal vector (from PCA)
    orientation: np.ndarray         # (3,) roll, pitch, yaw (radians)
    vertices: np.ndarray            # (5, 3) fingertip positions
    vertex_names: list              # Finger names in order


class PolygonController:
    """Controller for fingertip polygon of Inspire hand."""

    FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

    TIP_SITES = {
        "thumb": "right_thumb_tip",
        "index": "right_index_tip",
        "middle": "right_middle_tip",
        "ring": "right_ring_tip",
        "pinky": "right_pinky_tip",
    }

    # Actuator names
    WORLD_ACTUATORS = [
        "right_pos_x_position",
        "right_pos_y_position",
        "right_pos_z_position",
        "right_rot_x_position",
        "right_rot_y_position",
        "right_rot_z_position",
    ]

    FINGER_ACTUATORS = [
        "thumb_yaw", "thumb_proximal",
        "index", "middle", "ring", "pinky",
    ]

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Cache site IDs
        self.tip_site_ids = {}
        for finger, site_name in self.TIP_SITES.items():
            try:
                self.tip_site_ids[finger] = self.model.site(site_name).id
            except KeyError:
                print(f"Warning: site '{site_name}' not found")

        # Cache actuator IDs
        self.world_actuator_ids = []
        for name in self.WORLD_ACTUATORS:
            try:
                self.world_actuator_ids.append(self.model.actuator(name).id)
            except KeyError:
                print(f"Warning: actuator '{name}' not found")

        self.finger_actuator_ids = []
        for name in self.FINGER_ACTUATORS:
            try:
                self.finger_actuator_ids.append(self.model.actuator(name).id)
            except KeyError:
                print(f"Warning: actuator '{name}' not found")

        # Get joint IDs for direct qpos access
        self.world_joint_ids = []
        for i in range(6):
            joint_name = f"right_{'pos' if i < 3 else 'rot'}_{'xyz'[i%3]}"
            try:
                self.world_joint_ids.append(self.model.joint(joint_name).id)
            except KeyError:
                print(f"Warning: joint '{joint_name}' not found")

        # Mocap body for polygon visualization
        try:
            self.polygon_mocap_id = self.model.body("polygon_center").mocapid[0]
        except KeyError:
            self.polygon_mocap_id = -1
            print("Warning: polygon_center mocap body not found")

        print(f"PolygonController initialized")
        print(f"  Tip sites: {len(self.tip_site_ids)}")
        print(f"  World actuators: {len(self.world_actuator_ids)}")
        print(f"  Finger actuators: {len(self.finger_actuator_ids)}")

    def get_fingertip_positions(self) -> np.ndarray:
        """Get current fingertip positions in world frame.

        Returns:
            (5, 3) array of fingertip positions [thumb, index, middle, ring, pinky]
        """
        positions = np.zeros((5, 3))
        for i, finger in enumerate(self.FINGER_ORDER):
            if finger in self.tip_site_ids:
                site_id = self.tip_site_ids[finger]
                positions[i] = self.data.site_xpos[site_id].copy()
        return positions

    def compute_polygon_area(self, vertices: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute area of 3D polygon using best-fit plane projection.

        Uses PCA to find the best-fit plane, projects vertices onto it,
        and computes area using the Shoelace formula.

        Args:
            vertices: (N, 3) array of vertex positions

        Returns:
            area: Surface area (m²)
            normal: (3,) unit normal vector of best-fit plane
        """
        if len(vertices) < 3:
            return 0.0, np.array([0, 0, 1])

        # Center the vertices
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid

        # PCA to find best-fit plane normal
        # Normal is the eigenvector with smallest eigenvalue
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue -> normal direction

        # Ensure normal points "outward" (positive z component)
        if normal[2] < 0:
            normal = -normal

        # Project vertices onto the best-fit plane
        # Plane basis: two eigenvectors with largest eigenvalues
        basis_u = eigenvectors[:, 2]  # Largest eigenvalue
        basis_v = eigenvectors[:, 1]  # Second largest

        # 2D coordinates in plane
        coords_2d = np.column_stack([
            centered @ basis_u,
            centered @ basis_v,
        ])

        # Shoelace formula for polygon area
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        # Close the polygon
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

        return area, normal

    def compute_orientation_from_normal(self, normal: np.ndarray,
                                       centroid: np.ndarray,
                                       vertices: np.ndarray) -> np.ndarray:
        """Compute roll, pitch, yaw from polygon normal and vertices.

        Args:
            normal: (3,) unit normal vector
            centroid: (3,) polygon centroid
            vertices: (N, 3) vertex positions

        Returns:
            (3,) array of [roll, pitch, yaw] in radians
        """
        # Z-axis of polygon frame is the normal
        z_axis = normal / np.linalg.norm(normal)

        # X-axis points from centroid to first vertex (thumb)
        if len(vertices) > 0:
            to_thumb = vertices[0] - centroid
            # Project onto plane perpendicular to normal
            to_thumb = to_thumb - (to_thumb @ z_axis) * z_axis
            x_axis = to_thumb / (np.linalg.norm(to_thumb) + 1e-9)
        else:
            x_axis = np.array([1, 0, 0])

        # Y-axis completes right-handed frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

        # Rotation matrix: columns are frame axes
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Extract Euler angles (ZYX convention: yaw-pitch-roll)
        pitch = np.arcsin(-R[2, 0])
        if np.abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock
            yaw = 0
            roll = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([roll, pitch, yaw])

    def compute_polygon_state(self) -> PolygonState:
        """Compute current polygon state from fingertip positions.

        Returns:
            PolygonState with area, centroid, normal, orientation
        """
        vertices = self.get_fingertip_positions()
        centroid = vertices.mean(axis=0)
        area, normal = self.compute_polygon_area(vertices)
        orientation = self.compute_orientation_from_normal(normal, centroid, vertices)

        return PolygonState(
            area=area,
            centroid=centroid,
            normal=normal,
            orientation=orientation,
            vertices=vertices,
            vertex_names=self.FINGER_ORDER,
        )

    def update_polygon_visualization(self, state: PolygonState):
        """Update mocap body to visualize polygon centroid and normal.

        Args:
            state: Current polygon state
        """
        if self.polygon_mocap_id < 0:
            return

        # Update mocap position to centroid
        self.data.mocap_pos[self.polygon_mocap_id] = state.centroid

        # Update mocap orientation from roll, pitch, yaw
        roll, pitch, yaw = state.orientation
        # Convert to quaternion (w, x, y, z)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        self.data.mocap_quat[self.polygon_mocap_id] = [qw, qx, qy, qz]

    def inverse_kinematics_baseline(self,
                                    desired_centroid: np.ndarray,
                                    desired_orientation: np.ndarray,
                                    max_iters: int = 100,
                                    tol: float = 1e-4) -> np.ndarray:
        """Baseline IK: adjust only 6 DOF world actuations (fixed fingers).

        Simple approach that only positions/orients the hand, area is emergent.

        Args:
            desired_centroid: (3,) target centroid position
            desired_orientation: (3,) target [roll, pitch, yaw]
            max_iters: Maximum iterations
            tol: Convergence tolerance

        Returns:
            (6,) array of world control values [x, y, z, rx, ry, rz]
        """
        if not HAS_SCIPY:
            print("Error: scipy required for IK")
            return np.zeros(6)

        # Current world configuration
        world_qpos_current = np.zeros(6)
        for i, jid in enumerate(self.world_joint_ids):
            world_qpos_current[i] = self.data.qpos[self.model.jnt_qposadr[jid]]

        def residual(x_world):
            """Residual: difference between current and desired pose."""
            # Set world configuration
            for i, jid in enumerate(self.world_joint_ids):
                self.data.qpos[self.model.jnt_qposadr[jid]] = x_world[i]

            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Compute current state
            state = self.compute_polygon_state()

            # Residuals: position and orientation error
            pos_error = state.centroid - desired_centroid
            ori_error = state.orientation - desired_orientation

            # Wrap orientation errors to [-pi, pi]
            ori_error = (ori_error + np.pi) % (2 * np.pi) - np.pi

            return np.concatenate([pos_error, ori_error])

        # Solve using least squares
        result = least_squares(
            residual,
            world_qpos_current,
            max_nfev=max_iters,
            ftol=tol,
            xtol=tol,
        )

        if result.success:
            print(f"IK converged in {result.nfev} iterations")
        else:
            print(f"IK failed: {result.message}")

        return result.x

    def inverse_kinematics_full(self,
                                desired_area: float,
                                desired_centroid: np.ndarray,
                                desired_orientation: np.ndarray,
                                w_area: float = 1000.0,
                                w_pose: float = 1.0,
                                w_planar: float = 1000.0,
                                w_reg: float = 0.01,
                                max_iters: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Full IK: optimize over world + finger actuations.

        Formulates as weighted least squares:
            minimize:  w_area * (A_desired - A_actual)^2 +
                      w_pose * ||pose_desired - pose_actual||^2 +
                      w_planar * sum(|n · (p_i - centroid)|^2) +
                      w_reg * ||q - q_0||^2

        Args:
            desired_area: Target polygon area (m²)
            desired_centroid: (3,) target centroid position
            desired_orientation: (3,) target [roll, pitch, yaw]
            w_area: Weight for area error
            w_pose: Weight for pose error
            w_planar: Weight for planarity constraint (polygon should be flat)
            w_reg: Regularization weight (stay close to current config)
            max_iters: Maximum iterations

        Returns:
            world_ctrl: (6,) world control values
            finger_ctrl: (6,) finger control values [thumb_yaw, thumb_prox, index, middle, ring, pinky]
        """
        if not HAS_SCIPY:
            print("Error: scipy required for IK")
            return np.zeros(6), np.zeros(6)

        # Current configuration (as initial guess)
        q0_world = np.zeros(6)
        for i, jid in enumerate(self.world_joint_ids):
            q0_world[i] = self.data.qpos[self.model.jnt_qposadr[jid]]

        # Finger actuator controls
        # Use actual joint positions as initial guess, not ctrl (which might be 0)
        # This gives a better starting point for optimization
        finger_joint_names = [
            'pinky_proximal_joint',
            'ring_proximal_joint',
            'middle_proximal_joint',
            'index_proximal_joint',
            'thumb_proximal_pitch_joint',
            'thumb_proximal_yaw_joint',
        ]

        q0_finger = np.zeros(6)
        for i, jname in enumerate(finger_joint_names):
            try:
                jid = self.model.joint(jname).id
                q0_finger[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
            except:
                # If joint not found, use ctrl as fallback
                if i < len(self.finger_actuator_ids):
                    q0_finger[i] = self.data.ctrl[self.finger_actuator_ids[i]]

        q0 = np.concatenate([q0_world, q0_finger])

        # Get joint limits
        bounds = []
        for jid in self.world_joint_ids:
            jnt_range = self.model.jnt_range[jid]
            if np.any(jnt_range != 0):
                bounds.append((jnt_range[0], jnt_range[1]))
            else:
                bounds.append((-np.inf, np.inf))

        # Finger actuator limits
        for act_id in self.finger_actuator_ids:
            ctrl_range = self.model.actuator_ctrlrange[act_id]
            if np.any(ctrl_range != 0):
                bounds.append((ctrl_range[0], ctrl_range[1]))
            else:
                bounds.append((-np.inf, np.inf))

        def residual(q):
            """Weighted residual vector."""
            q_world = q[:6]
            q_finger = q[6:]

            # Set world configuration
            for i, jid in enumerate(self.world_joint_ids):
                self.data.qpos[self.model.jnt_qposadr[jid]] = q_world[i]

            # Set finger joint positions DIRECTLY in qpos
            # This is necessary for the residual function to see the effect
            finger_joint_names = [
                'pinky_proximal_joint',
                'ring_proximal_joint',
                'middle_proximal_joint',
                'index_proximal_joint',
                'thumb_proximal_pitch_joint',
                'thumb_proximal_yaw_joint',
            ]

            for i, jname in enumerate(finger_joint_names):
                try:
                    jid = self.model.joint(jname).id
                    self.data.qpos[self.model.jnt_qposadr[jid]] = q_finger[i]
                except:
                    pass  # Skip if joint not found

            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Compute current state
            state = self.compute_polygon_state()

            # Area error (normalized)
            area_error = (state.area - desired_area) * np.sqrt(w_area)

            # Pose errors
            pos_error = (state.centroid - desired_centroid) * np.sqrt(w_pose)
            ori_error = (state.orientation - desired_orientation) * np.sqrt(w_pose)
            ori_error = (ori_error + np.pi) % (2 * np.pi) - np.pi  # Wrap

            # Planarity error: measure how far each fingertip deviates from best-fit plane
            # For a perfect planar polygon, all points lie on the plane
            vertices = state.vertices  # (5, 3) fingertip positions
            centroid = state.centroid
            normal = state.normal  # Normal vector from PCA

            # Distance of each vertex from the best-fit plane
            # d_i = |n · (p_i - centroid)|
            planar_errors = np.zeros(5)
            for i in range(5):
                deviation = vertices[i] - centroid
                planar_errors[i] = abs(np.dot(normal, deviation)) * np.sqrt(w_planar)

            # Regularization: stay close to initial config
            reg_error = (q - q0) * np.sqrt(w_reg)

            return np.concatenate([
                [area_error],
                pos_error,
                ori_error,
                planar_errors,  # 5 planarity errors (one per vertex)
                reg_error,
            ])

        # Solve
        print(f"Starting full IK optimization...")
        print(f"  Desired area: {desired_area:.6f} m²")
        print(f"  Desired centroid: [{desired_centroid[0]:.3f}, {desired_centroid[1]:.3f}, {desired_centroid[2]:.3f}]")
        print(f"  Weights: area={w_area}, pose={w_pose}, reg={w_reg}")

        result = least_squares(
            residual,
            q0,
            bounds=(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),
            max_nfev=max_iters,
            ftol=1e-6,
            xtol=1e-6,
            verbose=1,
        )

        if result.success:
            print(f"IK converged in {result.nfev} iterations")
            print(f"  Final residual: {result.cost:.6f}")
        else:
            print(f"IK failed: {result.message}")

        q_opt = result.x
        return q_opt[:6], q_opt[6:]

    def print_state(self, state: PolygonState):
        """Print polygon state to console."""
        print(f"\n{'='*70}")
        print(f"Polygon State:")
        print(f"  Area:     {state.area:.6f} m² ({state.area * 1e4:.2f} cm²)")
        print(f"  Centroid: [{state.centroid[0]:.4f}, {state.centroid[1]:.4f}, {state.centroid[2]:.4f}] m")
        print(f"  Normal:   [{state.normal[0]:.4f}, {state.normal[1]:.4f}, {state.normal[2]:.4f}]")
        print(f"  Orientation (RPY): [{np.rad2deg(state.orientation[0]):.1f}°, "
              f"{np.rad2deg(state.orientation[1]):.1f}°, {np.rad2deg(state.orientation[2]):.1f}°]")
        print(f"\nVertex positions:")
        for i, (name, pos) in enumerate(zip(state.vertex_names, state.vertices)):
            print(f"  {name:>6s}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
        print(f"{'='*70}\n")

    def add_viewer_geometry(self, viewer, state: PolygonState):
        """Add polygon edges to viewer for visualization."""
        if viewer is None:
            return

        scn = viewer.user_scn
        # Don't clear existing geometry, just add polygon

        # Draw polygon edges
        vertices = state.vertices
        n_edges = len(vertices)

        for i in range(n_edges):
            if scn.ngeom >= scn.maxgeom:
                break

            p1 = vertices[i]
            p2 = vertices[(i + 1) % n_edges]

            # Color based on edge (cycle through finger colors)
            colors = [
                np.array([1.0, 0.2, 0.2, 0.8]),  # thumb-index
                np.array([0.2, 0.4, 1.0, 0.8]),  # index-middle
                np.array([0.2, 0.8, 0.2, 0.8]),  # middle-ring
                np.array([0.8, 0.2, 0.8, 0.8]),  # ring-pinky
                np.array([1.0, 0.6, 0.1, 0.8]),  # pinky-thumb
            ]
            color = colors[i % len(colors)]

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                color.astype(np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                0.002,  # width
                p1.astype(np.float64),
                p2.astype(np.float64),
            )
            scn.geoms[scn.ngeom].rgba[:] = color
            scn.ngeom += 1

        # Draw normal vector at centroid
        if scn.ngeom < scn.maxgeom:
            normal_scale = 0.05  # 5cm arrow
            end = state.centroid + state.normal * normal_scale

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.003,
                state.centroid.astype(np.float64),
                end.astype(np.float64),
            )
            scn.ngeom += 1


def run_interactive(xml_path: str,
                   desired_area: Optional[float] = None,
                   desired_pos: Optional[np.ndarray] = None,
                   desired_ori: Optional[np.ndarray] = None,
                   use_optimization: bool = False):
    """Run interactive visualization and control.

    Args:
        xml_path: Path to MuJoCo XML model
        desired_area: Target area (m²), None to skip IK
        desired_pos: Target position (3,), None to use current
        desired_ori: Target orientation (3,) roll/pitch/yaw, None to use current
        use_optimization: If True, use full optimization; else baseline
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    controller = PolygonController(model, data)

    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("Reset to 'home' keyframe")
    mujoco.mj_forward(model, data)

    # Compute initial state
    initial_state = controller.compute_polygon_state()
    controller.print_state(initial_state)

    # Perform IK if desired state specified
    if desired_area is not None or desired_pos is not None:
        target_area = desired_area if desired_area is not None else initial_state.area
        target_pos = desired_pos if desired_pos is not None else initial_state.centroid
        target_ori = desired_ori if desired_ori is not None else initial_state.orientation

        print(f"\nComputing IK...")
        print(f"  Mode: {'Full optimization' if use_optimization else 'Baseline (fixed fingers)'}")

        if use_optimization:
            world_ctrl, finger_ctrl = controller.inverse_kinematics_full(
                target_area, target_pos, target_ori
            )
            # Apply controls
            data.ctrl[controller.world_actuator_ids] = world_ctrl
            data.ctrl[controller.finger_actuator_ids] = finger_ctrl
        else:
            world_ctrl = controller.inverse_kinematics_baseline(
                target_pos, target_ori
            )
            # Apply world controls
            data.ctrl[controller.world_actuator_ids] = world_ctrl

        # Run forward to settle
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Print final state
        final_state = controller.compute_polygon_state()
        controller.print_state(final_state)

        print(f"\nIK Results:")
        if desired_area is not None:
            print(f"  Area error:     {abs(final_state.area - target_area):.6f} m² "
                  f"({abs(final_state.area - target_area) / target_area * 100:.2f}%)")
        print(f"  Position error: {np.linalg.norm(final_state.centroid - target_pos):.6f} m")
        ori_err = final_state.orientation - target_ori
        ori_err = (ori_err + np.pi) % (2 * np.pi) - np.pi
        print(f"  Orientation error: {np.linalg.norm(ori_err):.6f} rad ({np.rad2deg(np.linalg.norm(ori_err)):.2f}°)")

    # Launch viewer
    running = True
    paused = False

    def key_callback(key):
        nonlocal running, paused
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
            running = False
        elif key == glfw.KEY_P:
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == glfw.KEY_S:
            state = controller.compute_polygon_state()
            controller.print_state(state)
        elif key == glfw.KEY_R:
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                print("Reset to 'home' keyframe")

    print("\nControls:")
    print("  P     - Pause/resume")
    print("  R     - Reset to home keyframe")
    print("  S     - Print polygon state")
    print("  Q/Esc - Quit")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Camera setup
    viewer.cam.azimuth = 90
    viewer.cam.distance = 0.5
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0.07, 0.029, 0.15]

    try:
        while viewer.is_running() and running:
            if not paused:
                mujoco.mj_step(model, data)

                # Update polygon state and visualization
                state = controller.compute_polygon_state()
                controller.update_polygon_visualization(state)
                controller.add_viewer_geometry(viewer, state)

            viewer.sync()
            time.sleep(max(model.opt.timestep, 0.001))

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Done")


def main():
    parser = argparse.ArgumentParser(
        description='Fingertip Polygon Control for Inspire Hand')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to XML model (default: polygon_grasp.xml)')
    parser.add_argument('--area', type=float, default=None,
                       help='Desired polygon area in m² (e.g., 0.002 = 20 cm²)')
    parser.add_argument('--pos', type=float, nargs=3, default=None,
                       help='Desired centroid position [x y z] in meters')
    parser.add_argument('--ori', type=float, nargs=3, default=None,
                       help='Desired orientation [roll pitch yaw] in degrees')
    parser.add_argument('--optimize', action='store_true',
                       help='Use full optimization (vs baseline fixed fingers)')
    args = parser.parse_args()

    if args.model is None:
        args.model = str(Path(__file__).parent / "polygon_grasp.xml")

    # Convert orientation to radians
    desired_ori = None
    if args.ori is not None:
        desired_ori = np.deg2rad(args.ori)

    run_interactive(
        args.model,
        desired_area=args.area,
        desired_pos=np.array(args.pos) if args.pos else None,
        desired_ori=desired_ori,
        use_optimization=args.optimize,
    )


if __name__ == '__main__':
    main()
