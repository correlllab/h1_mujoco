#!/usr/bin/env python3
"""
High-level polygon control using 7 virtual actuators.

This script implements a control loop that:
1. Reads desired polygon state from 7 high-level actuators
2. Computes inverse kinematics
3. Sets the actual finger and world actuations

The 7 high-level actuators are:
  - polygon_surface_area (m²)
  - polygon_x, polygon_y, polygon_z (position in meters)
  - polygon_rx, polygon_ry, polygon_rz (orientation in radians)

Usage:
    python polygon_high_level_control.py
    python polygon_high_level_control.py --no-ik  # Direct control only
    python polygon_high_level_control.py --freq 10  # IK at 10 Hz
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
from polygon_control import PolygonController
import time


class HighLevelPolygonController:
    """High-level polygon controller using virtual actuators."""

    # Indices for the 12 low-level actuators (from inspire_right_floating.xml)
    # These come first because inspire_right_floating.xml is included before polygon actuators
    LOW_LEVEL_WORLD_START = 0  # Indices 0-5: world frame actuators
    LOW_LEVEL_FINGER_START = 6  # Indices 6-11: finger actuators

    # Indices for the 7 high-level polygon actuators (come after low-level)
    HIGH_LEVEL_START = 12  # Start after 12 low-level actuators
    HIGH_LEVEL_ACTUATORS = {
        'polygon_surface_area': 12,
        'polygon_x': 13,
        'polygon_y': 14,
        'polygon_z': 15,
        'polygon_rx': 16,
        'polygon_ry': 17,
        'polygon_rz': 18,
    }

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 ik_frequency: float = 10.0, use_ik: bool = True):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            ik_frequency: How often to run IK (Hz)
            use_ik: If True, compute IK; if False, just visualize
        """
        self.model = model
        self.data = data
        self.controller = PolygonController(model, data)
        self.ik_frequency = ik_frequency
        self.ik_period = 1.0 / ik_frequency
        self.use_ik = use_ik

        # Last IK time
        self.last_ik_time = 0.0

        # Cache actuator IDs
        self.high_level_ids = []
        for name in ['polygon_surface_area', 'polygon_x', 'polygon_y', 'polygon_z',
                     'polygon_rx', 'polygon_ry', 'polygon_rz']:
            try:
                self.high_level_ids.append(self.model.actuator(name).id)
            except KeyError:
                print(f"Warning: High-level actuator '{name}' not found")
                self.high_level_ids.append(-1)

        # Cache joint IDs for reading qpos
        self.polygon_joint_ids = []
        for name in ['polygon_area_joint', 'polygon_pos_x', 'polygon_pos_y', 'polygon_pos_z',
                     'polygon_rot_x', 'polygon_rot_y', 'polygon_rot_z']:
            try:
                jid = self.model.joint(name).id
                self.polygon_joint_ids.append(jid)
            except KeyError:
                print(f"Warning: Joint '{name}' not found")
                self.polygon_joint_ids.append(-1)

        print(f"HighLevelPolygonController initialized")
        print(f"  IK frequency: {ik_frequency} Hz")
        print(f"  IK enabled: {use_ik}")
        print(f"  High-level actuators: {len([x for x in self.high_level_ids if x >= 0])}")

    def get_desired_polygon_state(self):
        """Read desired polygon state from high-level actuator controls.

        Returns:
            desired_area: Target area (m²)
            desired_pos: (3,) target position
            desired_ori: (3,) target orientation [roll, pitch, yaw]
        """
        # Read from ctrl values of high-level actuators
        desired_area = 0.002  # Default
        desired_pos = np.zeros(3)
        desired_ori = np.zeros(3)

        # Area (ctrl[12])
        if self.high_level_ids[0] >= 0:
            desired_area = self.data.ctrl[self.high_level_ids[0]]

        # Position (ctrl[13:16])
        for i in range(3):
            if self.high_level_ids[i + 1] >= 0:
                desired_pos[i] = self.data.ctrl[self.high_level_ids[i + 1]]

        # Orientation (ctrl[16:19])
        for i in range(3):
            if self.high_level_ids[i + 4] >= 0:
                desired_ori[i] = self.data.ctrl[self.high_level_ids[i + 4]]

        return desired_area, desired_pos, desired_ori

    def set_desired_polygon_state(self, area: float, pos: np.ndarray, ori: np.ndarray):
        """Set desired polygon state via high-level actuator controls.

        Args:
            area: Target area (m²)
            pos: (3,) target position
            ori: (3,) target orientation [roll, pitch, yaw]
        """
        # Set control values for high-level actuators
        if self.high_level_ids[0] >= 0:
            self.data.ctrl[self.high_level_ids[0]] = area

        for i in range(3):
            if self.high_level_ids[i + 1] >= 0:
                self.data.ctrl[self.high_level_ids[i + 1]] = pos[i]

        for i in range(3):
            if self.high_level_ids[i + 4] >= 0:
                self.data.ctrl[self.high_level_ids[i + 4]] = ori[i]

    def compute_and_apply_ik(self):
        """Compute IK based on desired state and apply to low-level actuators."""
        if not self.use_ik:
            return

        # Get desired state from high-level controls
        desired_area, desired_pos, desired_ori = self.get_desired_polygon_state()

        # Compute IK
        try:
            world_ctrl, finger_ctrl = self.controller.inverse_kinematics_full(
                desired_area=desired_area,
                desired_centroid=desired_pos,
                desired_orientation=desired_ori,
                w_area=10000.0,   # Very high area weight to prioritize area matching
                w_pose=100.0,     # Moderate pose weight
                w_planar=5000.0,  # High planarity weight - polygon must be flat
                w_reg=0.001,      # Very low regularization to allow large movements
                max_iters=100,    # More iterations for convergence
            )

            # Apply to low-level actuators
            # Indices 0-5: world actuators
            # Indices 6-11: finger actuators
            self.data.ctrl[self.LOW_LEVEL_WORLD_START:self.LOW_LEVEL_WORLD_START + 6] = world_ctrl
            self.data.ctrl[self.LOW_LEVEL_FINGER_START:self.LOW_LEVEL_FINGER_START + 6] = finger_ctrl

        except Exception as e:
            print(f"IK failed: {e}")

    def update(self):
        """Update control loop - run IK periodically."""
        current_time = self.data.time

        # Run IK at specified frequency
        if current_time - self.last_ik_time >= self.ik_period:
            self.compute_and_apply_ik()
            self.last_ik_time = current_time

    def print_state(self):
        """Print current and desired polygon states."""
        # Actual state
        actual_state = self.controller.compute_polygon_state()

        # Desired state
        desired_area, desired_pos, desired_ori = self.get_desired_polygon_state()

        print(f"\n{'='*70}")
        print(f"DESIRED Polygon State (from high-level actuators):")
        print(f"  Area:     {desired_area:.6f} m² ({desired_area * 1e4:.2f} cm²)")
        print(f"  Position: [{desired_pos[0]:.4f}, {desired_pos[1]:.4f}, {desired_pos[2]:.4f}] m")
        print(f"  Orientation (RPY): [{np.rad2deg(desired_ori[0]):.1f}°, "
              f"{np.rad2deg(desired_ori[1]):.1f}°, {np.rad2deg(desired_ori[2]):.1f}°]")

        print(f"\nACTUAL Polygon State (from fingertips):")
        print(f"  Area:     {actual_state.area:.6f} m² ({actual_state.area * 1e4:.2f} cm²)")
        print(f"  Position: [{actual_state.centroid[0]:.4f}, {actual_state.centroid[1]:.4f}, "
              f"{actual_state.centroid[2]:.4f}] m")
        print(f"  Orientation (RPY): [{np.rad2deg(actual_state.orientation[0]):.1f}°, "
              f"{np.rad2deg(actual_state.orientation[1]):.1f}°, "
              f"{np.rad2deg(actual_state.orientation[2]):.1f}°]")

        print(f"\nErrors:")
        print(f"  Area error:     {abs(actual_state.area - desired_area):.6f} m² "
              f"({abs(actual_state.area - desired_area) / desired_area * 100:.2f}%)")
        print(f"  Position error: {np.linalg.norm(actual_state.centroid - desired_pos):.6f} m")
        ori_err = actual_state.orientation - desired_ori
        ori_err = (ori_err + np.pi) % (2 * np.pi) - np.pi
        print(f"  Orientation error: {np.linalg.norm(ori_err):.6f} rad "
              f"({np.rad2deg(np.linalg.norm(ori_err)):.2f}°)")
        print(f"{'='*70}\n")


def run_interactive(args):
    """Run interactive high-level polygon control."""

    # Load model
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # Initialize controller
    hl_controller = HighLevelPolygonController(
        model, data,
        ik_frequency=args.freq,
        use_ik=not args.no_ik
    )

    # Reset to home
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Set initial desired state to current state
    initial_state = hl_controller.controller.compute_polygon_state()
    hl_controller.set_desired_polygon_state(
        initial_state.area,
        initial_state.centroid,
        initial_state.orientation
    )

    print("\n=== High-Level Polygon Control ===")
    print(f"\nTotal actuators: {model.nu}")
    print(f"  [0-5]   Low-level world: x, y, z, roll, pitch, yaw")
    print(f"  [6-11]  Low-level fingers: pinky, ring, middle, index, thumb_prox, thumb_yaw")
    print(f"  [12-18] High-level polygon: area, x, y, z, rx, ry, rz")

    print("\n\nKeyboard controls:")
    print("  A - Increase area (+5 cm²)")
    print("  Z - Decrease area (-5 cm²)")
    print("  W/S - Move forward/backward")
    print("  A/D - Move left/right (hold Shift)")
    print("  Q/E - Move up/down")
    print("  I/K - Pitch up/down")
    print("  J/L - Yaw left/right")
    print("  U/O - Roll left/right")
    print("  R - Print state & errors")
    print("  P - Pause/resume")
    print("  1/2/3 - Load keyframes")
    print("  Esc - Quit\n")

    # State
    running = True
    paused = False

    def key_callback(key):
        nonlocal running, paused
        glfw = mujoco.glfw.glfw

        if key == glfw.KEY_ESCAPE:
            running = False
        elif key == glfw.KEY_P:
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == glfw.KEY_R:
            hl_controller.print_state()

        # Get current desired state
        area, pos, ori = hl_controller.get_desired_polygon_state()

        # Area control
        if key == glfw.KEY_A and not (mujoco.glfw.glfw.get_key(
                mujoco.viewer._gui.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
            area += 0.0005  # +5 cm²
            area = min(0.005, area)
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Area: {area*1e4:.2f} cm²")
        elif key == glfw.KEY_Z:
            area -= 0.0005  # -5 cm²
            area = max(0.0005, area)
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Area: {area*1e4:.2f} cm²")

        # Position control
        elif key == glfw.KEY_W:
            pos[0] += 0.01
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        elif key == glfw.KEY_S:
            pos[0] -= 0.01
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        elif key == glfw.KEY_D:
            pos[1] -= 0.01
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        elif key == glfw.KEY_Q:
            pos[2] += 0.01
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        elif key == glfw.KEY_E:
            pos[2] -= 0.01
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # Orientation control
        elif key == glfw.KEY_I:
            ori[1] += 0.1  # pitch up
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")
        elif key == glfw.KEY_K:
            ori[1] -= 0.1  # pitch down
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")
        elif key == glfw.KEY_J:
            ori[2] += 0.1  # yaw left
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")
        elif key == glfw.KEY_L:
            ori[2] -= 0.1  # yaw right
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")
        elif key == glfw.KEY_U:
            ori[0] += 0.1  # roll left
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")
        elif key == glfw.KEY_O:
            ori[0] -= 0.1  # roll right
            hl_controller.set_desired_polygon_state(area, pos, ori)
            print(f"Orientation: [{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]")

        # Keyframes
        elif key == glfw.KEY_1:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            state = hl_controller.controller.compute_polygon_state()
            hl_controller.set_desired_polygon_state(state.area, state.centroid, state.orientation)
            print("→ Home keyframe")
        elif key == glfw.KEY_2:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'spread')
            mujoco.mj_resetDataKeyframe(model, data, kid)
            state = hl_controller.controller.compute_polygon_state()
            hl_controller.set_desired_polygon_state(state.area, state.centroid, state.orientation)
            print("→ Spread keyframe")
        elif key == glfw.KEY_3:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'close')
            mujoco.mj_resetDataKeyframe(model, data, kid)
            state = hl_controller.controller.compute_polygon_state()
            hl_controller.set_desired_polygon_state(state.area, state.centroid, state.orientation)
            print("→ Close keyframe")

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Camera setup
    viewer.cam.azimuth = 90
    viewer.cam.distance = 0.5
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0.07, 0.029, 0.15]

    # Print initial state
    hl_controller.print_state()

    try:
        while viewer.is_running() and running:
            if not paused:
                # Update high-level controller (runs IK periodically)
                hl_controller.update()

                # Step simulation
                mujoco.mj_step(model, data)

                # Update visualization
                state = hl_controller.controller.compute_polygon_state()
                hl_controller.controller.update_polygon_visualization(state)
                hl_controller.controller.add_viewer_geometry(viewer, state)

            viewer.sync()
            time.sleep(max(model.opt.timestep, 0.001))

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='High-level polygon control with automatic IK')
    parser.add_argument('--model', type=str, default='polygon_grasp.xml',
                       help='Path to XML model')
    parser.add_argument('--freq', type=float, default=10.0,
                       help='IK update frequency (Hz)')
    parser.add_argument('--no-ik', action='store_true',
                       help='Disable IK (visualization only)')
    args = parser.parse_args()

    run_interactive(args)


if __name__ == '__main__':
    main()
