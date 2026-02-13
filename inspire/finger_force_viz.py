#!/usr/bin/env python3
"""
Finger Force Visualization and Grasp Analysis Tool for Inspire Hand.

Tracks contact forces, fingertip sensor wrenches, and evaluates grasp
quality (force closure via Ferrari-Canny metric) as the Inspire hand
moves to the 'home' keyframe and grasps a hanging box.

Visualizations:
  - MuJoCo viewer: contact force arrows, force closure indicator
  - Matplotlib (up to 6 panels): wrench cone, contact vs sensor, time series,
    quality, contact count (optional), local quality window (optional)

Usage:
    # Live simulation
    python finger_force_viz.py
    python finger_force_viz.py --record data.npz
    python finger_force_viz.py --cone-edges 12
    python finger_force_viz.py --no-viz

    # Replay from recording
    python finger_force_viz.py --replay data.npz

Controls (live mode):
    P         - Pause/resume simulation
    R         - Reset to home keyframe
    S         - Print detailed status
    F         - Toggle friction cone visualization
    C         - Toggle contact count plot
    L         - Toggle local Ferrari-Canny window plot
    Q/Esc     - Quit
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, force closure analysis disabled")

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import warnings
    # Suppress tkinter thread warnings - matplotlib in threads works but complains
    warnings.filterwarnings('ignore', message='.*Matplotlib.*main thread.*')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, external visualization disabled")


# ──────── Data classes ────────

@dataclass
class FingerContact:
    """A single contact between a finger geom and the box."""
    finger_name: str
    contact_id: int
    body_name: str
    position: np.ndarray       # (3,) contact point in world frame
    frame: np.ndarray          # (3,3) rows: normal, tangent1, tangent2
    wrench: np.ndarray         # (6,) from mj_contactForce (force on geom1)
    normal_force: float        # scalar along normal
    friction_force: np.ndarray # (2,) tangential components
    box_is_geom1: bool = True  # if True, wrench is on box; negate for finger


@dataclass
class FingerSensorData:
    """Force/torque sensor reading for one fingertip site."""
    finger_name: str
    force_site: np.ndarray     # (3,) in site frame
    torque_site: np.ndarray    # (3,) in site frame
    force_world: np.ndarray    # (3,) transformed to world frame
    torque_world: np.ndarray   # (3,) transformed to world frame


@dataclass
class GraspState:
    """Complete snapshot of grasp state at one timestep."""
    time: float
    contacts: list
    sensor_data: dict
    comparison: dict
    object_pos: np.ndarray
    force_closure: bool
    ferrari_canny: float
    primitive_wrenches: np.ndarray


@dataclass
class TimeSeriesBuffer:
    """Rolling buffer for time-series plotting."""
    max_len: int = 2000
    times: list = field(default_factory=list)
    ferrari_canny: list = field(default_factory=list)
    force_closure: list = field(default_factory=list)
    contact_forces: dict = field(default_factory=dict)
    sensor_forces: dict = field(default_factory=dict)
    num_contacts: list = field(default_factory=list)

    def append(self, state: GraspState):
        self.times.append(state.time)
        self.ferrari_canny.append(state.ferrari_canny)
        self.force_closure.append(state.force_closure)
        self.num_contacts.append(len(state.contacts))

        for finger, comp in state.comparison.items():
            if finger not in self.contact_forces:
                self.contact_forces[finger] = []
                self.sensor_forces[finger] = []
            self.contact_forces[finger].append(
                np.linalg.norm(comp['contact_force_world']))
            self.sensor_forces[finger].append(
                np.linalg.norm(comp['sensor_force_world']))

        if len(self.times) > self.max_len:
            self._trim()

    def _trim(self):
        excess = len(self.times) - self.max_len
        self.times = self.times[excess:]
        self.ferrari_canny = self.ferrari_canny[excess:]
        self.force_closure = self.force_closure[excess:]
        self.num_contacts = self.num_contacts[excess:]
        for k in self.contact_forces:
            self.contact_forces[k] = self.contact_forces[k][excess:]
            self.sensor_forces[k] = self.sensor_forces[k][excess:]


# ──────── Main class ────────

class FingerForceVisualizer:
    """Grasp force analysis and visualization for the Inspire hand."""

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    TIP_SITES = {
        "thumb": "right_thumb_tip",
        "index": "right_index_tip",
        "middle": "right_middle_tip",
        "ring": "right_ring_tip",
        "pinky": "right_pinky_tip",
    }

    FORCE_SENSORS = {
        "thumb": "thumb_tip_force",
        "index": "index_tip_force",
        "middle": "middle_tip_force",
        "ring": "ring_tip_force",
        "pinky": "pinky_tip_force",
    }

    TORQUE_SENSORS = {
        "thumb": "thumb_tip_torque",
        "index": "index_tip_torque",
        "middle": "middle_tip_torque",
        "ring": "ring_tip_torque",
        "pinky": "pinky_tip_torque",
    }

    FINGER_COLORS = {
        "thumb":  np.array([1.0, 0.2, 0.2, 0.9]),
        "index":  np.array([0.2, 0.4, 1.0, 0.9]),
        "middle": np.array([0.2, 0.8, 0.2, 0.9]),
        "ring":   np.array([0.8, 0.2, 0.8, 0.9]),
        "pinky":  np.array([1.0, 0.6, 0.1, 0.9]),
        "palm":   np.array([0.5, 0.5, 0.5, 0.9]),
    }

    FINGER_COLORS_RGB = {
        "thumb":  (1.0, 0.2, 0.2),
        "index":  (0.2, 0.4, 1.0),
        "middle": (0.2, 0.8, 0.2),
        "ring":   (0.8, 0.2, 0.8),
        "pinky":  (1.0, 0.6, 0.1),
        "palm":   (0.5, 0.5, 0.5),
    }

    def __init__(self, xml_path=None, friction_cone_edges=8, record_path=None):
        if xml_path is None:
            xml_path = str(Path(__file__).parent / "inspire_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Body-to-finger mapping (for unnamed collision geoms)
        self.body_to_finger = self._build_body_to_finger_map()
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object")

        # Cache sensor addresses
        self.sensor_addrs = self._cache_sensor_addresses()

        # Cache site IDs
        self.tip_site_ids = {}
        for finger, site_name in self.TIP_SITES.items():
            try:
                self.tip_site_ids[finger] = self.model.site(site_name).id
            except KeyError:
                print(f"Warning: site '{site_name}' not found")

        # Parameters
        self.friction_cone_edges = friction_cone_edges
        self.record_path = record_path

        # State
        self.current_state = None
        self.history = TimeSeriesBuffer()
        self.recording_data = []

        # Control
        self.running = True
        self.paused = False
        self.show_friction_cones = True
        self.show_contact_count_plot = False
        self.show_fc_local_plot = False
        self.fc_local_window = 0.2  # seconds
        self.viewer = None
        self.viz_thread = None

        print(f"Loaded model from: {xml_path}")
        print(f"Box geom ID: {self.box_geom_id}")
        print(f"Body-to-finger map: { {self._body_name(bid): f for bid, f in self.body_to_finger.items()} }")
        print(f"Tip site IDs: {self.tip_site_ids}")
        print(f"Friction cone edges: {self.friction_cone_edges}")

    def _body_name(self, body_id):
        """Get body name from ID."""
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def _build_body_to_finger_map(self):
        """Build mapping from body ID to finger name.

        Uses body names from inspire_right.xml:
          base -> palm
          thumb_* -> thumb
          index_* -> index
          middle_* -> middle
          ring_* -> ring
          pinky_* -> pinky
        """
        body_to_finger = {}
        patterns = [
            (re.compile(r'^thumb_'), "thumb"),
            (re.compile(r'^index_'), "index"),
            (re.compile(r'^middle_'), "middle"),
            (re.compile(r'^ring_'), "ring"),
            (re.compile(r'^pinky_'), "pinky"),
            (re.compile(r'^base$'), "palm"),
        ]

        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None:
                continue
            for pattern, finger in patterns:
                if pattern.match(name):
                    body_to_finger[body_id] = finger
                    break

        return body_to_finger

    def _cache_sensor_addresses(self):
        """Pre-compute sensor data addresses for fast reading."""
        addrs = {}
        for finger in self.FINGER_NAMES:
            try:
                f_id = self.model.sensor(self.FORCE_SENSORS[finger]).id
                t_id = self.model.sensor(self.TORQUE_SENSORS[finger]).id
                addrs[finger] = {
                    'force_adr': self.model.sensor_adr[f_id],
                    'torque_adr': self.model.sensor_adr[t_id],
                }
            except KeyError as e:
                print(f"Warning: sensor not found for {finger}: {e}")
        return addrs

    # ──────── Simulation ────────

    def setup_simulation(self):
        """Reset to home keyframe and run forward kinematics."""
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            print("Reset to 'home' keyframe")
        else:
            print("Warning: 'home' keyframe not found")
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """One simulation step + full analysis."""
        mujoco.mj_step(self.model, self.data)

        contacts = self.detect_contacts()
        sensor_data = self.read_sensors()
        comparison = self.compare_contact_vs_sensor(contacts, sensor_data)

        object_pos = self.data.xpos[self.object_body_id].copy()
        primitive_wrenches = self.compute_wrench_cone(contacts, object_pos)
        force_closure, ferrari_canny = self.evaluate_force_closure(
            primitive_wrenches)

        self.current_state = GraspState(
            time=self.data.time,
            contacts=contacts,
            sensor_data=sensor_data,
            comparison=comparison,
            object_pos=object_pos,
            force_closure=force_closure,
            ferrari_canny=ferrari_canny,
            primitive_wrenches=primitive_wrenches,
        )
        self.history.append(self.current_state)

        if self.record_path:
            self.recording_data.append({
                'time': self.data.time,
                'num_contacts': len(contacts),
                'force_closure': force_closure,
                'ferrari_canny': ferrari_canny,
                'object_pos': object_pos.copy(),
                'contact_wrenches': {c.finger_name: c.wrench.copy()
                                     for c in contacts},
                'sensor_wrenches': {f: np.concatenate([sd.force_world,
                                                       sd.torque_world])
                                    for f, sd in sensor_data.items()},
                'comparison': {f: {k: v.copy() if isinstance(v, np.ndarray)
                                   else v
                                   for k, v in comp.items()}
                               for f, comp in comparison.items()},
            })

    # ──────── Contact detection ────────

    def detect_contacts(self):
        """Scan active contacts for box-finger pairs."""
        contacts = []
        result = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Determine which geom is box, which is finger
            finger_name = None
            finger_body_name = None
            box_is_geom1 = False

            if geom1 == self.box_geom_id:
                body_id = self.model.geom_bodyid[geom2]
                finger_name = self.body_to_finger.get(body_id)
                finger_body_name = self._body_name(body_id)
                box_is_geom1 = True
            elif geom2 == self.box_geom_id:
                body_id = self.model.geom_bodyid[geom1]
                finger_name = self.body_to_finger.get(body_id)
                finger_body_name = self._body_name(body_id)
                box_is_geom1 = False

            if finger_name is None:
                continue

            # Extract contact wrench
            mujoco.mj_contactForce(self.model, self.data, i, result)

            # contact.frame: first 9 elements = 3x3 rotation
            # rows: normal, tangent1, tangent2
            frame = contact.frame.reshape(3, 3)

            contacts.append(FingerContact(
                finger_name=finger_name,
                contact_id=i,
                body_name=finger_body_name or "",
                position=contact.pos.copy(),
                frame=frame.copy(),
                wrench=result.copy(),
                normal_force=result[0],
                friction_force=result[1:3].copy(),
                box_is_geom1=box_is_geom1,
            ))

        return contacts

    # ──────── Sensor reading ────────

    def read_sensors(self):
        """Read force/torque sensors at each fingertip site."""
        sensor_data = {}

        for finger in self.FINGER_NAMES:
            if finger not in self.sensor_addrs:
                continue
            if finger not in self.tip_site_ids:
                continue

            addrs = self.sensor_addrs[finger]
            site_id = self.tip_site_ids[finger]

            # Read in site frame
            force_site = self.data.sensordata[
                addrs['force_adr']:addrs['force_adr'] + 3].copy()
            torque_site = self.data.sensordata[
                addrs['torque_adr']:addrs['torque_adr'] + 3].copy()

            # Transform to world frame
            site_xmat = self.data.site_xmat[site_id].reshape(3, 3)
            force_world = site_xmat @ force_site
            torque_world = site_xmat @ torque_site

            sensor_data[finger] = FingerSensorData(
                finger_name=finger,
                force_site=force_site,
                torque_site=torque_site,
                force_world=force_world,
                torque_world=torque_world,
            )

        return sensor_data

    # ──────── Wrench cone / force closure ────────

    def linearize_friction_cone(self, contact, mu):
        """Linearize friction cone into k primitive force directions.

        Each primitive is: normal + mu * (cos(theta)*t1 + sin(theta)*t2)
        normalized to unit length.

        Returns (k, 3) array of unit force directions in world frame.
        """
        k = self.friction_cone_edges
        normal = contact.frame[0]
        tangent1 = contact.frame[1]
        tangent2 = contact.frame[2]

        primitives = np.zeros((k, 3))
        for j in range(k):
            theta = 2.0 * np.pi * j / k
            f = normal + mu * (np.cos(theta) * tangent1
                               + np.sin(theta) * tangent2)
            norm = np.linalg.norm(f)
            if norm > 1e-12:
                f /= norm
            primitives[j] = f

        return primitives

    def compute_wrench_cone(self, contacts, object_pos):
        """Build the contact wrench cone at the object center.

        For each contact, linearize the friction cone and map each
        primitive force direction to a 6D wrench [f; r x f] at
        the object center.

        Returns (N, 6) array of primitive wrenches.
        """
        if not contacts:
            return np.zeros((0, 6))

        # Use the box friction coefficient
        mu = self.model.geom_friction[self.box_geom_id, 0]

        all_wrenches = []
        for contact in contacts:
            primitives = self.linearize_friction_cone(contact, mu)
            r = contact.position - object_pos  # moment arm

            for f_dir in primitives:
                torque = np.cross(r, f_dir)
                wrench = np.concatenate([f_dir, torque])
                all_wrenches.append(wrench)

        return np.array(all_wrenches)

    def evaluate_force_closure(self, primitive_wrenches):
        """Evaluate grasp quality via convex hull of primitive wrenches.

        Ferrari-Canny metric: radius of the largest ball centered at
        the origin that fits inside the convex hull. Positive means
        force closure.

        Returns (is_force_closure, ferrari_canny_metric).
        """
        if not HAS_SCIPY:
            return False, 0.0

        n = len(primitive_wrenches)
        if n < 7:  # need at least d+1=7 points for 6D hull
            return False, 0.0

        # Check rank - if wrenches are degenerate, hull will fail
        rank = np.linalg.matrix_rank(primitive_wrenches, tol=1e-8)
        if rank < 6:
            # Try force-only (3D) analysis as fallback
            return self._evaluate_force_closure_3d(primitive_wrenches[:, :3])

        try:
            hull = ConvexHull(primitive_wrenches)
        except Exception:
            return False, 0.0

        # hull.equations: (nfacets, 7), each row [A(6), b]
        # Facet equation: A @ x + b <= 0, with ||A|| = 1
        # For origin x=0: condition is b <= 0 for all facets
        offsets = hull.equations[:, -1]

        if np.all(offsets <= 1e-10):
            # Origin inside hull; Ferrari-Canny = min distance to boundary
            ferrari_canny = float(np.min(np.abs(offsets)))
            return True, ferrari_canny
        else:
            return False, 0.0

    def _evaluate_force_closure_3d(self, force_primitives):
        """Fallback: evaluate force closure in 3D force-only subspace."""
        if len(force_primitives) < 4:
            return False, 0.0
        try:
            hull = ConvexHull(force_primitives)
        except Exception:
            return False, 0.0

        offsets = hull.equations[:, -1]
        if np.all(offsets <= 1e-10):
            return True, float(np.min(np.abs(offsets)))
        return False, 0.0

    # ──────── Contact vs sensor comparison ────────

    def compare_contact_vs_sensor(self, contacts, sensor_data):
        """Compare per-finger contact wrenches with site sensor readings.

        Contact wrenches (from mj_contactForce) are negated when box is
        geom1 so we get the force ON the finger (reaction force), which
        is what the site sensor measures.

        Site sensors measure total constraint forces on the body (contacts
        + gravity + actuators + equality constraints), so a residual
        difference is expected.
        """
        comparison = {}

        for finger in self.FINGER_NAMES:
            finger_contacts = [c for c in contacts if c.finger_name == finger]
            sd = sensor_data.get(finger)

            if not finger_contacts and sd is None:
                continue

            # Sum contact forces ON THE FINGER in world frame
            contact_force = np.zeros(3)
            contact_torque = np.zeros(3)

            # Reference point: fingertip site position
            site_pos = np.zeros(3)
            if finger in self.tip_site_ids:
                site_pos = self.data.site_xpos[self.tip_site_ids[finger]].copy()

            for c in finger_contacts:
                # Reconstruct world-frame force from contact frame components
                f_world = (c.wrench[0] * c.frame[0]   # normal
                           + c.wrench[1] * c.frame[1]  # tangent1
                           + c.wrench[2] * c.frame[2])  # tangent2

                # mj_contactForce returns force on geom1.
                # If box is geom1, negate to get force on finger.
                if c.box_is_geom1:
                    f_world = -f_world

                contact_force += f_world

                # Torque contribution at the fingertip site
                r = c.position - site_pos
                contact_torque += np.cross(r, f_world)

            # Sensor wrench (already world frame)
            sensor_force = sd.force_world if sd else np.zeros(3)
            sensor_torque = sd.torque_world if sd else np.zeros(3)

            comparison[finger] = {
                'contact_force_world': contact_force,
                'contact_torque_world': contact_torque,
                'sensor_force_world': sensor_force,
                'sensor_torque_world': sensor_torque,
                'force_error': np.linalg.norm(contact_force - sensor_force),
                'torque_error': np.linalg.norm(contact_torque - sensor_torque),
                'num_contacts': len(finger_contacts),
            }

        return comparison

    # ──────── MuJoCo viewer geometry ────────

    def add_viewer_geometry(self):
        """Inject contact force arrows and closure indicator into viewer."""
        if self.viewer is None or self.current_state is None:
            return

        scn = self.viewer.user_scn
        scn.ngeom = 0

        state = self.current_state
        force_scale = 0.02  # meters per Newton

        # Draw contact force arrows (force ON the finger at contact point)
        for contact in state.contacts:
            if scn.ngeom >= scn.maxgeom:
                break

            color = self.FINGER_COLORS.get(
                contact.finger_name, np.array([0.5, 0.5, 0.5, 0.9]))

            # Reconstruct world-frame force on finger
            f_world = (contact.wrench[0] * contact.frame[0]
                       + contact.wrench[1] * contact.frame[1]
                       + contact.wrench[2] * contact.frame[2])
            if contact.box_is_geom1:
                f_world = -f_world
            f_mag = np.linalg.norm(f_world)

            if f_mag < 1e-6:
                continue

            end = contact.position + f_world * force_scale

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                color.astype(np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.002,
                np.array(contact.position, dtype=np.float64),
                np.array(end, dtype=np.float64),
            )
            scn.geoms[scn.ngeom].rgba[:] = color
            scn.ngeom += 1

        # Draw sensor force arrows at fingertip sites (thinner, dashed-style)
        for finger, sd in state.sensor_data.items():
            if scn.ngeom >= scn.maxgeom:
                break
            if finger not in self.tip_site_ids:
                continue

            f_mag = np.linalg.norm(sd.force_world)
            if f_mag < 1e-6:
                continue

            site_pos = self.data.site_xpos[self.tip_site_ids[finger]].copy()
            end = site_pos + sd.force_world * force_scale

            color = self.FINGER_COLORS.get(
                finger, np.array([0.5, 0.5, 0.5, 0.9])).copy()
            color[3] = 0.5  # semi-transparent for sensor arrows

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                color.astype(np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.001,
                np.array(site_pos, dtype=np.float64),
                np.array(end, dtype=np.float64),
            )
            scn.geoms[scn.ngeom].rgba[:] = color
            scn.ngeom += 1

        # # Draw friction cones at contact points (if enabled)
        # if self.show_friction_cones:
        #     mu = self.model.geom_friction[self.box_geom_id, 0]
        #     cone_height = 0.015  # meters
        #     for contact in state.contacts:
        #         if scn.ngeom >= scn.maxgeom:
        #             break

        #         # Get contact frame (normal, tangent1, tangent2)
        #         normal = contact.frame[0]
        #         tangent1 = contact.frame[1]
        #         tangent2 = contact.frame[2]

        #         # Build rotation matrix from contact frame
        #         # MuJoCo pyramid has Z-axis pointing up, so we align it with contact normal
        #         # Columns: tangent1 (X), tangent2 (Y), normal (Z)
        #         R_contact = np.column_stack([tangent1, tangent2, normal])

        #         # Friction cone: base radius = height * tan(atan(mu)) = height * mu
        #         base_radius = cone_height * mu
        #         num_edges = 8  # Octagonal approximation for smoother cone

        #         # mju_encodePyramid(pyramid, height, num_edges)
        #         # Creates a pyramid with given height and number of base edges
        #         # pyramid output: [half_width_x, half_width_y, half_height, unused, unused]
        #         size = np.array([
        #             base_radius,
        #             base_radius,
        #             cone_height / 2
        #         ], dtype=np.float64)

        #         mujoco.mjv_initGeom(
        #             scn.geoms[scn.ngeom],
        #             mujoco.mjtGeom.mjGEOM_PYRAMID,
        #             size,
        #             contact.position,
        #             R_contact.flatten(),
        #         )

        #         scn.ngeom += 1

        # Force closure indicator sphere at object center
        if scn.ngeom < scn.maxgeom:
            fc_color = (np.array([0.1, 0.9, 0.1, 0.6]) if state.force_closure
                        else np.array([0.9, 0.1, 0.1, 0.6]))
            radius = 0.005 + min(state.ferrari_canny * 0.5, 0.015)

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0, 0]),
                state.object_pos,
                np.eye(3).flatten(),
                fc_color.astype(np.float32),
            )
            scn.ngeom += 1

    # ──────── Keyboard ────────

    def key_callback(self, key):
        """Handle keyboard input."""
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_P:
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
            self.running = False
        elif key == glfw.KEY_R:
            self.setup_simulation()
        elif key == glfw.KEY_S:
            self.print_status()
        elif key == glfw.KEY_C:
            self.show_contact_count_plot = not self.show_contact_count_plot
            print(f"Contact count plot: {'shown' if self.show_contact_count_plot else 'hidden'}")
        elif key == glfw.KEY_L:
            self.show_fc_local_plot = not self.show_fc_local_plot
            print(f"Local Ferrari-Canny plot: {'shown' if self.show_fc_local_plot else 'hidden'}")
        # elif key == glfw.KEY_F:
        #     self.show_friction_cones = not self.show_friction_cones
        #     print(f"Friction cones: {'shown' if self.show_friction_cones else 'hidden'}")

    def print_status(self):
        """Print detailed grasp state to console."""
        if self.current_state is None:
            print("No state available yet")
            return

        s = self.current_state
        print(f"\n{'='*70}")
        print(f"Time: {s.time:.3f}s | Contacts: {len(s.contacts)} | "
              f"Force Closure: {s.force_closure} | "
              f"Ferrari-Canny: {s.ferrari_canny:.6f}")
        print(f"Object pos: [{s.object_pos[0]:.4f}, {s.object_pos[1]:.4f}, "
              f"{s.object_pos[2]:.4f}]")

        if s.contacts:
            print(f"\n  {'Contact':>10s}  {'Body':>22s}  "
                  f"{'Normal':>8s}  {'Fric1':>8s}  {'Fric2':>8s}")
            for c in s.contacts:
                print(f"  {c.finger_name:>10s}  {c.body_name:>22s}  "
                      f"{c.normal_force:8.4f}  {c.friction_force[0]:8.4f}  "
                      f"{c.friction_force[1]:8.4f}")

        if s.comparison:
            print(f"\n  {'Finger':>8s}  {'#Con':>4s}  "
                  f"{'Contact F':>28s}  {'Sensor F':>28s}  {'Err':>8s}")
            for finger, comp in s.comparison.items():
                cf = comp['contact_force_world']
                sf = comp['sensor_force_world']
                err = comp['force_error']
                nc = comp['num_contacts']
                print(f"  {finger:>8s}  {nc:4d}  "
                      f"[{cf[0]:8.4f},{cf[1]:8.4f},{cf[2]:8.4f}]  "
                      f"[{sf[0]:8.4f},{sf[1]:8.4f},{sf[2]:8.4f}]  "
                      f"{err:8.4f}")

            print(f"\n  {'Finger':>8s}  "
                  f"{'Contact T':>28s}  {'Sensor T':>28s}  {'Err':>8s}")
            for finger, comp in s.comparison.items():
                ct = comp['contact_torque_world']
                st = comp['sensor_torque_world']
                err = comp['torque_error']
                print(f"  {finger:>8s}  "
                      f"[{ct[0]:8.5f},{ct[1]:8.5f},{ct[2]:8.5f}]  "
                      f"[{st[0]:8.5f},{st[1]:8.5f},{st[2]:8.5f}]  "
                      f"{err:8.5f}")

        print(f"{'='*70}")

    # ──────── Matplotlib visualization ────────

    def viz_thread_func(self):
        """External 4-panel matplotlib visualization in a separate thread."""
        if not HAS_MATPLOTLIB:
            return

        fig = None
        try:
            plt.ion()
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle('Inspire Hand Grasp Force Analysis', fontsize=14)

            # Create 2x3 grid for up to 6 panels
            gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

            # Fixed panels (always shown)
            ax_cone = fig.add_subplot(gs[0, 0], projection='3d')
            ax_compare = fig.add_subplot(gs[0, 1])
            ax_ts = fig.add_subplot(gs[1, 0])
            ax_quality = fig.add_subplot(gs[1, 1])

            # Optional panels (shown on demand)
            ax_contacts = fig.add_subplot(gs[0, 2])
            ax_fc_local = fig.add_subplot(gs[1, 2])

            # Initially hide optional panels
            ax_contacts.set_visible(False)
            ax_fc_local.set_visible(False)

            while self.running:
                if self.current_state is None:
                    time.sleep(0.1)
                    continue

                state = self.current_state
                history = self.history

                try:
                    # Update visibility of optional panels
                    ax_contacts.set_visible(self.show_contact_count_plot)
                    ax_fc_local.set_visible(self.show_fc_local_plot)

                    # Panel 1: Force cone (3D force subspace)
                    ax_cone.cla()
                    ax_cone.set_title(
                        f'Force Cone (FC={state.force_closure}, '
                        f'Q={state.ferrari_canny:.4f})', fontsize=10)
                    ax_cone.set_xlabel('Fx')
                    ax_cone.set_ylabel('Fy')
                    ax_cone.set_zlabel('Fz')

                    if len(state.primitive_wrenches) > 0:
                        forces = state.primitive_wrenches[:, :3]
                        fc_color = 'green' if state.force_closure else 'red'

                        # Color points by finger
                        ax_cone.scatter(forces[:, 0], forces[:, 1], forces[:, 2],
                                       c=fc_color, alpha=0.3, s=10)

                        # Draw convex hull surface
                        if HAS_SCIPY and len(forces) >= 4:
                            try:
                                hull = ConvexHull(forces)
                                for simplex in hull.simplices:
                                    tri = forces[simplex]
                                    poly = Poly3DCollection(
                                        [tri], alpha=0.08, linewidths=0.3)
                                    poly.set_facecolor(fc_color)
                                    poly.set_edgecolor(fc_color)
                                    ax_cone.add_collection3d(poly)
                            except Exception:
                                pass

                    # Mark origin
                    ax_cone.scatter(0, 0, 0, c='black', s=80, marker='x',
                                   linewidths=2, label='Origin')
                    lim = 2.0
                    ax_cone.set_xlim([-lim, lim])
                    ax_cone.set_ylim([-lim, lim])
                    ax_cone.set_zlim([-lim, lim])

                    # Panel 2: Contact vs sensor force comparison
                    ax_compare.cla()
                    ax_compare.set_title('Contact vs Sensor Force (N)', fontsize=10)

                    if state.comparison:
                        fingers = list(state.comparison.keys())
                        contact_mags = [np.linalg.norm(
                            state.comparison[f]['contact_force_world'])
                            for f in fingers]
                        sensor_mags = [np.linalg.norm(
                            state.comparison[f]['sensor_force_world'])
                            for f in fingers]
                        x = np.arange(len(fingers))
                        w = 0.35
                        bars1 = ax_compare.bar(x - w/2, contact_mags, w,
                                              label='Contact', color='steelblue')
                        bars2 = ax_compare.bar(x + w/2, sensor_mags, w,
                                              label='Sensor', color='coral')
                        ax_compare.set_xticks(x)
                        ax_compare.set_xticklabels(fingers, rotation=30)
                        ax_compare.set_ylabel('|F| (N)')
                        ax_compare.legend(fontsize=8)

                        # Annotate contact count
                        for i, f in enumerate(fingers):
                            nc = state.comparison[f]['num_contacts']
                            if nc > 0:
                                ax_compare.annotate(f'{nc}c',
                                                  (x[i] - w/2, contact_mags[i]),
                                                  ha='center', va='bottom',
                                                  fontsize=7)

                    # Panel 3: Per-finger force time series
                    ax_ts.cla()
                    ax_ts.set_title('Fingertip Force Magnitudes', fontsize=10)
                    ax_ts.set_xlabel('Time (s)')
                    ax_ts.set_ylabel('|F| (N)')

                    for finger in self.FINGER_NAMES:
                        color = self.FINGER_COLORS_RGB.get(finger, (0.5, 0.5, 0.5))
                        if finger in history.contact_forces:
                            n = len(history.contact_forces[finger])
                            t = history.times[:n]
                            ax_ts.plot(t, history.contact_forces[finger],
                                      color=color, linewidth=1.5,
                                      label=f'{finger} (contact)')
                        if finger in history.sensor_forces:
                            n = len(history.sensor_forces[finger])
                            t = history.times[:n]
                            ax_ts.plot(t, history.sensor_forces[finger],
                                      color=color, linewidth=1, linestyle='--',
                                      alpha=0.6, label=f'{finger} (sensor)')

                    ax_ts.legend(fontsize=6, ncol=2, loc='upper left')

                    # Panel 4: Ferrari-Canny quality metric
                    ax_quality.cla()
                    ax_quality.set_title('Grasp Quality (Ferrari-Canny)',
                                       fontsize=10)
                    ax_quality.set_xlabel('Time (s)')
                    ax_quality.set_ylabel('Metric')

                    if history.times:
                        t = np.array(history.times)
                        fc = np.array(history.ferrari_canny)
                        ax_quality.plot(t, fc, 'b-', linewidth=1.5)
                        ax_quality.axhline(y=0, color='r', linestyle='--',
                                          alpha=0.4, linewidth=0.8)

                        # Fill green above zero, red below
                        ax_quality.fill_between(
                            t, 0, fc, where=(fc > 0),
                            color='green', alpha=0.15, interpolate=True)
                        ax_quality.fill_between(
                            t, 0, fc, where=(fc <= 0),
                            color='red', alpha=0.15, interpolate=True)

                    # Panel 5: Contact count over time (optional)
                    if self.show_contact_count_plot:
                        ax_contacts.cla()
                        ax_contacts.set_title('Contact Count', fontsize=10)
                        ax_contacts.set_xlabel('Time (s)')
                        ax_contacts.set_ylabel('# Contacts')
                        if history.times:
                            t = np.array(history.times)
                            nc = np.array(history.num_contacts)
                            ax_contacts.plot(t, nc, 'k-', linewidth=2)
                            ax_contacts.fill_between(t, 0, nc, alpha=0.2, color='gray')
                            max_contacts = max(nc) if len(nc) > 0 else 5
                            ax_contacts.set_ylim([0, max_contacts + 1])
                            ax_contacts.grid(True, alpha=0.3)

                    # Panel 6: Local Ferrari-Canny (windowed view, optional)
                    if self.show_fc_local_plot:
                        ax_fc_local.cla()
                        ax_fc_local.set_title(
                            f'Ferrari-Canny (last {self.fc_local_window}s)',
                            fontsize=10)
                        ax_fc_local.set_xlabel('Time (s)')
                        ax_fc_local.set_ylabel('Metric')
                        if history.times and len(history.times) > 0:
                            t = np.array(history.times)
                            fc = np.array(history.ferrari_canny)
                            # Window filter
                            current_time = t[-1]
                            window_start = current_time - self.fc_local_window
                            mask = t >= window_start
                            t_local = t[mask]
                            fc_local = fc[mask]
                            if len(t_local) > 0:
                                ax_fc_local.plot(t_local, fc_local, 'b-',
                                               linewidth=1.5)
                                ax_fc_local.axhline(y=0, color='r', linestyle='--',
                                                  alpha=0.4)
                                ax_fc_local.fill_between(t_local, 0, fc_local,
                                    where=(fc_local > 0), color='green', alpha=0.15)
                                ax_fc_local.fill_between(t_local, 0, fc_local,
                                    where=(fc_local <= 0), color='red', alpha=0.15)
                                # Auto-scale y-axis for better visibility of small values
                                fc_min, fc_max = fc_local.min(), fc_local.max()
                                margin = max(0.001, (fc_max - fc_min) * 0.1)
                                ax_fc_local.set_ylim([fc_min - margin, fc_max + margin])
                                ax_fc_local.grid(True, alpha=0.3)

                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.1)

                except Exception as e:
                    if self.running:
                        print(f"Viz error: {e}")
                    time.sleep(0.5)

        except Exception as e:
            print(f"Matplotlib thread error: {e}")
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except:
                    pass  # Suppress errors during cleanup

    # ──────── Recording ────────

    def save_recording(self):
        """Save recorded data to npz file."""
        if not self.record_path or not self.recording_data:
            return

        print(f"Saving {len(self.recording_data)} timesteps to "
              f"{self.record_path}...")

        # Flatten for npz serialization
        times = np.array([d['time'] for d in self.recording_data])
        fc = np.array([d['force_closure'] for d in self.recording_data])
        fcm = np.array([d['ferrari_canny'] for d in self.recording_data])
        nc = np.array([d['num_contacts'] for d in self.recording_data])
        obj_pos = np.array([d['object_pos'] for d in self.recording_data])

        # Per-finger sensor wrenches
        sensor_wrenches = {}
        for finger in self.FINGER_NAMES:
            wrenches = []
            for d in self.recording_data:
                w = d['sensor_wrenches'].get(finger, np.zeros(6))
                wrenches.append(w)
            sensor_wrenches[f'sensor_wrench_{finger}'] = np.array(wrenches)

        np.savez(self.record_path,
                 times=times,
                 force_closure=fc,
                 ferrari_canny=fcm,
                 num_contacts=nc,
                 object_pos=obj_pos,
                 **sensor_wrenches)
        print(f"Saved to {self.record_path}")

    # ──────── Main loop ────────

    def run(self, show_viz=True):
        """Main entry point."""
        self.setup_simulation()

        print("\nControls:")
        print("  P     - Pause/resume")
        print("  R     - Reset to home keyframe")
        print("  S     - Print detailed status")
        print("  F     - Toggle friction cone visualization")
        print("  C     - Toggle contact count plot")
        print("  L     - Toggle local Ferrari-Canny window plot")
        print("  Q/Esc - Quit")
        print()

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=self.key_callback,
        )

        # Camera pointed at the hand/object
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 0.5
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[:] = [0.07, 0.029, 0.12]

        # Enable built-in contact visualization
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        # Start matplotlib visualization thread
        if show_viz and HAS_MATPLOTLIB:
            self.viz_thread = threading.Thread(
                target=self.viz_thread_func, daemon=True)
            self.viz_thread.start()

        last_print = 0
        try:
            while self.viewer.is_running() and self.running:
                if not self.paused:
                    self.step()
                    self.add_viewer_geometry()

                    # Periodic status
                    if (self.data.time - last_print > 2.0
                            and self.current_state
                            and self.current_state.contacts):
                        self.print_status()
                        last_print = self.data.time

                self.viewer.sync()
                time.sleep(max(self.model.opt.timestep, 0.001))

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.running = False
            if self.viz_thread:
                self.viz_thread.join(timeout=2.0)
            self.save_recording()
            print("Done")


def replay_recording(npz_path):
    """Replay a recorded session from .npz file with matplotlib visualization."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for replay")
        return

    print(f"Loading recording from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    times = data['times']
    force_closure = data['force_closure']
    ferrari_canny = data['ferrari_canny']
    num_contacts = data['num_contacts']

    # Extract per-finger sensor forces
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    finger_colors = {
        "thumb":  (1.0, 0.2, 0.2),
        "index":  (0.2, 0.4, 1.0),
        "middle": (0.2, 0.8, 0.2),
        "ring":   (0.8, 0.2, 0.8),
        "pinky":  (1.0, 0.6, 0.1),
    }
    sensor_wrenches = {}
    for finger in finger_names:
        key = f'sensor_wrench_{finger}'
        if key in data:
            sensor_wrenches[finger] = data[key]

    print(f"Loaded {len(times)} timesteps")
    print(f"  Time range: {times[0]:.3f} - {times[-1]:.3f}s")
    print(f"  Force closure: {np.sum(force_closure)}/{len(force_closure)} steps")
    print(f"  Mean contacts: {np.mean(num_contacts):.1f}")

    # Create matplotlib figure
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Replay: {Path(npz_path).name}', fontsize=14)

    ax_fc = axes[0, 0]
    ax_forces = axes[0, 1]
    ax_ts = axes[1, 0]
    ax_quality = axes[1, 1]

    # Panel 1: Force closure over time
    ax_fc.set_title('Force Closure', fontsize=10)
    ax_fc.set_xlabel('Time (s)')
    ax_fc.set_ylabel('Force Closure')
    ax_fc.fill_between(times, 0, force_closure.astype(float),
                       where=force_closure, color='green', alpha=0.3,
                       label='Force Closure')
    ax_fc.fill_between(times, 0, (~force_closure).astype(float),
                       where=~force_closure, color='red', alpha=0.3,
                       label='No Closure')
    ax_fc.set_ylim([-0.1, 1.1])
    ax_fc.legend(fontsize=8)

    # Panel 2: Force magnitudes per finger (final state)
    ax_forces.set_title('Sensor Force Magnitudes (Final)', fontsize=10)
    ax_forces.set_ylabel('|F| (N)')
    final_mags = []
    for finger in finger_names:
        if finger in sensor_wrenches and len(sensor_wrenches[finger]) > 0:
            final_wrench = sensor_wrenches[finger][-1]
            mag = np.linalg.norm(final_wrench[:3])
            final_mags.append(mag)
        else:
            final_mags.append(0)
    bars = ax_forces.bar(finger_names, final_mags,
                         color=[finger_colors.get(f, (0.5, 0.5, 0.5))
                                for f in finger_names])
    ax_forces.set_xticklabels(finger_names, rotation=30)

    # Panel 3: Per-finger force time series
    ax_ts.set_title('Fingertip Force Magnitudes', fontsize=10)
    ax_ts.set_xlabel('Time (s)')
    ax_ts.set_ylabel('|F| (N)')
    for finger in finger_names:
        if finger in sensor_wrenches:
            wrenches = sensor_wrenches[finger]
            forces = np.linalg.norm(wrenches[:, :3], axis=1)
            color = finger_colors.get(finger, (0.5, 0.5, 0.5))
            ax_ts.plot(times[:len(forces)], forces, color=color,
                      linewidth=1.5, label=finger)
    ax_ts.legend(fontsize=8, ncol=2, loc='upper left')

    # Panel 4: Ferrari-Canny quality metric
    ax_quality.set_title('Grasp Quality (Ferrari-Canny)', fontsize=10)
    ax_quality.set_xlabel('Time (s)')
    ax_quality.set_ylabel('Metric')
    ax_quality.plot(times, ferrari_canny, 'b-', linewidth=1.5)
    ax_quality.axhline(y=0, color='r', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_quality.fill_between(times, 0, ferrari_canny,
                           where=(ferrari_canny > 0),
                           color='green', alpha=0.15, interpolate=True)
    ax_quality.fill_between(times, 0, ferrari_canny,
                           where=(ferrari_canny <= 0),
                           color='red', alpha=0.15, interpolate=True)

    # Add contact count on secondary axis
    ax_nc = ax_quality.twinx()
    ax_nc.plot(times, num_contacts, 'k-', alpha=0.3, linewidth=0.8)
    ax_nc.set_ylabel('# Contacts', color='gray', fontsize=8)
    ax_nc.tick_params(axis='y', labelcolor='gray', labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)
    print("Close the plot window to exit")


def main():
    parser = argparse.ArgumentParser(
        description='Finger Force Visualization for Inspire Hand')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to scene XML (default: inspire_scene.xml)')
    parser.add_argument('--record', type=str, default=None,
                        help='Path to save recording (.npz)')
    parser.add_argument('--replay', type=str, default=None,
                        help='Replay from recorded .npz file')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable external matplotlib visualization')
    parser.add_argument('--cone-edges', type=int, default=8,
                        help='Friction cone linearization edges (default: 8)')
    args = parser.parse_args()

    if args.replay:
        replay_recording(args.replay)
        return

    viz = FingerForceVisualizer(
        xml_path=args.model,
        friction_cone_edges=args.cone_edges,
        record_path=args.record,
    )
    viz.run(show_viz=not args.no_viz)


if __name__ == '__main__':
    main()
