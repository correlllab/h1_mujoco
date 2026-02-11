#!/usr/bin/env python3
"""
Workspace Visualization Tool for Inspire Hand

Visualizes the operational space (reachable workspace) of fingertip sites,
respecting the mechanical coupling constraints defined in MuJoCo equality elements.

Usage:
    python workspace_viz.py                          # All fingertips
    python workspace_viz.py --fingers thumb index    # Specific fingers
    python workspace_viz.py --samples 100            # Higher resolution
    python workspace_viz.py --export workspace.npz   # Export data
    python workspace_viz.py --save workspace.png     # Save to image
    python workspace_viz.py --stats-only             # Print stats only
    python workspace_viz.py --list-sites             # List available sites
    python workspace_viz.py --add-site palm right_palm thumb  # Add custom site
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.widgets import CheckButtons

# Default model path
DEFAULT_MODEL = Path(__file__).parent / "inspire" / "right_ur5_mount.xml"


@dataclass
class FingerConfig:
    """Configuration for a single finger's workspace computation."""
    name: str
    tip_site: str
    actuated_joints: list[str]
    coupled_joints: dict[str, str]  # coupled_joint -> source_joint
    joint_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    color: str = "blue"


@dataclass
class WorkspaceData:
    """Computed workspace data for a finger."""
    name: str
    points: np.ndarray  # Nx3 array of (x, y, z) positions
    joint_configs: np.ndarray  # Nx(num_joints) array of joint values


class InspireWorkspaceVisualizer:
    """Visualizes the operational space of the Inspire hand fingertips."""

    # Color scheme for fingers
    FINGER_COLORS = {
        "thumb": "#e41a1c",   # Red
        "index": "#377eb8",   # Blue
        "middle": "#4daf4a", # Green
        "ring": "#984ea3",   # Purple
        "pinky": "#ff7f00",  # Orange
    }

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        """Initialize the visualizer with a MuJoCo model.

        Args:
            model_path: Path to the MuJoCo XML model file.
        """
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Extract finger configurations from model
        self.finger_configs = self._build_finger_configs()

        # Storage for computed workspaces
        self.workspaces: dict[str, WorkspaceData] = {}

    def _build_finger_configs(self) -> dict[str, FingerConfig]:
        """Build finger configurations from the model structure."""
        configs = {}

        # Thumb: 2 DOFs (yaw + pitch), with intermediate and distal coupled to pitch
        configs["thumb"] = FingerConfig(
            name="thumb",
            tip_site="right_thumb_tip",
            actuated_joints=[
                "right_thumb_proximal_yaw_joint",
                "right_thumb_proximal_pitch_joint",
            ],
            coupled_joints={
                "right_thumb_intermediate_joint": "right_thumb_proximal_pitch_joint",
                "right_thumb_distal_joint": "right_thumb_intermediate_joint",
            },
            color=self.FINGER_COLORS["thumb"],
        )

        # Four fingers: 1 DOF each, intermediate coupled to proximal
        for finger in ["index", "middle", "ring", "pinky"]:
            configs[finger] = FingerConfig(
                name=finger,
                tip_site=f"right_{finger}_tip",
                actuated_joints=[f"right_{finger}_proximal_joint"],
                coupled_joints={
                    f"right_{finger}_intermediate_joint": f"right_{finger}_proximal_joint"
                },
                color=self.FINGER_COLORS[finger],
            )

        # Extract joint ranges from model
        for config in configs.values():
            for joint_name in config.actuated_joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    # Get joint limits
                    limited = self.model.jnt_limited[joint_id]
                    if limited:
                        range_low = self.model.jnt_range[joint_id, 0]
                        range_high = self.model.jnt_range[joint_id, 1]
                        config.joint_ranges[joint_name] = (range_low, range_high)
                    else:
                        # Default range if not limited
                        config.joint_ranges[joint_name] = (-np.pi, np.pi)

        return configs

    def list_sites(self) -> list[str]:
        """List all site names in the model."""
        sites = []
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name:
                sites.append(name)
        return sites

    def add_custom_site(
        self,
        name: str,
        site_name: str,
        finger: str,
        color: str = "#808080"
    ) -> None:
        """Add a custom site to track on an existing finger's kinematic chain.

        Args:
            name: Display name for this site configuration.
            site_name: MuJoCo site name to track.
            finger: Which finger's joints affect this site (thumb, index, etc.).
            color: Color for visualization.
        """
        if finger not in self.finger_configs:
            raise ValueError(f"Unknown finger: {finger}")

        # Copy the finger config but change the site
        base_config = self.finger_configs[finger]
        new_config = FingerConfig(
            name=name,
            tip_site=site_name,
            actuated_joints=base_config.actuated_joints.copy(),
            coupled_joints=base_config.coupled_joints.copy(),
            joint_ranges=base_config.joint_ranges.copy(),
            color=color,
        )
        self.finger_configs[name] = new_config

    def _get_joint_id(self, joint_name: str) -> int:
        """Get the joint ID for a named joint."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    def _get_site_id(self, site_name: str) -> int:
        """Get the site ID for a named site."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def _set_joint_position(self, joint_name: str, value: float) -> None:
        """Set a joint to a specific position."""
        joint_id = self._get_joint_id(joint_name)
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = value

    def _get_site_position(self, site_name: str) -> np.ndarray:
        """Get the world position of a site after forward kinematics."""
        site_id = self._get_site_id(site_name)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)

    def _apply_coupled_joints(self, config: FingerConfig) -> None:
        """Apply coupling constraints to set coupled joint values."""
        # Handle chained coupling (e.g., distal -> intermediate -> pitch)
        # We need to resolve in the right order
        resolved = set()
        max_iterations = 10

        for _ in range(max_iterations):
            made_progress = False
            for coupled_joint, source_joint in config.coupled_joints.items():
                if coupled_joint in resolved:
                    continue

                # Check if source is an actuated joint or already resolved
                source_id = self._get_joint_id(source_joint)
                if source_id >= 0:
                    qpos_adr = self.model.jnt_qposadr[source_id]
                    source_value = self.data.qpos[qpos_adr]

                    # Apply 1:1 coupling (polycoef = [0, 1, 0, 0, 0])
                    self._set_joint_position(coupled_joint, source_value)
                    resolved.add(coupled_joint)
                    made_progress = True

            if not made_progress or len(resolved) == len(config.coupled_joints):
                break

    def compute_workspace(
        self,
        finger: str,
        samples_per_dof: int = 50
    ) -> WorkspaceData:
        """Compute the workspace for a single finger.

        Args:
            finger: Name of the finger (thumb, index, middle, ring, pinky).
            samples_per_dof: Number of samples per degree of freedom.

        Returns:
            WorkspaceData containing the sampled positions and joint configs.
        """
        config = self.finger_configs[finger]
        num_dofs = len(config.actuated_joints)

        # Generate sample grid
        joint_samples = []
        for joint_name in config.actuated_joints:
            low, high = config.joint_ranges[joint_name]
            joint_samples.append(np.linspace(low, high, samples_per_dof))

        # Create meshgrid for multi-DOF case
        if num_dofs == 1:
            grid = joint_samples[0].reshape(-1, 1)
        else:
            meshes = np.meshgrid(*joint_samples, indexing='ij')
            grid = np.stack([m.flatten() for m in meshes], axis=-1)

        # Sample workspace
        positions = []
        joint_configs = []

        # Reset to home position first
        self.data.qpos[:] = 0

        for sample in grid:
            # Set actuated joints
            for i, joint_name in enumerate(config.actuated_joints):
                if num_dofs == 1:
                    self._set_joint_position(joint_name, sample[0])
                else:
                    self._set_joint_position(joint_name, sample[i])

            # Apply coupled joints
            self._apply_coupled_joints(config)

            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Get tip position
            pos = self._get_site_position(config.tip_site)
            positions.append(pos)
            joint_configs.append(sample.flatten())

        workspace = WorkspaceData(
            name=finger,
            points=np.array(positions),
            joint_configs=np.array(joint_configs),
        )

        self.workspaces[finger] = workspace
        return workspace

    def compute_all_workspaces(
        self,
        fingers: Optional[list[str]] = None,
        samples_per_dof: int = 50
    ) -> dict[str, WorkspaceData]:
        """Compute workspaces for multiple fingers.

        Args:
            fingers: List of finger names, or None for all fingers.
            samples_per_dof: Number of samples per degree of freedom.

        Returns:
            Dictionary mapping finger names to WorkspaceData.
        """
        if fingers is None:
            fingers = list(self.finger_configs.keys())

        for finger in fingers:
            print(f"Computing workspace for {finger}...")
            self.compute_workspace(finger, samples_per_dof)

        return self.workspaces

    def visualize(
        self,
        fingers: Optional[list[str]] = None,
        show_mesh: bool = False,
        alpha: float = 0.6,
        point_size: float = 2.0,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create an interactive 3D visualization of finger workspaces.

        Args:
            fingers: List of finger names to visualize, or None for all.
            show_mesh: Whether to show the hand mesh at home position.
            alpha: Transparency of workspace points.
            point_size: Size of workspace points.
            save_path: If provided, save figure to this path instead of displaying.
        """
        if fingers is None:
            fingers = list(self.finger_configs.keys())

        # Compute any missing workspaces
        for finger in fingers:
            if finger not in self.workspaces:
                self.compute_workspace(finger)

        # Create figure with space for controls
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.25)

        # Store scatter plots for toggling
        scatters = {}

        # Plot each finger's workspace
        for finger in fingers:
            if finger not in self.workspaces:
                continue

            ws = self.workspaces[finger]
            config = self.finger_configs[finger]

            scatter = ax.scatter(
                ws.points[:, 0],
                ws.points[:, 1],
                ws.points[:, 2],
                c=config.color,
                s=point_size,
                alpha=alpha,
                label=finger.capitalize(),
            )
            scatters[finger] = scatter

        # Add reference point for palm
        self.data.qpos[:] = 0
        mujoco.mj_forward(self.model, self.data)
        palm_pos = self._get_site_position("right_palm")
        ax.scatter(*palm_pos, c='black', s=50, marker='s', label='Palm')

        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Inspire Hand Fingertip Workspaces')
        ax.legend(loc='upper left')

        # Equal aspect ratio
        self._set_axes_equal(ax)

        # Add checkboxes for toggling
        checkbox_ax = plt.axes([0.02, 0.4, 0.15, 0.25])
        labels = [f.capitalize() for f in fingers]
        actives = [True] * len(fingers)
        check = CheckButtons(checkbox_ax, labels, actives)

        def toggle_visibility(label):
            finger = label.lower()
            if finger in scatters:
                scatter = scatters[finger]
                visible = not scatter.get_visible()
                scatter.set_visible(visible)
                plt.draw()

        check.on_clicked(toggle_visibility)

        # Add keyboard shortcuts
        def on_key(event):
            key_map = {'1': 'thumb', '2': 'index', '3': 'middle', '4': 'ring', '5': 'pinky'}
            if event.key in key_map:
                finger = key_map[event.key]
                if finger in scatters:
                    scatter = scatters[finger]
                    scatter.set_visible(not scatter.get_visible())
                    # Update checkbox state
                    idx = fingers.index(finger)
                    check.set_active(idx)
                    plt.draw()
            elif event.key == 'a':
                # Toggle all
                any_visible = any(s.get_visible() for s in scatters.values())
                for i, (finger, scatter) in enumerate(scatters.items()):
                    scatter.set_visible(not any_visible)
                plt.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Show statistics
        stats_text = self._get_stats_text(fingers)
        fig.text(0.02, 0.02, stats_text, fontsize=8, family='monospace',
                 verticalalignment='bottom')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D axes."""
        # Get current limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        # Compute ranges and center
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        # Set new limits
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    def _get_stats_text(self, fingers: list[str]) -> str:
        """Generate statistics text for the visualization."""
        lines = ["Workspace Statistics:", "-" * 30]

        for finger in fingers:
            if finger not in self.workspaces:
                continue
            ws = self.workspaces[finger]
            config = self.finger_configs[finger]

            # Compute bounding box
            min_pos = ws.points.min(axis=0)
            max_pos = ws.points.max(axis=0)
            extent = max_pos - min_pos

            lines.append(f"\n{finger.capitalize()}:")
            lines.append(f"  DOFs: {len(config.actuated_joints)}")
            lines.append(f"  Samples: {len(ws.points)}")
            lines.append(f"  Extent: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")

        lines.append("\n\nControls:")
        lines.append("  1-5: Toggle fingers")
        lines.append("  a: Toggle all")
        lines.append("  Mouse: Rotate/zoom")

        return "\n".join(lines)

    def export(self, filepath: str | Path) -> None:
        """Export workspace data to a numpy file.

        Args:
            filepath: Path for the output .npz file.
        """
        data = {}
        for finger, ws in self.workspaces.items():
            data[f"{finger}_points"] = ws.points
            data[f"{finger}_joints"] = ws.joint_configs

        np.savez(filepath, **data)
        print(f"Exported workspace data to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Inspire hand fingertip workspaces"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to MuJoCo XML model file"
    )
    parser.add_argument(
        "--fingers", "-f",
        nargs="+",
        choices=["thumb", "index", "middle", "ring", "pinky"],
        default=None,
        help="Fingers to visualize (default: all)"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=50,
        help="Samples per degree of freedom (default: 50)"
    )
    parser.add_argument(
        "--export", "-e",
        type=Path,
        default=None,
        help="Export workspace data to .npz file"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.6,
        help="Point transparency (default: 0.6)"
    )
    parser.add_argument(
        "--point-size", "-p",
        type=float,
        default=2.0,
        help="Point size (default: 2.0)"
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Show hand mesh at home position (not yet implemented)"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save visualization to image file instead of displaying"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics only, no visualization"
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List all available sites in the model and exit"
    )
    parser.add_argument(
        "--add-site",
        nargs=3,
        action="append",
        metavar=("NAME", "SITE", "FINGER"),
        help="Add custom site to track: NAME SITE_NAME FINGER (can be repeated)"
    )

    args = parser.parse_args()

    # Create visualizer
    print(f"Loading model from {args.model}")
    viz = InspireWorkspaceVisualizer(args.model)

    # List sites if requested
    if args.list_sites:
        print("\nAvailable sites in model:")
        for site in viz.list_sites():
            print(f"  - {site}")
        return

    # Add custom sites if specified
    custom_names = []
    if args.add_site:
        extra_colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
        for i, (name, site, finger) in enumerate(args.add_site):
            color = extra_colors[i % len(extra_colors)]
            viz.add_custom_site(name, site, finger, color)
            custom_names.append(name)
            print(f"Added custom site '{name}' tracking '{site}' on {finger}")

    # If --fingers specified, also include custom sites
    fingers_to_compute = args.fingers
    if fingers_to_compute and custom_names:
        fingers_to_compute = list(fingers_to_compute) + custom_names

    # Compute workspaces
    viz.compute_all_workspaces(fingers_to_compute, args.samples)

    # Export if requested
    if args.export:
        viz.export(args.export)

    # Print stats
    if args.stats_only:
        fingers = fingers_to_compute or list(viz.finger_configs.keys())
        print(viz._get_stats_text(fingers))
        return

    # Visualize
    viz.visualize(
        fingers=fingers_to_compute,
        show_mesh=args.show_mesh,
        alpha=args.alpha,
        point_size=args.point_size,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
