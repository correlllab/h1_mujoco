# Polygon P: Fingertip Polygon Control System

**A 7-DOF control interface for the polygon formed by dexterous hand fingertips**

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Control Interface](#control-interface)
- [Inverse Kinematics](#inverse-kinematics)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Research Context](#research-context)
- [Testing](#testing)

---

## Overview

This system enables visualization and control of **polygon P**, the geometric shape formed by connecting the 5 fingertips of the Inspire dexterous hand in order:

**Thumb → Index → Middle → Ring → Pinky → (back to Thumb)**

The polygon serves as a high-level control primitive for grasp planning and manipulation, allowing you to specify:
- **Surface area** (5-40 cm²)
- **Position** (x, y, z)
- **Orientation** (roll, pitch, yaw)
- **Planarity** (polygon must be flat)

### Key Features
- ✅ **7 DOF control**: 1 area + 6 pose
- ✅ **Automatic IK**: Computes finger + world actuations
- ✅ **Planarity constraint**: Ensures fingertips lie in same plane
- ✅ **Real-time visualization**: MuJoCo viewer integration
- ✅ **19 actuators**: 12 low-level + 7 high-level

---

## Quick Start

### 1. Basic Visualization
```bash
cd /home/humanoid/Programs/h1_mujoco/inspire

# View current polygon state
python polygon_control.py
```

### 2. High-Level Control (Recommended)
```bash
# Interactive control with automatic IK
python polygon_high_level_control.py

# Keyboard controls:
#   A/Z - Increase/decrease area
#   W/S/Q/E - Move position
#   I/K/J/L/U/O - Change orientation
```

### 3. Test Area Control
```bash
# See planarity constraint in action
python tests/test_planarity.py

# Diagnose IK performance
python tests/diagnose_ik.py
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ USER CONTROL                                                │
│  Set ctrl[12-18] = desired polygon (area, pos, orientation)│
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ PYTHON IK CONTROLLER (polygon_high_level_control.py)       │
│  • Reads ctrl[12-18]                                        │
│  • Computes inverse kinematics                              │
│  • Enforces planarity constraint                            │
│  • Sets ctrl[0-11] → finger + world actuations              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ MUJOCO SIMULATION                                           │
│  • Actuators move hand                                      │
│  • Fingers form polygon P                                   │
│  • Real-time visualization                                  │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
inspire/
├── polygon_grasp.xml              # MuJoCo scene (19 actuators)
├── polygon_control.py             # Base controller + IK solver
├── polygon_high_level_control.py  # High-level control interface
├── README_POLYGON.md              # This file
└── tests/
    ├── test_planarity.py          # Test planarity constraint
    ├── diagnose_ik.py             # IK diagnostic tool
    ├── test_polygon_area_control.py  # Interactive area control demo
    ├── demo_ik_pipeline.py        # Pipeline demonstration
    └── polygon_control_demo.py    # Manual control demo
```

---

## Control Interface

### Actuator Mapping

**Total: 19 actuators**

#### Low-Level Actuators (0-11) - Set by IK automatically
```
ctrl[0-5]   = World frame: x, y, z, roll, pitch, yaw
ctrl[6-11]  = Fingers: pinky, ring, middle, index, thumb_prox, thumb_yaw
```

#### High-Level Actuators (12-18) - **You control these**
```
ctrl[12] = polygon_surface_area   [0.0005, 0.004] m²  (5-40 cm²)
ctrl[13] = polygon_x              [-0.5, 0.5] m
ctrl[14] = polygon_y              [-0.5, 0.5] m
ctrl[15] = polygon_z              [0.0, 0.5] m
ctrl[16] = polygon_rx             [-π, π] rad
ctrl[17] = polygon_ry             [-π, π] rad
ctrl[18] = polygon_rz             [-π, π] rad
```

### Python API

```python
import mujoco, numpy as np
from polygon_high_level_control import HighLevelPolygonController

# Load model
model = mujoco.MjModel.from_xml_path('polygon_grasp.xml')
data = mujoco.MjData(model)

# Create controller (IK runs at 10 Hz by default)
controller = HighLevelPolygonController(model, data, ik_frequency=10.0)

# Set desired polygon state
controller.set_desired_polygon_state(
    area=0.002,                      # 20 cm²
    pos=np.array([0.3, 0.4, 0.2]),   # meters
    ori=np.array([0, 0.3, 0])        # roll, pitch, yaw (rad)
)

# Simulation loop
while True:
    controller.update()  # Runs IK periodically
    mujoco.mj_step(model, data)

    # Check actual state
    state = controller.controller.compute_polygon_state()
    print(f"Area: {state.area*1e4:.2f} cm²")
```

### Direct Control (C++/other languages)

```cpp
// Set desired state via actuators
data->ctrl[12] = 0.002;  // area: 20 cm²
data->ctrl[13] = 0.3;    // x: 0.3 m
data->ctrl[14] = 0.4;    // y: 0.4 m
data->ctrl[15] = 0.2;    // z: 0.2 m
data->ctrl[16] = 0.0;    // roll: 0 rad
data->ctrl[17] = 0.3;    // pitch: 0.3 rad
data->ctrl[18] = 0.0;    // yaw: 0 rad

// Python controller must be running to compute IK
```

---

## Inverse Kinematics

### Optimization Formulation

The IK solver minimizes a weighted least-squares objective:

```
minimize:  w_area * (A_desired - A_actual)² +
          w_pose * ||pose_desired - pose_actual||² +
          w_planar * Σ|n · (p_i - centroid)|² +
          w_reg * ||q - q_0||²

subject to: joint limits
```

Where:
- `q = [world_pose(6), finger_joints(6)]` - 12 decision variables
- `w_area` = Weight for area error (default: 10000)
- `w_pose` = Weight for pose error (default: 100)
- `w_planar` = Weight for planarity constraint (default: 5000)
- `w_reg` = Regularization weight (default: 0.001)

### Planarity Constraint ⭐

**Critical feature**: The polygon MUST be planar (all fingertips in same plane).

The planarity error measures deviation from the best-fit plane:
```
d_i = |n · (p_i - centroid)|
```

With `w_planar=5000`, typical results:
- **RMS deviation**: < 1mm (excellent)
- **Max deviation**: < 2mm
- All vertices marked ✓ Planar

### Tuning Parameters

In `polygon_high_level_control.py`:

```python
world_ctrl, finger_ctrl = controller.inverse_kinematics_full(
    desired_area=area,
    desired_centroid=pos,
    desired_orientation=ori,
    w_area=10000.0,   # ↑ Higher = prioritize area matching
    w_pose=100.0,     # ↑ Higher = prioritize pose matching
    w_planar=5000.0,  # ↑ Higher = stricter planarity (< 1mm)
    w_reg=0.001,      # ↓ Lower = allow larger movements
    max_iters=100,    # More iterations = better convergence
)
```

### Performance

- **IK solve time**: 100-500ms (depending on complexity)
- **Default frequency**: 10 Hz (configurable via `ik_frequency` parameter)
- **Typical convergence**: <15% area error, <5mm position error, <1mm planarity

---

## Usage Examples

### Example 1: Resize Polygon While Maintaining Position

```python
from polygon_high_level_control import HighLevelPolygonController

# ... setup model, data, controller ...

# Get current position
current_pos = controller.controller.compute_polygon_state().centroid

# Sequence through different areas
for area_cm2 in [15, 20, 25, 30]:
    area_m2 = area_cm2 / 1e4
    controller.set_desired_polygon_state(
        area=area_m2,
        pos=current_pos,
        ori=np.zeros(3)
    )

    # Simulate for 3 seconds
    for _ in range(int(3.0 / model.opt.timestep)):
        controller.update()
        mujoco.mj_step(model, data)
```

### Example 2: Move Polygon in Workspace

```python
# Define trajectory
waypoints = [
    np.array([0.2, 0.3, 0.15]),
    np.array([0.3, 0.4, 0.20]),
    np.array([0.4, 0.5, 0.25]),
]

# Maintain 15 cm² area while moving
for pos in waypoints:
    controller.set_desired_polygon_state(
        area=0.0015,  # 15 cm²
        pos=pos,
        ori=np.zeros(3)
    )
    time.sleep(3)  # Wait for convergence
```

### Example 3: Orient Polygon

```python
# Rotate polygon to different orientations
orientations = [
    (0, 0, 0),           # Flat
    (0, 45, 0),          # Pitched 45°
    (0, 0, 90),          # Rotated 90°
]

current_state = controller.controller.compute_polygon_state()

for roll, pitch, yaw in orientations:
    ori_rad = np.deg2rad([roll, pitch, yaw])
    controller.set_desired_polygon_state(
        area=current_state.area,
        pos=current_state.centroid,
        ori=ori_rad
    )
    time.sleep(3)
```

---

## Technical Details

### Polygon Definition

The polygon is formed by connecting 5 fingertip site positions:
- `right_thumb_tip`
- `right_index_tip`
- `right_middle_tip`
- `right_ring_tip`
- `right_pinky_tip`

### Area Calculation

Uses **Shoelace formula** on best-fit plane projection:

1. Compute centroid: `c = mean(vertices)`
2. Center vertices: `v_i' = v_i - c`
3. PCA to find best-fit plane normal (smallest eigenvector)
4. Project to 2D plane coordinates
5. Apply Shoelace: `A = 0.5 * |Σ(x_i*y_{i+1} - x_{i+1}*y_i)|`

### Achievable Range

Based on hand kinematics:
- **Minimum area**: ~5 cm² (fingers fully closed)
- **Maximum area**: ~40 cm² (fingers fully open)
- **Recommended range**: 10-30 cm²

### Why 7 DOF?

The polygon has exactly 7 degrees of freedom:
- 1 DOF: Surface area (scalar property)
- 3 DOF: Position of centroid (x, y, z)
- 3 DOF: Orientation of plane (roll, pitch, yaw)

The hand has 16 low-level DOF (6 world + 10 fingers), so the system is **over-actuated** (16 > 7). The IK solver chooses the solution closest to the current configuration via regularization.

### Visualization

Real-time visual feedback in MuJoCo viewer:
- **Cyan sphere**: Desired polygon center (from ctrl[13-15])
- **Magenta sphere**: Actual polygon center (from fingertips)
- **Colored edges**: Polygon formed by fingertips
  - Thumb-Index: Red
  - Index-Middle: Blue
  - Middle-Ring: Green
  - Ring-Pinky: Magenta
  - Pinky-Thumb: Orange
- **Green arrow**: Normal vector (perpendicular to polygon plane)

---

## Research Context

### Grasp Polygons & Geometric Primitives

- **[Grasp polyhedrons](https://www.researchgate.net/figure/The-grasp-polygon-blue-lines-formed-with-the-thumb-index-and-middle-finger-contact_fig3_258328329)**: Algorithms derive 3D grasp polyhedrons from fingertip coordinates, with areas used for grasp classification
- **[Shape primitives](https://www.cs.columbia.edu/~allen/PAPERS/grasp.plan.ra03.pdf)**: Automatic grasp planning using primitive models with quality-sorted candidate grasps
- **[Convex models](https://www.researchgate.net/publication/224318412_Fast_grasp_planning_for_handarm_systems_based_on_convex_model)**: Fast grasp planning using grasping rectangular convex (GRC) and object convex polygon (OCP)

### Multi-Fingered IK & Planning

- **[Reachability-aware planning](https://link.springer.com/article/10.1007/s10846-023-01829-y)**: Computing candidate contact points with finger IK tests for reachability
- **[Force-closure conditions](https://pmc.ncbi.nlm.nih.gov/articles/PMC11505224/)**: 3D force-closure transformed to 2D plane conditions via geometric analysis
- **[Adaptive motion planning](https://arxiv.org/html/2401.11977v2)**: Force feedback for multi-fingered functional grasp

### Recent Advances (2024-2026)

- **[High-resolution tactile](https://www.nature.com/articles/s42256-025-01053-3)**: F-TAC Hand with 0.1mm spatial resolution across 70% of surface area
- **[Palm-finger coordination](https://www.nature.com/articles/s41467-025-57741-6)**: Soft robotic hand with tactile palm-finger interaction
- **[Fine-grained contact maps](https://www.tandfonline.com/doi/abs/10.1080/01691864.2025.2524553)**: GrainGrasp predicts contact maps for each fingertip
- **[Grasp control survey](https://link.springer.com/article/10.1186/s10033-025-01346-z)**: Precision control via multimodal fusion identified as future hotspot

---

## Testing

All test files are located in the `tests/` directory.

### Test Planarity Constraint
```bash
python tests/test_planarity.py
```
Measures how flat the polygon is. With `w_planar=5000`:
- RMS deviation: < 1mm (excellent)
- All vertices within 1mm of plane

### Diagnose IK Performance
```bash
python tests/diagnose_ik.py
```
Tests:
1. Manual finger control (verify fingers affect area)
2. IK with fixed position
3. IK with position change

### Interactive Area Control Demo
```bash
python tests/test_polygon_area_control.py
```
Cycles through different target areas automatically:
- 30 cm² → 20 cm² → 40 cm² → 15 cm² → 25 cm²

### IK Pipeline Demo
```bash
python tests/demo_ik_pipeline.py
```
Text-only demo showing the complete IK pipeline without viewer.

### Manual Control Demo
```bash
python tests/polygon_control_demo.py
```
Interactive demo with keyboard control of low-level actuators.

---

## Troubleshooting

### IK doesn't converge
- Target may be out of workspace - try positions closer to hand
- Increase `max_iters` (e.g., 200)
- Reduce `w_area` if area constraint is too strict
- Check that target area is in achievable range (5-40 cm²)

### IK is too slow
- Reduce IK frequency: `ik_frequency=5.0`
- Reduce `max_iters` to 50-75
- Use `use_ik=False` to test without IK

### Polygon not planar enough
- Increase `w_planar` (e.g., 10000 for sub-mm precision)
- Check that fingers aren't hitting joint limits
- Verify fingertip sites are correctly defined

### Area doesn't match target
- Increase `w_area` (e.g., 20000)
- Reduce `w_planar` slightly if it's over-constrained
- Some area/pose combinations may be geometrically impossible
- Try allowing position to change (easier than fixed position)

### Fingers don't move
- Ensure `controller.update()` is called in simulation loop
- Check that IK frequency matches step rate
- Verify `use_ik=True` when creating controller

---

## Future Work

- **Trajectory optimization**: Plan smooth polygon motions over time
- **Contact-aware control**: Maintain forces while reshaping polygon
- **Learning-based IK**: Neural network for faster approximate solutions
- **Bimanual coordination**: Control two hands with two polygons
- **Tactile feedback integration**: Use force/torque sensors for closed-loop control
- **Wrench analysis**: Integrate with Ferrari-Canny force closure metrics

---

## Citation

If you use this polygon control system in your research, please cite the relevant papers listed in the Research Context section.

---

**Created**: 2026-02-13
**Framework**: MuJoCo, scipy
**Model**: Inspire Dexterous Hand
**Authors**: Claude Code (Anthropic)
