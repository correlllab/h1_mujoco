# Inspire Hand Workspace Visualization Tool

## Overview

This tool visualizes the operational space (reachable workspace) of fingertip sites on the Inspire robotic hand. It respects the mechanical coupling constraints defined in the MuJoCo equality elements.

## Hand Kinematics Summary

### Four Pointing Fingers (Index, Middle, Ring, Pinky)
- **1 effective DOF** per finger
- Proximal and intermediate joints are mechanically coupled 1:1
- Constraint: `proximal_joint == intermediate_joint`
- **Workspace shape**: Arc in the finger's plane of motion

### Thumb
- **2 effective DOFs**:
  1. **Yaw** (adduction): `right_thumb_proximal_yaw_joint` (range: -0.1 to 1.3 rad)
  2. **Pitch** (flexion): `right_thumb_proximal_pitch_joint` (range: 0 to 0.5 rad)
- Mechanical coupling:
  - `proximal_pitch == intermediate`
  - `distal == intermediate`
- **Workspace shape**: 2D surface patch (more complex than a simple arc)

## Joint Ranges

| Finger | Joint | Range (rad) | Coupled With |
|--------|-------|-------------|--------------|
| Thumb | yaw | -0.1 to 1.3 | - |
| Thumb | pitch | 0 to 0.5 | intermediate, distal |
| Index | proximal | 0 to 1.7 | intermediate |
| Middle | proximal | 0 to 1.7 | intermediate |
| Ring | proximal | 0 to 1.7 | intermediate |
| Pinky | proximal | 0 to 1.7 | intermediate |

## Sites Tracked

### Primary (Fingertips)
- `right_thumb_tip`
- `right_index_tip`
- `right_middle_tip`
- `right_ring_tip`
- `right_pinky_tip`

### Secondary (Available for future use)
- `right_palm`
- `eeff` (end effector reference)
- `track_hand_right_*_tip` sites (duplicate positions)

## Implementation Design

### Dependencies
- `mujoco` - Physics simulation and forward kinematics
- `numpy` - Numerical computations
- `matplotlib` - 3D visualization with interactive controls
- `argparse` - Command-line interface

### Core Algorithm

1. **Load Model**: Parse the MuJoCo XML and extract joint/site information
2. **Extract Constraints**: Parse equality constraints to determine coupling ratios
3. **Sample Joint Space**:
   - For 1-DOF fingers: Sample along the single actuated joint range
   - For thumb: Sample 2D grid over (yaw, pitch) space
4. **Apply Coupling**: For each sample, set coupled joints according to equality constraints
5. **Forward Kinematics**: Use `mj_forward()` to compute site positions
6. **Visualization**: Plot 3D workspace with interactive controls

### Coupling Implementation

The equality constraints use `polycoef="0 1 0 0 0"`, meaning:
```
joint2 = polycoef[0] + polycoef[1]*joint1 + polycoef[2]*joint1^2 + ...
       = 0 + 1*joint1 + 0 + 0 + 0
       = joint1
```

So all coupled joints have a 1:1 ratio.

### Visualization Features

1. **Toggle Sites**: Enable/disable visualization of individual finger workspaces
2. **Color Coding**: Different colors for each finger
3. **Transparency**: Adjustable alpha for overlapping workspaces
4. **Hand Mesh**: Optional overlay of hand mesh at reference pose
5. **Sampling Resolution**: Adjustable for performance vs. accuracy
6. **Export**: Save workspace point clouds to file

### Command-Line Interface

```bash
# Basic usage - all fingertips
python workspace_viz.py

# Specific fingers only
python workspace_viz.py --fingers thumb index

# Adjust sampling resolution
python workspace_viz.py --samples 100

# Show hand mesh overlay
python workspace_viz.py --show-mesh

# Export to file
python workspace_viz.py --export workspace_data.npz

# Save to image instead of interactive display
python workspace_viz.py --save workspace.png

# Print statistics only (no GUI)
python workspace_viz.py --stats-only

# List all available sites in the model
python workspace_viz.py --list-sites

# Add custom sites to track
python workspace_viz.py --add-site palm_ws right_palm thumb
```

### Interactive Controls

- **Checkboxes**: Toggle individual finger workspaces
- **Mouse**: Rotate, zoom, pan 3D view
- **Keyboard shortcuts**:
  - `1-5`: Toggle thumb, index, middle, ring, pinky
  - `a`: Toggle all on/off
  - `m`: Toggle mesh visibility
  - `r`: Reset view

## File Structure

```
workspace_viz.py          # Main visualization tool
WORKSPACE_VIZ.md          # This documentation
```

## Future Extensions

1. **Additional Sites**: Easy to add new sites (palm, knuckles, etc.)
2. **Collision Avoidance**: Filter workspace points that cause self-collision
3. **Reachability Maps**: Heat maps showing manipulability at each point
4. **Animation**: Animate finger motion through workspace
5. **UR5 Integration**: Include arm kinematics for full system workspace
