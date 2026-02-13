# Finger Force Visualization for Inspire Hand

Tracks contact forces, fingertip sensor wrenches, and evaluates grasp stability as the Inspire robotic hand grasps a hanging box.

## Features

### Contact Detection & Analysis
- Detects box-finger contacts using body-based mapping (works with unnamed collision geoms)
- Extracts 6D contact wrenches via `mj_contactForce`
- Properly handles Newton's third law (force on finger vs force on box)

### Fingertip Force/Torque Sensors
- 10 sensors total: 5 force + 5 torque (one pair per finger)
- Reads sensor data and transforms from site frame to world frame
- Compares contact-based wrenches vs sensor-based wrenches
  - **Expected difference**: sensors measure total constraint forces (contacts + gravity + actuators), while contact wrenches measure only contact contributions
  - Typical error: 0.03-0.05 N for fingers in contact

### Force Closure Analysis
- Linearizes friction cones into polyhedral cones (default: 8 edges per contact)
- Maps primitive forces to 6D wrenches at object center
- Computes convex hull of primitive wrenches using `scipy.spatial.ConvexHull`
- **Ferrari-Canny metric**: radius of largest inscribed ball at origin
  - Positive → force closure (grasp is stable)
  - Zero/negative → no force closure
- Falls back to 3D force-only analysis if 6D wrenches are rank-deficient

### Visualization

#### MuJoCo Viewer (3D)
- **Contact force arrows**: Color-coded by finger, drawn at contact points (solid, force ON finger)
- **Sensor force arrows**: Semi-transparent, drawn at fingertip sites (dashed-style)
- **Force closure indicator**: Green/red sphere at object center
  - Green = force closure achieved
  - Red = no force closure
  - Size scales with Ferrari-Canny metric

#### Matplotlib (4-panel window)
1. **Top-left: Force Cone**
   - 3D scatter of primitive wrench force components
   - Convex hull surface (green if force closure, red otherwise)
   - Origin marked with black X

2. **Top-right: Contact vs Sensor**
   - Grouped bar chart per finger
   - Blue bars = contact-based forces
   - Orange bars = sensor-based forces
   - Annotations show contact count

3. **Bottom-left: Time Series**
   - Per-finger force magnitudes over time
   - Solid lines = contact forces
   - Dashed lines = sensor forces
   - Color-coded by finger

4. **Bottom-right: Grasp Quality**
   - Ferrari-Canny metric over time (blue line)
   - Green fill = force closure region
   - Red fill = no closure region
   - Gray line (secondary axis) = number of contacts

### Recording & Replay
- `--record data.npz` saves timestep history
- `--replay data.npz` loads and visualizes recorded sessions
- Recorded data includes:
  - Time, force closure status, Ferrari-Canny metric
  - Number of contacts, object position
  - Per-finger sensor wrenches (6D: force + torque)

## Usage

```bash
# Live simulation
cd /home/humanoid/Programs/h1_mujoco/inspire
python finger_force_viz.py

# Record a session
python finger_force_viz.py --record my_grasp.npz

# Replay from file
python finger_force_viz.py --replay my_grasp.npz

# Adjust friction cone resolution
python finger_force_viz.py --cone-edges 12

# Disable external matplotlib (MuJoCo viewer only)
python finger_force_viz.py --no-viz
```

## Controls (Live Mode)

| Key | Action |
|-----|--------|
| **P** | Pause/resume simulation |
| **R** | Reset to home keyframe |
| **S** | Print detailed status to console |
| **Q** / **Esc** | Quit |

## Files Modified

### `inspire_scene.xml`
Added 10 fingertip force/torque sensors (replacing empty `<force/>` and `<torque/>` tags):

```xml
<sensor>
  <force name="thumb_tip_force" site="right_thumb_tip"/>
  <force name="index_tip_force" site="right_index_tip"/>
  <!-- ... 3 more fingers ... -->
  <torque name="thumb_tip_torque" site="right_thumb_tip"/>
  <torque name="index_tip_torque" site="right_index_tip"/>
  <!-- ... 3 more fingers ... -->
</sensor>
```

### `finger_force_viz.py` (new)
- 1,170 lines
- Main classes:
  - `FingerContact`: contact state (position, frame, wrench, finger name)
  - `FingerSensorData`: sensor reading (force/torque in site and world frames)
  - `GraspState`: complete snapshot (contacts, sensors, comparison, force closure, Ferrari-Canny)
  - `TimeSeriesBuffer`: rolling buffer for time-series plotting
  - `FingerForceVisualizer`: main class (simulation, analysis, visualization)
- Key methods:
  - `detect_contacts()`: scan `data.contact`, filter box-finger pairs
  - `read_sensors()`: read F/T sensors, transform to world frame
  - `compute_wrench_cone()`: friction cone → primitive wrenches
  - `evaluate_force_closure()`: ConvexHull → Ferrari-Canny metric
  - `compare_contact_vs_sensor()`: per-finger wrench comparison
  - `add_viewer_geometry()`: inject arrows/spheres into MuJoCo viewer
  - `viz_thread_func()`: matplotlib 4-panel in separate thread
  - `save_recording()`, `replay_recording()`: record/playback

## Implementation Notes

### Sign Convention
`mj_contactForce(model, data, i, result)` returns the force on **geom1**. When the box is geom1, we negate the result to get the force ON the finger (reaction force), which matches what the site sensors measure.

### Sensor vs Contact Discrepancy
Site-based F/T sensors measure **total constraint forces** on the body:
- Contact forces
- Gravity
- Actuator forces
- Equality constraint forces

Contact-based wrenches measure **only contact contributions**. The residual difference is expected and informative. For fingers in contact at the home keyframe:
- Force error: ~0.03-0.05 N
- Torque error: ~0.0001-0.002 Nm

### Matplotlib Threading
The matplotlib visualization runs in a separate thread to avoid blocking the MuJoCo viewer. TkAgg backend produces harmless warnings about "main thread not in main loop" on exit — these are suppressed via `warnings.filterwarnings`. The thread has proper cleanup with try/except/finally to avoid tkinter errors.

### Force Closure Algorithm
1. For each contact, linearize friction cone: normal + μ(cos θ·t₁ + sin θ·t₂), θ ∈ [0, 2π)
2. Normalize each primitive force to unit vector
3. Map to 6D wrench at object center: w = [f; (r - r_obj) × f]
4. Compute ConvexHull of all primitive wrenches
5. Check if origin is inside: all facet offsets ≤ 0
6. Ferrari-Canny = min |offset| if inside, else 0

## Example Output

After 500 steps at home keyframe:
```
Contacts: 3, FC: True, Q: 0.002020

Contact vs Sensor comparison (both are force ON finger):
     thumb (1c):  contact=[-2.8746,-0.2615, 3.4540]  sensor=[-2.8746,-0.2615, 3.4868]  err=0.0328
     index (1c):  contact=[ 1.4259, 0.0524,-1.8896]  sensor=[ 1.4259, 0.0524,-1.8448]  err=0.0448
    middle (1c):  contact=[ 1.7731, 0.0759,-2.2789]  sensor=[ 1.7731, 0.0759,-2.2295]  err=0.0494
      ring (0c):  contact=[ 0.0000, 0.0000, 0.0000]  sensor=[-0.0000,-0.0000, 0.0448]  err=0.0448
     pinky (0c):  contact=[ 0.0000, 0.0000, 0.0000]  sensor=[-0.0000,-0.0000, 0.0353]  err=0.0353
```

Force closure achieved with 3 contacts (thumb, index, middle). The residual sensor forces on ring/pinky (~0.04 N) are from gravity and actuator preload.