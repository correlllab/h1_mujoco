# h1_mujoco

MuJoCo simulation for the H1_2 robot.

## Installation

- This repo depends on [Unitree Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python) to communicate with the robot.
- Download Unitree SDK under `submodules/unitree_sdk2_python` by initializing git submodules:

  ```bash
  git submodule update --init --recursive
  ```

### `uv` Installation

- Easiest way to run scripts in this repo is to use [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Commands:

    ```bash
    uv sync # install dependencies for this repo including unitree sdk
    uv run PATH_TO_SCRIPT
    ```

### `pip` Installation

- `requirements.txt` lists dependencies that can be installed by `pip`.
- Commands:

    ```bash
    pip install -r requirements.txt # install dependencies for this repo including unitree sdk
    ```

## Files

- `archive/` contains old implementations not in use.
- `submodules/CL_Assets/` contains the robot and scene assets used by the simulators.
- `utility/` contains useful scripts.
- `h12_mujoco.py` is the main simulation program of h12 robot.
- `mujoco_env.py` provides utilities for MuJoCo simulation.
- `pv_interface.py` provides visualization in pyvista is needed.
- `unitree_interface.py` contains wrappers for the Unitree SDK.

## Usage

- Run `python h12_mujoco.py` to simulate the robot in Mujoco, subscribe to
    `rt/lowcmd` and publish to `rt/lowstate`.
- The simulation is a good test before deploying controllers on the real robot
    in debug mode.

### `h12_mujoco.py` arguments

- `--handless`: use handless model scenes.
- `--magpie`: use magpie model scenes (default).
- `--fixed`: use pelvis-fixed scene variant (no elastic band).
- `--force <link1> <link2> ...`: enable external force interface on given links.
- `--scene <name-or-path>`: compose a configured Magpie workcell.
- `--headless-steps N`: run without the viewer or Unitree DDS and check numerical stability.

Scene selection behavior:

- default / `--magpie`: `submodules/CL_Assets/mujoco_assets/scene_h1_2_magpie.xml`
- `--handless`: `submodules/CL_Assets/mujoco_assets/scene_h1_2_handless.xml`
- `--fixed` + default / `--magpie`: `submodules/CL_Assets/mujoco_assets/scene_h1_2_magpie_pelvis_fixed.xml`
- `--fixed` + `--handless`: `submodules/CL_Assets/mujoco_assets/scene_h1_2_handless_pelvis_fixed.xml`

### Configured Magpie Workcells

Scene YAMLs live in the local `scene/` directory. All component XMLs, original
meshes, physical metadata, and generated collision assets live in the
`submodules/CL_Assets` asset repository. The existing Magpie robot XML is not
modified by composition. Handless models do not support `--scene` yet.

```bash
uv run h12_mujoco.py --fixed --scene magpie_table
uv run h12_mujoco.py --fixed --scene magpie_peg_in_hole
uv run h12_mujoco.py --fixed --scene magpie_peg_in_hole --headless-steps 3000
```

The last command simulates three seconds at the insertion scene's 1 ms timestep.
Headless mode does not initialize DDS, PyVista, or a viewer. It checks finite
state, MuJoCo numerical-instability warnings, and elapsed simulation time so an
automatic instability reset cannot silently pass the smoke check.

`--scene` accepts the following forms, with an optional `.yaml` extension:

| Argument | Resolution |
| --- | --- |
| `magpie_peg_in_hole` | `<project>/scene/magpie_peg_in_hole.yaml` |
| `magpie_peg_in_hole.yaml` | Same |
| `scene/magpie_peg_in_hole` | `<project>/scene/magpie_peg_in_hole.yaml` |
| `./scene/magpie_peg_in_hole.yaml` | Same |
| `/absolute/path/custom` | `/absolute/path/custom.yaml` |
| `/absolute/path/custom.yaml` | Used directly |

Explicit relative paths are anchored to the project root, not the shell's
working directory. Plain filenames default to `scene/`. There is no fallback
search that might silently select another configuration.

Asset references such as `mujoco_assets/manipulation_net/manipulation_catalog.json` always resolve
under `<project>/submodules/CL_Assets`, even when an external YAML is used.
Absolute asset paths, `..` traversal, and symlinks escaping that root are rejected.
New component XMLs use the same asset-root namespace for file-backed assets.
Components are standalone; the Python builder performs their composition.

Manipulation component XMLs and the physical catalog are grouped under
`CL_Assets/mujoco_assets/manipulation_net/`: `workcell.xml`, the detailed and
functional board XMLs, five peg XMLs, and `manipulation_catalog.json`. Source
meshes and prepared collision OBJ files remain under
`CL_Assets/meshes/manipulation_net/`. The preparation utility writes regenerated
component assets to the same nested `mujoco_assets/manipulation_net/` directory.

### Table and Board Coordinates

`scene/magpie_table.yaml` provides a complete table example. The table is a
static body with a box tabletop and four box legs; it needs no mesh.

- Units are metres, kilograms, and seconds. Quaternions use `[w, x, y, z]`.
- `table.position` and `table.quaternion` locate a floor-level reference frame
  under the tabletop centre.
- `table.size` contains full X/Y dimensions, not MuJoCo half-extents.
- `table.height` is the surface height above the table reference frame.
- `table.top_thickness`, `leg_width`, and `leg_inset` control the box geometry.
- `board.position_on_table` is relative to the tabletop surface centre.
- The board mounting origin is its bottom plane at its XY centre.
- `board.quaternion` is relative to the table. Component body/site orientations
  use explicit quaternions, rather than MJCF alternate orientation attributes.
- `simulation.timestep` is preserved by `MujocoEnv`. Legacy XML loading retains
  the previous 5 ms timestep.

### Initial Peg Layouts

The supplied board is 0.32 x 0.25 m, with thickness 0.025 m. Its supplied STL uses
centimetres, so the prepared assets apply a uniform scale of `0.01`. The holes
are **blind**, with entry at local Z = 0.025 m and floor at Z = 0.003 m, giving
0.022 m insertion depth. A board resting on the table therefore retains its own
3 mm floor; no tabletop cut-out is needed for this asset.

The five canonical shape names are `l`, `rectangle`, `triangle`, `hex`, and
`cylinder`. Each has hole IDs `<shape>_0` through `<shape>_3`.

**Size IDs identify clearance rows, not scaled pegs.** Row `0` is the upper,
tightest row in the board's local XY frame; row `3` is the lowest, loosest row.
All four rows use the same physical source peg for their shape. For example, the
12 mm cylinder has approximately 0.03, 0.2, 1, and 3 mm diametral clearance across
the four rows. Hole orientations differ; the catalog and sites preserve them.

`scene/magpie_peg_in_hole.yaml` starts just one cylinder seated in level 1 hole
`cylinder_1`. All other 19 placements are commented out as examples; uncomment
the desired entries under `board.pegs` to populate additional holes. For example:

```yaml
board:
  enabled: true
  catalog: mujoco_assets/manipulation_net/manipulation_catalog.json
  collision: functional
  position_on_table: [0.0, 0.0, 0.0]
  quaternion: [1.0, 0.0, 0.0, 0.0]
  pegs:
    - id: cylinder_1
      shape: cylinder
      size: 1
      hole: cylinder_1
      insertion_depth: 0.0
```

This is a replacement board section inside a complete scene YAML, not a full
configuration by itself. Keep `version: 1` and an enabled table as in the supplied
examples. An empty `pegs: []` creates an empty board.

Omit `insertion_depth` for the calibrated seated depth, or specify a value from
0 to 0.022 m. Zero places the peg tip at the entry plane; positive depth moves
the tip inward along the hole's local negative Z axis. Unique IDs and compatible
shape/size/hole combinations are required. Duplicate hole occupancy is rejected.

The builder computes free-body poses from the hole entry frames and peg tip
frames, so translating or rotating the board also moves the initial peg layout.
Pegs remain dynamic: they are not welded, and no loop teleports them back into
their holes. Press `R` or call `MujocoEnv.reset()` to restore the configured poses,
clear velocities/controls/applied forces, and reset simulation time. `P` pauses.

Stable names for experiments are:

- `task_table`: fixed table body.
- `task_board`: fixed board body.
- `task_hole_<shape>_<size>_entry` and `_target`: calibrated board sites.
- `task_peg_<id>_peg`: dynamic peg body.
- `task_peg_<id>_freejoint`: peg state addresses.
- `task_peg_<id>_tip`: peg-tip site.

Use MuJoCo name lookup and joint addresses for peg state; do not hard-code the
robot/peg boundary. Existing robot joint, actuator, sensor, and gripper constraint
ordering is preserved.

### Left and Right Standing Workspaces

Two world-fixed fixtures position the seated `cylinder_1` top/grasp end just
above the lower Z bound and at the XY centre of the requested easy manipulation
boxes. They use the **measured standing pelvis position**, with axes assumed
perfectly aligned with global XYZ. The buried bottom tip is not the grasp target:

```bash
uv run h12_mujoco.py --scene magpie_peg_in_hole_left
uv run h12_mujoco.py --scene magpie_peg_in_hole_right
```

Use one variant at a time with the same standing controller. Each keeps only the
level-1 cylinder enabled, with all other placements commented out as examples.
The existing `magpie_peg_in_hole` scene remains unchanged. These files do not
hot-reload into an already running simulator.

The current reference is a brief read-only DDS domain 1 snapshot on 2026-11-03
at low-state tick 122542 (two packets reporting the same simulation state).
`rt/sportmodestate.position` is the torso **IMU site**, not the pelvis. The torso
and pelvis share an origin, so the pelvis position is obtained by subtracting
the IMU offset `[-0.04452, -0.01891, 0.27756]` rotated by the reported IMU
orientation. This measurement did not load a MuJoCo model or step simulation.
The reported pelvis world position was
`[-0.074005682981, -0.027101960939, 0.955172239917]` m. Its measured rotation
is used only to remove the sensor offset, then deliberately ignored for fixture
placement; the reference axes are world-aligned. Standing height was
**0.955172 m**, versus the earlier captured 0.962321 m and XML initial 1.03 m.
This snapshot is not a time-averaged or calibrated accuracy claim.

| Variant | Seated Cylinder Top Relative to Pelvis Position, World-Aligned Axes (m) | Gripper Target RPY (deg) | Table Surface World Z (m) |
| --- | --- | --- | --- |
| Left | `[0.45, 0.225, 0.053]` | `[0, 90, -60]` | 0.955172 |
| Right | `[0.45, -0.225, 0.053]` | `[0, 90, 60]` | 0.955172 |

The conversion is translation-only: `target_world = pelvis_position + target_relative`.
Both fixtures keep the board world-horizontal and use the same table height.
The 50 mm cylinder's exposed free end stays inside the position box through the
vertical extraction stroke. The compact 0.36 x 0.29 m tables leave 20 mm of support
around the board and put their near edges about 0.398 m forward of the captured
pelvis X position, rather than pushing a large tabletop toward the robot.

The tabletop is level with the captured pelvis at relative Z = 0.000 m. The
board top/hole entrance is at 0.025 m. With the cylinder seated, its buried bottom
tip is at 0.003 m and its top at 0.053 m, just above the workspace lower bound of
0.050 m. Raising it through the 22 mm insertion depth puts its bottom at the
entrance and its top at 0.075 m, leaving 150 mm below the workspace ceiling of
0.225 m. The hole entrance and buried bottom tip deliberately lie below the
grasp workspace. A small additional lift is needed
to fully separate the peg from the rim; this remaining height is planning space,
not a guarantee of gripper/arm clearance.

Both table heights equal the reported pelvis height, `0.955172239917` m.
Left/right X/Y placements were also recentered using the same new position
snapshot, while continuing to ignore pelvis rotation.

The RPY values are gripper/end-effector targets, **not board orientations**.
Under the usual `Rz(yaw) Ry(pitch) Rx(roll)` convention they point the tool's local
X downward; applying them to the board would turn its hole Z axis horizontal.
Account for the controller's wrist-to-grasp transform when commanding a grasp:
the peg tip is not automatically the wrist or grasp-frame origin.

The board and table do not move with the pelvis. If standing height or root
translation changes, recalibrate their world placements. Rotation is intentionally
excluded from this ideal calibration. The snapshotted position is documented in
each YAML. The calibration specifically
targets `cylinder_1`, not every hole. Cartesian-box checks do not certify an IK
solution, grasp, or collision-free arm trajectory.
If a real controller uses the actual rotated pelvis frame, its motion planner
must still handle that frame transform. This ideal scene calibration does not
change the robot's physical orientation or controller frame conventions.

### Collision Preparation

Collision assets are prepared offline from the supplied STL surfaces, not from a
single convex hull of the board. Source horizontal sections are constrained-
triangulated, merged only where convex, and extruded into convex prisms. This
preserves the openings, blind floors, and hole profiles. The L peg
uses multiple convex pieces; the other peg families have convex profiles. Visual
geoms are massless and noncolliding. Peg dynamics use explicit inertia from their
source cross-sections rather than double-counting visual/collision masses.

`board.collision` selects the fidelity/performance tradeoff:

- `functional` (default): 3,005 static collision geoms. Preserves all 20 original
  hole profiles, their entry openings, blind floors, and board thickness, but
  treats cosmetic engraved recesses as solid. The original visual mesh, including
  its engravings, is unchanged.
- `detailed`: 10,895 static collision geoms. Also preserves the engraved recesses
  in collisions. Use this when contact with engraved labels matters.

This switch does not change hole clearances, peg dimensions, contact parameters,
or the 1 ms insertion-scene timestep. Initial parsing of the detailed XML takes
about 11 seconds on the development machine. Preparation is not repeated at
simulator startup.

Each active peg adds free-body/contact-solving work. Prefer the one-peg default
for single-hole experiments and enable the full arrangement only when required.
Before these optimizations, headless fixed-Magpie benchmarks measured about
17 ms per physics step for 20 pegs versus 1.2 ms for one cylinder with the detailed
board. At a 1 ms timestep, 17 ms/step is only about 0.06x real time. Performance
depends on hardware, contact state, robot controller, and active peg shapes.

With the functional board, the same benchmark measured about 0.58 ms/step for
one cylinder (1.7x real time headlessly) and 8.4 ms/step for 20 pegs (0.12x real
time). Parsing fell from about 10.7 s to 0.9 s. The full arrangement can still be
too slow for real-time experiments; the one-cylinder default avoids that load.

GUI synchronization and force overlays are limited to 60 Hz using wall-clock
time, including while paused. Physics and external-force evaluation still run
every simulation step. GUI mouse/option changes can take up to one frame to
reach the simulation. Collision visualization group 3 remains hidden by MuJoCo's
default settings; hiding it does not disable physical collisions.

Install preparation/test dependencies and reproduce the assets with:

```bash
uv sync --group dev
uv run --group dev utility/prepare_manipulation_assets.py --inspect
uv run --group dev utility/prepare_manipulation_assets.py
uv run --group dev utility/prepare_manipulation_assets.py --check
uv run --group dev pytest
```

Generation never modifies the original STLs. `--check` regenerates in memory and
rejects missing or stale generated files without writing. The catalog records
source hashes, geometry tolerances, transforms, clearances, and physical
assumptions. Regenerate and rerun tests whenever the source meshes change.

The geometry follows the supplied float32 tessellation, not measured production
tolerances. Peg density is assumed uniform polymer at 1200 kg/m3; friction and
soft-contact parameters are simulation assumptions, not measured material data.
The tightest nominal fit has only about 15 micrometres radial clearance. Validate
timestep, contact stiffness, manufacturing dimensions, and measured force data
before interpreting simulation insertion/extraction forces quantitatively.
This supplies collision-ready free objects and target frames, not an autonomous
robot grasp/insertion controller.
The supplied shafts and bores are straight and blunt. A released peg is not
guaranteed to self-align or fall back into a hole: it may tip or hit the rim.
Insertion and extraction require alignment by a gripper/controller. Dynamic
vertical extraction/reinsertion tests use an explicit test-only slide guide to
isolate collision physics; production peg components retain all six free DOFs.

The STL inputs and prepared assets belong to the separate `CL_Assets` Git/LFS
repository. They must be recorded there before a parent-repository submodule
pointer can preserve them; untracked submodule files are not included by a
parent pointer change alone.
