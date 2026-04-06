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
- `unitree_robots/` contains the robot description files.
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

- `--handless`: use handless model scenes (default).
- `--inspire`: use inspire-hand model scenes.
- `--fixed`: use pelvis-fixed scene variant (no elastic band).
- `--force <link1> <link2> ...`: enable external force interface on given links.

Scene selection behavior:

- default / `--handless`: `scene_handless.xml`
- `--inspire`: `scene.xml`
- `--fixed` + default / `--handless`: `scene_handless_pelvis_fixed.xml`
- `--fixed` + `--inspire`: `scene_pelvis_fixed.xml`
