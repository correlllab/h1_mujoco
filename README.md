# h1_mujoco

MuJoCo simulation for the H1 robot

## Installation

- Install Python dependencies from environment.yml:
    ```bash
    conda env create -f environment.yml
    # mamba env create -f environment.yml
    ```

- Install the Unitree Python SDK from here.

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
