# h1_mujoco

MuJoCo simulation for the H1_2 robot.

## Installation

- Clone the Unitree Python SDK from this [repo](https://github.com/unitreerobotics/unitree_sdk2_python).
- Install python dependencies from `environment.yml` using [`conda`](https://github.com/conda-forge/miniforge):

    ```bash
    conda env create -f environment.yml
    conda activate sim_env
    cd PATH_TO_UNITREE_SDK
    pip install -e .
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
