# h1_mujoco
MuJoCo simulation for the H1 robot

## Installation
- Install Python dependencies from environment.yml:
    ```bash
    conda env create -f environment.yml
    # mamba env create -f environment.yml
    ```

- Install the Unitree Python SDK from here.

## Usage

- Run `python h1_mujoco.py` to start the main simulation.
- `unitree_robots/` contains the robot description files.
- `utility/` contains useful scripts.
- `mink_interface.py` provides utilities for the Mink inverse kinematics solver.
- `mujoco_interface.py` provides utilities for MuJoCo simulation.
- `unitree_sdk_interface.py` contains wrappers for the Unitree SDK.
