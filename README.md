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

- `unitree_robots/` contains the robot description files.
- `utility/` contains useful scripts.
- `h1_2_mujoco.py` & `h1_mujoco.py` are the main simulations for the two robots.
- `mink_interface.py` provides utilities for the Mink inverse kinematics solver.
- `mujoco_env.py` & `mujoco_interface.py` provide utilities for MuJoCo simulation.
- `unitree_h1_2_interface.py` & `unitree_h1_interface.py` contains wrappers for the Unitree SDK.

## Usage

- Run `python h1_2_mujoco.py --mode sim` to simulate the robot in Mujoco and subscribe to motor commands.
- Run `python h1_2_mujoco.py --mode shadow` to subscribe to motor states and shadow the real robot in Mujoco.
