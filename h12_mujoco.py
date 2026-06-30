import time
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface


CL_ASSETS_MUJOCO = (
    Path(__file__).resolve().parent / "submodules" / "CL_Assets" / "mujoco_assets"
)

SCENE_PATHS = {
    ("handless", False): CL_ASSETS_MUJOCO / "scene_h1_2_handless.xml",
    ("handless", True): CL_ASSETS_MUJOCO / "scene_h1_2_handless_pelvis_fixed.xml",
    ("magpie", False): CL_ASSETS_MUJOCO / "scene_h1_2_magpie.xml",
    ("magpie", True): CL_ASSETS_MUJOCO / "scene_h1_2_magpie_pelvis_fixed.xml",
}


def sim_loop(fixed=False, force_links=None, model="magpie"):
    """
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    Includes end-effector force interface for human interaction simulation
    """
    # initialize mujoco environment
    scene_path = SCENE_PATHS[(model, fixed)]
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    mujoco_env = MujocoEnv(str(scene_path))
    if not fixed:
        mujoco_env.init_elastic_band("torso_link")
    # initialize sdk interface
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    # initialize external force interface if link names are provided
    if force_links:
        mujoco_env.init_external_force(force_links, magnitude=5.0)

    # # define body name & id for wrench calculation (optional)
    # body_name = 'left_wrist_yaw_link'
    # body_id = mujoco_env.model.body(body_name).id

    # # pyvista visualization (optional)
    # pv_interface = PVInterface(mujoco_env.model, mujoco_env.data)
    # pv_interface.track_body(body_name)

    print("Press P to pause/resume the simulation loop")

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        start_time = time.time()

        if mujoco_env.paused:
            # keep viewer responsive while paused
            mujoco_env.viewer.sync()
        else:
            mujoco_env.sim_step()

        # # optional: get and display wrench on end-effector
        # force, torque = mujoco_env.get_body_wrench(body_id)
        # print(f'Force: {force}, Torque: {torque}')

        # # optional: update pyvista visualization
        # pv_interface.update_vector(force)
        # pv_interface.pv_render()

        time.sleep(max(0, mujoco_env.timestep - (time.time() - start_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1-2 Mujoco simulation")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use pelvis-fixed scene without elastic band",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--handless",
        dest="model",
        action="store_const",
        const="handless",
        help="Load handless model scene files from CL_Assets",
    )
    model_group.add_argument(
        "--magpie",
        dest="model",
        action="store_const",
        const="magpie",
        help="Load magpie-hand model scene files from CL_Assets (default)",
    )
    parser.set_defaults(model="magpie")
    parser.add_argument(
        "--force",
        nargs="+",
        default=None,
        help="Enable external force interface for specified link names",
    )
    args = parser.parse_args()

    sim_loop(fixed=args.fixed, force_links=args.force, model=args.model)
