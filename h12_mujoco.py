import time
import argparse
import numpy as np
import pyvista as pv

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface


def sim_loop(fixed=False, force_links=None, handless=False):
    """
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    Includes end-effector force interface for human interaction simulation
    """
    # initialize mujoco environment
    if fixed:
        if handless:
            scene_path = "unitree_robots/h1_2/scene_handless_pelvis_fixed.xml"
        else:
            scene_path = "unitree_robots/h1_2/scene_pelvis_fixed.xml"
        mujoco_env = MujocoEnv(scene_path)
    else:
        if handless:
            scene_path = "unitree_robots/h1_2/scene_handless.xml"
        else:
            scene_path = "unitree_robots/h1_2/scene.xml"
        mujoco_env = MujocoEnv(scene_path)
        # Lower the suspension anchor 2.0 -> 1.73 so the robot hangs ~0.27 m lower
        # (pelvis ~1.07 m, feet ~5 cm above the floor) -> small drop on SPACE-release
        # instead of the old ~0.32 m fall. (default point was [0,0,2.0].)
        mujoco_env.init_elastic_band("torso_link", point=np.array([0.0, 0.0, 1.73]))
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
    # main simulation loop -- paced to real-time CUMULATIVELY: sleep until n_steps*timestep of
    # wall-clock has elapsed since the anchor, so the occasional render frame's cost is absorbed by
    # the slack of the cheap physics-only steps (true ~1.0x, no per-step drift). Physics still steps
    # exactly once per iteration, so the sim trajectory is unchanged -- only wall-clock spacing moves.
    sim_t0 = time.time()
    n_steps = 0
    while mujoco_env.viewer.is_running():
        if mujoco_env.paused:
            # keep viewer responsive while paused; rebase the clock so we don't sprint on resume
            mujoco_env.viewer.sync()
            time.sleep(1.0 / 60.0)
            sim_t0 = time.time()
            n_steps = 0
            continue

        mujoco_env.sim_step()
        n_steps += 1

        # # optional: get and display wrench on end-effector
        # force, torque = mujoco_env.get_body_wrench(body_id)
        # print(f'Force: {force}, Torque: {torque}')

        # # optional: update pyvista visualization
        # pv_interface.update_vector(force)
        # pv_interface.pv_render()

        ahead = (sim_t0 + n_steps * mujoco_env.timestep) - time.time()
        if ahead > 0:
            time.sleep(ahead)


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
        dest="handless",
        action="store_true",
        help="Load handless model scene files (default)",
    )
    model_group.add_argument(
        "--inspire",
        dest="handless",
        action="store_false",
        help="Load inspire-hand model scene files",
    )
    # Default to handless model when no model option is provided.
    parser.set_defaults(handless=True)
    parser.add_argument(
        "--force",
        nargs="+",
        default=None,
        help="Enable external force interface for specified link names",
    )
    args = parser.parse_args()

    sim_loop(fixed=args.fixed, force_links=args.force, handless=args.handless)
