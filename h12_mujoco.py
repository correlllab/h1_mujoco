import time
import argparse
from pathlib import Path

CL_ASSETS_MUJOCO = (
    Path(__file__).resolve().parent / 'submodules' / 'CL_Assets' / 'mujoco_assets'
)

SCENE_PATHS = {
    ('handless', False): CL_ASSETS_MUJOCO / 'scene_h1_2_handless.xml',
    ('handless', True): CL_ASSETS_MUJOCO / 'scene_h1_2_handless_pelvis_fixed.xml',
    ('magpie', False): CL_ASSETS_MUJOCO / 'scene_h1_2_magpie.xml',
    ('magpie', True): CL_ASSETS_MUJOCO / 'scene_h1_2_magpie_pelvis_fixed.xml',
}


def sim_loop(fixed=False, force_links=None, model='magpie', scene=None,
             headless_steps=None):
    '''
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    Includes end-effector force interface for human interaction simulation
    '''
    if scene is not None and model == 'handless':
        raise ValueError('--scene cannot be used with --handless; scenes require Magpie')
    if headless_steps is not None and headless_steps < 0:
        raise ValueError('--headless-steps must be non-negative')

    from mujoco_env import MujocoEnv

    # initialize mujoco environment
    if scene is not None:
        from scene_builder import load_scene_config, build_magpie_scene

        config = load_scene_config(scene)
        mujoco_env = MujocoEnv(model=build_magpie_scene(config, fixed=fixed))
    else:
        scene_path = SCENE_PATHS[(model, fixed)]
        if not scene_path.exists():
            raise FileNotFoundError(f'Scene file not found: {scene_path}')
        mujoco_env = MujocoEnv(str(scene_path))

    if not fixed:
        mujoco_env.init_elastic_band('torso_link')

    # initialize external force interface if link names are provided
    if force_links:
        mujoco_env.init_external_force(force_links, magnitude=5.0)

    if headless_steps is not None:
        import mujoco
        import numpy as np

        numerical_warnings = (
            mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
        )
        for _ in range(headless_steps):
            mujoco_env.sim_step()
            for warning in numerical_warnings:
                if mujoco_env.data.warning[warning].number:
                    raise RuntimeError(f'Headless simulation instability: {warning.name}')
        finite = all(np.isfinite(state).all() for state in (
            mujoco_env.data.qpos, mujoco_env.data.qvel,
            mujoco_env.data.qacc, mujoco_env.data.ctrl,
        )) and np.isfinite(mujoco_env.data.time)
        if not finite:
            raise RuntimeError('Headless simulation produced non-finite state')
        expected_time = headless_steps * mujoco_env.timestep
        if not np.isclose(mujoco_env.data.time, expected_time, atol=1e-10, rtol=1e-10):
            raise RuntimeError('Headless simulation time reset unexpectedly')
        print(f'Headless simulation: steps={headless_steps}, '
              f'time={mujoco_env.data.time:.6f}, finite={bool(finite)}')
        return

    # initialize sdk interface only for interactive simulation
    from unitree_interface import SimInterface

    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    # # define body name & id for wrench calculation (optional)
    # body_name = 'left_wrist_yaw_link'
    # body_id = mujoco_env.model.body(body_name).id

    # # pyvista visualization (optional)
    # pv_interface = PVInterface(mujoco_env.model, mujoco_env.data)
    # pv_interface.track_body(body_name)

    print('Press P to pause/resume the simulation loop')
    print('Press R to reset the simulation')

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        start_time = time.time()

        if mujoco_env.paused:
            # keep viewer responsive while paused
            mujoco_env.sync_viewer()
        else:
            mujoco_env.sim_step()

        # # optional: get and display wrench on end-effector
        # force, torque = mujoco_env.get_body_wrench(body_id)
        # print(f'Force: {force}, Torque: {torque}')

        # # optional: update pyvista visualization
        # pv_interface.update_vector(force)
        # pv_interface.pv_render()

        time.sleep(max(0, mujoco_env.timestep - (time.time() - start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run H1-2 Mujoco simulation')
    parser.add_argument(
        '--fixed',
        action='store_true',
        help='Use pelvis-fixed scene without elastic band',
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--handless',
        dest='model',
        action='store_const',
        const='handless',
        help='Load handless model scene files from CL_Assets',
    )
    model_group.add_argument(
        '--magpie',
        dest='model',
        action='store_const',
        const='magpie',
        help='Load magpie-hand model scene files from CL_Assets (default)',
    )
    parser.set_defaults(model='magpie')
    parser.add_argument(
        '--scene',
        help='Load a configured Magpie scene from YAML',
    )
    parser.add_argument(
        '--headless-steps',
        type=int,
        default=None,
        metavar='N',
        help='Step N times without DDS or a viewer and report finite state',
    )
    parser.add_argument(
        '--force',
        nargs='+',
        default=None,
        help='Enable external force interface for specified link names',
    )
    args = parser.parse_args()
    if args.scene is not None and args.model == 'handless':
        parser.error('--scene cannot be used with --handless; scenes require Magpie')
    if args.headless_steps is not None and args.headless_steps < 0:
        parser.error('--headless-steps must be non-negative')

    sim_loop(fixed=args.fixed, force_links=args.force, model=args.model,
             scene=args.scene, headless_steps=args.headless_steps)
