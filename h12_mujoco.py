import time
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface, ShadowInterface

def sim_loop():
    '''
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    '''
    # initialize mujoco environment
    mujoco_env = MujocoEnv('unitree_robots/h1_2/scene_pelvis_fixed.xml')
    # initialize sdk interface
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    # # define body name & id
    # body_name = 'left_wrist_pitch_link'
    # body_id = mujoco_env.model.body(body_name).id

    # # pyvista visualization
    # pv_interface = PVInterface(mujoco_env.model, mujoco_env.data)
    # pv_interface.track_body(body_name)

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.sim_step()

        # # get wrench
        # force, torque = mujoco_env.get_body_wrench(body_id)
        # print(f'Force: {force}, Torque: {torque}')

        # # update pyvista visualization
        # pv_interface.update_vector(force)
        # pv_interface.pv_render()

        # ensure correct time stepping
        time_until_next_step = mujoco_env.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def main():
    sim_loop()

if __name__ == '__main__':
    main()
