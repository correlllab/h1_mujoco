import time
import argparse
import numpy as np

from mujoco_env import MujocoEnv
from unitree_h1_2_interface import SimInterface, ShadowInterface

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

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.sim_step()

        # print body wrench in world frame
        body_id = mujoco_env.model.body('L_index_proximal').id
        force, torque = mujoco_env.get_body_wrench(body_id)
        print(f'Force: {force}, Torque: {torque}')

        # ensure correct time stepping
        time_until_next_step = mujoco_env.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def shadow_loop():
    '''
    Shadowing real robot in mujoco.
    Subscribing to low state to overwrite mujoco state.
    '''
    # initialize mujoco environment
    mujoco_env = MujocoEnv('unitree_robots/h1_2/scene_pelvis_fixed.xml')
    # initialize sdk interface
    shadow_interface = ShadowInterface(mujoco_env.model, mujoco_env.data)

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.sim_step()

        # print body wrench in world frame
        body_id = mujoco_env.model.body('L_index_proximal').id
        # get torque from tau_est
        joint_torque = np.zeros(mujoco_env.model.nv)
        joint_torque[0:27] = shadow_interface.get_joint_torque()
        # get wrench
        force, torque = mujoco_env.get_body_wrench(body_id, joint_torque)
        print(f'Force: {force}, Torque: {torque}')

        # ensure correct time stepping
        time_until_next_step = mujoco_env.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def command_loop():
    '''
    Running IK solver in mujoco to command the robot.
    Publishing low command to move the robot arm. #TODO
    '''
    # initialize mujoco environment
    mujoco_env = MujocoEnv('unitree_robots/h1_2/scene_with_target.xml')

    # set IK task
    mujoco_env.set_ik_task(
        link_name='left_wrist',
        target_position=np.array([0.7, 0.1, 1.5]),
        enabled_link_mask=[i for i in range(18, 26)]
    )

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.ik_step()

        # ensure correct time stepping
        time_until_next_step = mujoco_env.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def main():
    '''
    Main function to choose the operation mode.
    '''
    parser = argparse.ArgumentParser(description='Choose mode: sim, shadow, or command.')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sim', 'shadow', 'command'],
        required=True,
        help='Mode to run the script: sim, shadow, or command.'
    )
    args = parser.parse_args()

    if args.mode == 'sim':
        print('Running in simulation mode...')
        sim_loop()
    elif args.mode == 'shadow':
        print('Running in shadow mode...')
        shadow_loop()
    elif args.mode == 'command':
        print('Running in command mode...')
        command_loop()
    else:
        print(f'Invalid mode {args.mode}')
        exit()

if __name__ == '__main__':
    main()
