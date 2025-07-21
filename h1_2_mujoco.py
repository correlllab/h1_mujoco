import time
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
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

    # define body name & id
    body_name = 'left_wrist_pitch_link'
    body_id = mujoco_env.model.body(body_name).id

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

def shadow_loop():
    '''
    Shadowing real robot in mujoco.
    Subscribing to low state to overwrite mujoco state.
    '''
    # initialize mujoco environment
    mujoco_env = MujocoEnv('unitree_robots/h1_2/scene_pelvis_fixed.xml')
    # initialize sdk interface
    shadow_interface = ShadowInterface(mujoco_env.model, mujoco_env.data)

    # define body name, id & info IO directory
    body_name = 'left_wrist_yaw_link'
    body_id = mujoco_env.model.body(body_name).id
    dump_dir = '../CORLCode/Subscriber_Dump'

    # update pyvista visualization
    pv_interface = PVInterface(mujoco_env.model, mujoco_env.data)
    pv_interface.track_body(body_name)

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.sim_step()

        # get torque from tau_est
        joint_torque = np.zeros(mujoco_env.model.nv)
        motor_torque = shadow_interface.get_motor_torque()
        joint_torque[6:26] = motor_torque[0:20]
        joint_torque[38:45] = motor_torque[20:27]
        # get wrench
        force, torque = mujoco_env.get_body_wrench(body_id, joint_torque)
        print(f'Force: {force}, Torque: {torque}')

        # get transformation matrix
        matrix = mujoco_env.data.xmat[body_id]
        # transform to roll pitch yaw
        r = R.from_matrix(matrix.reshape(3, 3))
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        # write to file
        with open(f'{dump_dir}/wrist_orientation.txt', 'w') as f:
            f.write(f'[{roll}, {pitch}, {yaw}]\n')
            f.write(f'{[i for i in matrix]}\n')
        with open(f'{dump_dir}/wrist_wrench.txt', 'w') as f:
            force = list(force)
            torque = list(torque)
            #print(f"{force=}, {type(force)=})")
            #print(f"{torque=}, {type(torque)=})")

            f.write(f'{force}\n')
            f.write(f'{torque}\n')

        # update pyvista visualization
        pv_interface.update_vector(force)
        pv_interface.pv_render()

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
    mujoco_env.init_mink()

    # set IK task
    mujoco_env.set_ik_task(
        link_name='left_wrist_yaw_link',
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
