import time
import argparse
import numpy as np
import pyvista as pv

from mujoco_env import MujocoEnv
from unitree_h1_2_interface import SimInterface, ShadowInterface
from utility.mujoco_mesh_extract import mj_get_body_mesh, mj_get_contact_mesh, mj_mesh_to_polydata, mj_get_body_transform

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

    # get copy of mesh for pyvista visualization
    body_meshes = {}
    body_mesh_points = {}
    for i in range(mujoco_env.model.nbody):
        body_meshes[i] = mj_get_body_mesh(mujoco_env.model, i)
        if body_meshes[i] is not None:
            body_mesh_points[i] = np.array(body_meshes[i].points)
    # initialize pyvista
    pv.set_plot_theme('document')
    pl = pv.Plotter()
    pl.add_axes()
    pl.show(interactive_update=True)
    # add meshes to pyvista
    for i in range(mujoco_env.model.nbody):
        if body_meshes[i] is not None:
            color = 'red' if i == mujoco_env.model.body('left_wrist_yaw_link').id else 'lightblue'
            pl.add_mesh(body_meshes[i], color=color, show_edges=True, name=f'body_{i}')
    # add visualization arrow
    arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1)
    actor = pl.add_mesh(arrow, color='red', name='arrow')

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop
    while mujoco_env.viewer.is_running():
        # record frame start time
        step_start = time.time()

        mujoco_env.sim_step()

        # print body wrench in world frame
        body_id = mujoco_env.model.body('left_wrist_yaw_link').id
        # get torque from tau_est
        joint_torque = np.zeros(mujoco_env.model.nv)
        joint_torque[0:27] = shadow_interface.get_joint_torque()
        # get wrench
        force, torque = mujoco_env.get_body_wrench(body_id, joint_torque)
        print(f'Force: {force}, Torque: {torque}')

        # update pyvista meshes
        for i in range(mujoco_env.model.nbody):
            if body_meshes[i] is not None:
                # update mesh points
                body_meshes[i].points = body_mesh_points[i]
                # update mesh transform
                body_meshes[i].transform(mj_get_body_transform(mujoco_env.data, i))
        # update arrow transform
        position = mj_get_body_transform(mujoco_env.data, body_id)[:3, 3]
        new_arrow = pv.Arrow(start=position, direction=force, scale=0.1 * np.linalg.norm(force, ord=2))
        actor.mapper.SetInputData(new_arrow)
        actor.mapper.update()
        pl.update()

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
