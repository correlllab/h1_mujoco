import time
import numpy as np

from mujoco_env import MujocoEnv
from unitree_h1_interface import SimInterface

# initialize mujoco environment
mujoco_env = MujocoEnv('unitree_robots/h1/scene_with_target.xml')
# initialize sdk interface
sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

# set IK task
mujoco_env.set_ik_task(
    link_name='left_wrist',
    target_position=np.array([0.7, 0.1, 1.5]),
    enabled_link_mask=[i for i in range(16, 21)]
)

# launch viewer
mujoco_env.launch_viewer()
# main simulation loop
while mujoco_env.viewer.is_running():
    # record frame start time
    step_start = time.time()

    mujoco_env.sim_step()

    # ensure correct time stepping
    time_until_next_step = mujoco_env.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
