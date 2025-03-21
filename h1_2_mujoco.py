import time
import mujoco
import mujoco.viewer
import numpy as np

from mink_interface import MinkInterface
from mujoco_interface import MujocoInterface

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk_interface import UnitreeSDKInterface

# initialize robot model
model = mujoco.MjModel.from_xml_path('unitree_robots/h1_2/scene_with_target.xml')
# data = mujoco.MjData(model)

# # initialize mink interface
mink_interface = MinkInterface()
model, data = mink_interface.init_model(model)
# initialize task and set target position
mink_interface.init_task('left_wrist', 'site')
mink_interface.set_target(np.array([0.7, 0.1, 1.5]))
enabled_link_mask = [i for i in range(18, 38)]

# initialize mujoco interface
mujoco_interface = MujocoInterface(model, data)
mujoco_interface.link_mink(mink_interface)
# initialize elastic band
# mujoco_interface.init_elastic_band()

# initialize sdk
# ChannelFactoryInitialize(id=0, networkInterface='lo')
# sdk_interface = UnitreeSDKInterface(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=mujoco_interface.key_callback) as viewer:
    # set camera position
    viewer.cam.azimuth = 88.650
    viewer.cam.distance = 5.269
    viewer.cam.elevation = -29.7
    viewer.cam.lookat = [0.001, -0.103, 0.623]

    # set simulation parameters
    model.opt.timestep = 0.005

    # main simulation loop
    while viewer.is_running():
        # record frame start time
        step_start = time.time()

        # evaluate elastic band
        # mujoco_interface.eval_band()

        # # step the simulator
        # mujoco.mj_step(model, data)
        # solve IK
        mink_interface.solve_IK(enabled_link_mask)

        # sync user input
        viewer.sync()

        # print camera info so we can set at desired position
        # print(viewer.cam)

        # ensure correct time stepping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
