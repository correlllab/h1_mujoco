import time
import mujoco
import mujoco.viewer

from mujoco_interface import MujocoInterface
from unitree_sdk_interface import UnitreeSDKInterface

# initialize robot model
# model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene.xml')
model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene_pelvis_fixed.xml')
data = mujoco.MjData(model)
# initialize mujoco interface
mujoco_interface = MujocoInterface(model, data)
sdk_interface = UnitreeSDKInterface(model, data)
# initialize elastic band
# mujoco_interface.init_elastic_band()

with mujoco.viewer.launch_passive(model, data, key_callback=mujoco_interface.key_callback) as viewer:
    # set camera position
    viewer.cam.azimuth = 88.650
    viewer.cam.distance = 5.269
    viewer.cam.elevation = -29.7
    viewer.cam.lookat = [0.001, -0.103, 0.623]

    # set simulation parameters
    model.opt.timestep = 0.005

    # print simulation information

    # main simulation loop
    while viewer.is_running():
        # record frame start time
        step_start = time.time()

        # evaluate elastic band
        # mujoco_interface.eval_band()

        # step the simulator
        mujoco.mj_step(model, data)
        # sync user input
        viewer.sync()

        # print camera info so we can set at desired position
        # print(viewer.cam)

        # ensure correct time stepping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
