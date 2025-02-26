import time
import mujoco
import mujoco.viewer

from mujoco_interface import MujocoInterface

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk_interface import UnitreeSDKInterface

import mink

# initialize robot model
# model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene.xml')
model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene_pelvis_fixed.xml')
data = mujoco.MjData(model)
# initialize mujoco interface
mujoco_interface = MujocoInterface(model, data)
# initialize elastic band
# mujoco_interface.init_elastic_band()

# initialize mink IK
configuration = mink.Configuration(model)
# set IK task
hand = 'left_wrist'
hand_task = mink.FrameTask(
    frame_name=hand,
    frame_type='site',
    position_cost=200.0,
    orientation_cost=0.0,
    lm_damping=1.0
)
hand_mid = model.body(f'{hand}_target').mocapid[0]
model = configuration.model
data = configuration.data
solver = 'quadprog'

# initialize sdk
ChannelFactoryInitialize(id=0, networkInterface='lo')
sdk_interface = UnitreeSDKInterface(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=mujoco_interface.key_callback) as viewer:
    # set camera position
    viewer.cam.azimuth = 88.650
    viewer.cam.distance = 5.269
    viewer.cam.elevation = -29.7
    viewer.cam.lookat = [0.001, -0.103, 0.623]

    # set simulation parameters
    model.opt.timestep = 0.005

    # move hand to initial position
    # mink.move_mocap_to_frame(model, data, f'{hand}_target', hand, 'site')

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

        # set target position
        hand_task.set_target(mink.SE3.from_mocap_id(data, hand_mid))
        vel = mink.solve_ik(configuration, [hand_task], model.opt.timestep, solver, 1e-1)
        vel[0:16] = 0
        vel[25] = 0
        configuration.integrate_inplace(vel, model.opt.timestep)

        # print camera info so we can set at desired position
        # print(viewer.cam)

        # ensure correct time stepping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
