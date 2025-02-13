import time
import mujoco
import mujoco.viewer

# initialize
model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene_terrain.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
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
