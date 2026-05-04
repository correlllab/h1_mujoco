import mujoco
import mujoco.viewer
import numpy as np

class MujocoEnv:
    def __init__(self, xml_path):
        # initialize robot model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # initialize elastic band
        self.elastic_band = None

        # simulation pause toggle
        self.paused = False

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

    def reset_geom_buffer(self):
        scn = self.viewer.user_scn
        scn.ngeom = 0

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # set camera position
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -29.7
        self.viewer.cam.lookat = [0.001, -0.103, 0.623]

    def sim_step(self):
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # reset custom geometry buffer for visualization
        self.reset_geom_buffer()
        # sync user input
        self.viewer.sync()






