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

        # lookat follows the pelvis spawn baked in by scene_builder.SPAWN_POSES;
        # body_pos is the static offset on the merged XML, so no mj_forward needed.
        self.viewer.cam.lookat = self.model.body('pelvis').pos.copy()
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -50.0

        # Group 0 holds collision primitives (joint spheres, leg cylinders,
        # untextured collision-mesh duplicates); group 1 holds the textured
        # visual meshes. Hiding group 0 = the same effect as pressing "0".
        self.viewer.opt.geomgroup[0] = 0

    def sim_step(self):
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # reset custom geometry buffer for visualization
        self.reset_geom_buffer()
        # sync user input
        self.viewer.sync()






