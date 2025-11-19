import time
import numpy as np
import pyvista as pv

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface
import mujoco
import numpy as np

class CapacitiveSkin:
    def __init__(self, model, data, eps=1.0, sensing_radius=0.15):
        self.model = model
        self.data = data
        self.eps = eps
        self.sensing_radius = sensing_radius
        self.sensor_site_ids = []

    def register_all_skin_sites(self):
        self.sensor_site_ids = []
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name is not None and 'sensor' in name.lower():
                self.sensor_site_ids.append(i)

    def compute_capacitance_pair(self, sensor_pos, obstacle_pos, obstacle_radius):
        d = np.linalg.norm(sensor_pos - obstacle_pos)
        if d > self.sensing_radius + obstacle_radius:
            return np.inf
        effective_d = max(0.01, d - obstacle_radius)
        return self.eps / effective_d

    def compute_distance(self, sensor_pos, obstacle_pos):
        dist = np.linalg.norm(sensor_pos - obstacle_pos)
        return np.inf if dist > self.sensing_radius else dist

    def compute_all_capacitances(self, mujoco_env):
        readings = {}
        for sid in self.sensor_site_ids:
            spos = self.data.site_xpos[sid]
            oid = mujoco_env.model.body('obstacle').id
            opos = mujoco_env.data.xpos[oid]
            radius = mujoco_env.model.geom_size[mujoco_env.model.body_geomadr[oid]][0]
            # readings[sid] = self.compute_capacitance_pair(spos, opos, radius)
            readings[sid] = self.compute_distance(spos, opos)
        return readings



def sim_loop():
    '''
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    '''
    # initialize mujoco environment
    mujoco_env = MujocoEnv('unitree_robots/h1_2/scene_obstacle_avoidance.xml')
    # initialize sdk interface
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)
    cap_skin = CapacitiveSkin(mujoco_env.model, mujoco_env.data)
    cap_skin.register_all_skin_sites()

    mujoco_env.launch_viewer()
    # main simulation loop
    obstacle_id = mujoco_env.model.body('obstacle').id
    goal_position = np.array([0.3, 0.0, 1.0])
    speed = 0.2

    while mujoco_env.viewer.is_running():
        start = time.time()

        # Move obstacle
        # current_pos = mujoco_env.data.xpos[obstacle_id]
        # direction = goal_position - current_pos
        # dist = np.linalg.norm(direction)
        # if dist > 1e-3:
        #     direction /= dist
        #     mujoco_env.data.qvel[mujoco_env.model.body_jntadr[obstacle_id]:][:3] = direction * speed

        mujoco_env.sim_step()

        readings = cap_skin.compute_all_capacitances(mujoco_env)
        print(readings)


        time.sleep(max(0, mujoco_env.timestep - (time.time() - start)))
if __name__ == '__main__':
    sim_loop()
