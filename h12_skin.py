import time
import numpy as np
import pyvista as pv

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface
import mujoco
import numpy as np

class CapacitiveSkin:
    def __init__(self, model, data, mode="spherical_distance", target_obj_geom_id=None):
        """
        mode: one of:
            - 'spherical_distance'
            - 'gt_site_obj_distance'
            - 'ray_depth_estimate'
        """
        self.model = model
        self.data = data
        self.mode = mode
        self.target_obj_geom_id = target_obj_geom_id

        # resolve site IDs
        self.register_all_skin_sites()

    def register_all_skin_sites(self):
        self.site_ids = []
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name is not None and 'sensor' in name.lower():
                self.site_ids.append(i)

    def compute_all_capacitances(self):
        if self.mode == "spherical_distance":
            return [self._spherical_distance(i) for i in self.site_ids]

        elif self.mode == "gt_site_obj_distance":
            return [self._gt_distance(i) for i in self.site_ids]

        elif self.mode == "ray_depth_estimate":
            return [self._ray_depth(i) for i in self.site_ids]

        else:
            raise ValueError(f"Unknown mode {self.mode}")

    # ---- MODES ----

    def _spherical_distance(self, sid):
        site_pos = self.data.site_xpos[sid]
        obj_pos = self.data.geom_xpos[self.target_obj_geom_id]
        d = np.linalg.norm(site_pos - obj_pos)
        return np.exp(-d)  # whatever your nonlinear mapping is

    def _gt_distance(self, sid):
        site_pos = self.data.site_xpos[sid]
        obj_pos = self.data.geom_xpos[self.target_obj_geom_id]
        return float(np.linalg.norm(site_pos - obj_pos))

    def _ray_depth(self, sid):
        """
        Use MuJoCo's built-in ray query:
        - origin: site position
        - direction: +Z axis of the site
        """
        origin = self.data.site_xpos[sid].copy()
        zaxis = self.data.site_xmat[sid].reshape(3, 3)[:, 2]  # column 2 = +Z
        geom_id = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            zaxis,
            4.0,          # ray extent
            None,
            1             # flags: 1 = exclude geoms on same body
        )

        if geom_id < 0:
            return -1.0
        else:
            hit = self.data.ray_xpos.copy()
            return np.linalg.norm(hit - origin)

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
        current_pos = mujoco_env.data.xpos[obstacle_id]
        direction = goal_position - current_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-3:
            direction /= dist
            mujoco_env.data.qvel[mujoco_env.model.body_jntadr[obstacle_id]:][:3] = direction * speed

        mujoco_env.sim_step()

        readings = cap_skin.compute_all_capacitances(mujoco_env)
        print(readings)


        time.sleep(max(0, mujoco_env.timestep - (time.time() - start)))
if __name__ == '__main__':
    sim_loop()
