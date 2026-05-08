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

        # viewer handle (set in launch_viewer)
        self.viewer = None

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

    def key_callback(self, key):
        if self.elastic_band is not None:
            self.elastic_band.key_callback(key)

    def init_elastic_band(self, link_name='torso_link', point=None,
                          length=0, stiffness=1e3, damping=1e3,
                          height_above=0.6):
        # attach band to model
        self.band_attached_body_id = self.model.body(link_name).id

        # Default the anchor to "directly above the attached body" so the
        # spring force is purely vertical regardless of where the robot
        # spawns. xpos for non-root bodies is only valid after a forward
        # pass, since MjData starts with identity-ish kinematics.
        if point is None:
            mujoco.mj_forward(self.model, self.data)
            point = self.data.xpos[self.band_attached_body_id].copy()
            point[2] += height_above

        self.elastic_band = ElasticBand(
            np.asarray(point, dtype=np.float64), length, stiffness, damping
        )

        print(f'Elastic band initialized at {self.elastic_band.point}')
        self.elastic_band.print_instructions()

    def eval_elastic_band(self):
        '''Evaluate and apply elastic band force if enabled'''
        if self.elastic_band is not None and self.elastic_band.enabled:
            # Use the attached body's world pose so the spring length matches
            # the visualized line. qvel[:3] is the floating base's linear
            # velocity — close enough to the torso's for a damping term.
            x = self.data.xpos[self.band_attached_body_id]
            v = self.data.qvel[:3]
            f = self.elastic_band.evaluate_force(x, v)
            self.data.xfrc_applied[self.band_attached_body_id, :3] += f

    def reset_geom_buffer(self):
        scn = self.viewer.user_scn
        scn.ngeom = 0

    def draw_elastic_band(self):
        '''Draw elastic band and anchor marker in the viewer.'''
        if self.viewer is None:
            return
        if self.elastic_band is None or not self.elastic_band.enabled:
            return

        scn = self.viewer.user_scn
        anchor = self.elastic_band.point
        body_pos = self.data.xpos[self.band_attached_body_id]
        color = np.array([0.5, 0.6, 1.0, 0.8])

        # anchor sphere
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0.0, 0.0]),
                anchor,
                np.eye(3).flatten(),
                color,
            )
            scn.ngeom += 1

        # band capsule from anchor to attached body
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                color,
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.01,
                anchor,
                body_pos,
            )
            scn.ngeom += 1

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        )

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



class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = True

    def evaluate_force(self, x, v):
        # displacement
        displacement = np.linalg.norm(x - self.point)
        # direction
        direction = (x - self.point) / displacement
        # tangential velocity
        v_tan = np.dot(v, direction)
        # spring damper model
        f = (-self.stiffness * (displacement - self.length) - self.damping * v_tan) * direction

        return f

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_SPACE:
            self.toggle()
            return

    def print_instructions(self):
        print('Elastic Band Controls:')
        print('  SPACE - Toggle elastic band on/off')
