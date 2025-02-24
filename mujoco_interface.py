import mujoco
import mujoco.glfw
import numpy as np

class MujocoInterface:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.elastic_band = None
        self.band_enabled = None

    def init_elastic_band(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        # initialize member variables
        self.elastic_band = ElasticBand(point, length, stiffness, damping)
        self.band_enabled = True
        # attach band to model
        self.band_attached_link = self.model.body('torso_link').id

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        # toggle elastic band
        if key == glfw.KEY_SPACE and self.elastic_band is not None:
            self.band_enabled = not self.band_enabled
            print(f"Elastic band {'enabled' if self.band_enabled else 'disabled'}")
        # handle input if elastic band is enabled
        if self.band_enabled:
            self.elastic_band.key_callback(key)

    def eval_band(self):
        if self.elastic_band is not None and self.band_enabled:
            # get x and v of pelvis joint (joint on the hip)
            x = self.data.qpos[:3]
            v = self.data.qvel[:3]
            # evaluate elastic band force
            f = self.elastic_band.evalute_force(x, v)
            # apply force to the band-attached link
            self.data.xfrc_applied[self.band_attached_link, :3] = f

class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping

    def evalute_force(self, x, v):
        # displacement
        displacement = np.linalg.norm(x - self.point)
        # direction
        direction = (x - self.point) / displacement
        # tangential velocity
        v_tan = np.dot(v, direction)
        # spring damper model
        f = (-self.stiffness * (displacement - self.length) - self.damping * v_tan) * direction

        return f

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_LEFT:
            self.point[0] -= 0.1
        elif key == glfw.KEY_RIGHT:
            self.point[0] += 0.1
        elif key == glfw.KEY_UP:
            self.point[1] += 0.1
        elif key == glfw.KEY_DOWN:
            self.point[1] -= 0.1
        elif key == glfw.KEY_COMMA:
            self.length -= 0.1
        elif key == glfw.KEY_PERIOD:
            self.length += 0.1
