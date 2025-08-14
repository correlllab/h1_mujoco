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
        self.band_enabled = False

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

    def init_elastic_band(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        # initialize member variables
        self.elastic_band = ElasticBand(point, length, stiffness, damping)
        self.band_enabled = True
        # attach band to model
        self.band_attached_body_id = self.model.body('torso_link').id

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        # toggle elastic band
        if key == glfw.KEY_SPACE and self.elastic_band is not None:
            self.band_enabled = not self.band_enabled
            print(f"Elastic band {'enabled' if self.band_enabled else 'disabled'}")
        # handle input if elastic band is enabled
        if self.band_enabled:
            self.elastic_band.key_callback(key)
        # handle input if we have link a mink IK solver

    def eval_band(self):
        if self.elastic_band is not None and self.band_enabled:
            # get x and v of pelvis joint (joint on the hip)
            x = self.data.qpos[:3]
            v = self.data.qvel[:3]
            # evaluate elastic band force
            f = self.elastic_band.evalute_force(x, v)
            # apply force to the band-attached link
            self.data.xfrc_applied[self.band_attached_body_id, :3] = f

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        # set camera position
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -29.7
        self.viewer.cam.lookat = [0.001, -0.103, 0.623]

    def sim_step(self):
        # evaluate elastic band
        self.eval_band()
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # sync user input
        self.viewer.sync()

    def get_body_jacobian(self, body_id):
        '''
        Get the Jacobian of a given body.
        Return 3xN positional jacobian and 3xN rotational jacobian.
        '''
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)

        # jac_pos is 3xN positional jacobian, jac_rot is 3xN rotational jacobian
        # the first 6 DOF are fixed, so set to 0
        jac_pos[0:3, 0:7] = 0
        jac_rot[0:3, 0:7] = 0
        return jac_pos, jac_rot

    def get_site_jacobian(self, site_id):
        '''
        Get the Jacobian of a given site.
        Return 3xN positional jacobian and 3xN rotational jacobian.
        '''
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
        # jac_pos is 3xN positional jacobian, jac_rot is 3xN rotational jacobian
        return jac_pos, jac_rot

    def get_motor_torque(self):
        '''
        Get the motor torque.
        '''
        return np.array(self.data.actuator_force)

    def get_body_wrench(self, body_id, joint_torque=None):
        '''
        Get the body torque in world frame.
        '''
        if joint_torque is None:
            joint_torque = np.zeros(self.model.nv)
            motor_torque = self.get_motor_torque()
            joint_torque[6:26] = motor_torque[0:20]
            joint_torque[38:45] = motor_torque[20:27]
        # subtract gravity bias
        joint_torque[6:26] -= self.data.qfrc_bias[6:26]
        joint_torque[38:45] -= self.data.qfrc_bias[38:45]
        # get positional and rotational jacobian
        jac_pos, jac_rot = self.get_body_jacobian(body_id)
        # compute wrench in world frame
        jac = np.vstack((jac_pos, jac_rot))
        world_wrench = np.linalg.inv(jac @ jac.T) @ jac @ joint_torque
        world_force = world_wrench[:3]
        world_torque = world_wrench[3:]

        return world_force, world_torque

    def get_site_wrench(self, body_id, joint_torque=None):
        '''
        Get the body torque in world frame.
        '''
        if joint_torque is None:
            joint_torque = np.zeros(self.model.nv)
            motor_torque = self.get_motor_torque()
            joint_torque[7:27] = motor_torque[0:20]
            joint_torque[39:46] = motor_torque[20:27]
        # subtract gravity bias
        joint_torque[6:26] -= self.data.qfrc_bias[6:26]
        joint_torque[38:45] -= self.data.qfrc_bias[38:45]
        # get positional and rotational jacobian
        jac_pos, jac_rot = self.get_site_jacobian(body_id)
        # compute wrench in world frame
        jac = np.vstack((jac_pos, jac_rot))
        world_wrench = np.linalg.inv(jac @ jac.T) @ jac @ joint_torque
        world_force = world_wrench[:3]
        world_torque = world_wrench[3:]

        return world_force, world_torque

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
