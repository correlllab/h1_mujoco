import time
import mujoco
import mujoco.viewer
import numpy as np

from mink_interface import MinkInterface
from mujoco_interface import MujocoInterface

class MujocoEnv:
    def __init__(self, xml_path):
        # initialize robot model
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        # initialize mink interface
        self.mink_interface = MinkInterface()
        self.model, self.data = self.mink_interface.init_model(self.model)

        # initialize mujoco interface
        self.mujoco_interface = MujocoInterface(self.model, self.data)
        self.mujoco_interface.link_mink(self.mink_interface)
        # initialize elastic band
        # self.mujoco_interface.init_elastic_band()

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

    def set_ik_task(self, link_name, target_position, enabled_link_mask):
        # initialize task and set target position
        self.mink_interface.init_task(link_name, 'site')
        self.mink_interface.set_target(target_position)
        self.enabled_link_mask = enabled_link_mask

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.mujoco_interface.key_callback)
        # set camera position
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -29.7
        self.viewer.cam.lookat = [0.001, -0.103, 0.623]

    def sim_step(self):
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # sync user input
        self.viewer.sync()

    def ik_step(self):
        # solve IK
        self.mink_interface.solve_IK(self.enabled_link_mask)
        # sync user input
        self.viewer.sync()

    def get_body_jacobian(self, body_id):
        '''
        Get the Jacobian of a given body.
        Return 3xN positional jacobian and 3xN rotational jacobian.
        '''
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
        # jac_pos is 3xN positional jacobian, jac_rot is 3xN rotational jacobian
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
        return np.array(self.data.ctrl)

    def get_body_wrench(self, body_id, joint_torque=None):
        '''
        Get the body torque in world frame.
        '''
        if joint_torque is None:
            joint_torque = np.zeros(self.model.nv)
            motor_torque = self.get_motor_torque()
            joint_torque[7:27] = motor_torque[0:20]
            joint_torque[39:46] = motor_torque[20:27]
        # get positional and rotational jacobian
        jac_pos, jac_rot = self.get_body_jacobian(body_id)
        # compute force in world frame
        world_force = np.linalg.inv(jac_pos @ jac_pos.T) @ jac_pos @ joint_torque
        world_torque = np.linalg.inv(jac_rot @ jac_rot.T) @ jac_rot @ joint_torque

        return world_force, world_torque

    def get_site_wrench(self, body_id, joint_torque=None):
        '''
        Get the body torque in world frame.
        '''
        if joint_torque is None:
            joint_torque = np.zeros(self.model.nv)
            joint_torque[0:27] = self.get_joint_torque()
        # get positional and rotational jacobian
        jac_pos, jac_rot = self.get_site_jacobian(body_id)
        # compute force & torque in world frame
        world_force = np.linalg.inv(jac_pos @ jac_pos.T) @ jac_pos @ joint_torque
        world_torque = np.linalg.inv(jac_rot @ jac_rot.T) @ jac_rot @ joint_torque

        return world_force, world_torque
