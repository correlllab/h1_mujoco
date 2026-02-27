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

        # initialize end-effector force interface
        self.external_force = None
        self.force_target_body_id = None
        self.force_enabled = False

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

    def init_elastic_band(self, link_name='torso_link',
                          point=np.array([0, 0, 2.0]),
                          length=0, stiffness=1e3, damping=1e3):
        # initialize member variables
        self.elastic_band = ElasticBand(point, length, stiffness, damping)
        # attach band to model
        self.band_attached_body_id = self.model.body(link_name).id

        # print instructions
        print('Elastic band initialized!')
        self.elastic_band.print_instructions()

    def init_external_force(self, link_name='torso_link',
                            magnitude=10.0, damping=10.0):
        '''
        Initialize external force interface for human interaction simulation.

        Args:
            body_name: Name of the body to apply forces to (e.g., 'torso_link')
            magnitude: Maximum force magnitude (N)
            damping: Damping coefficient for velocity-based force reduction
        '''
        try:
            self.external_force = EndEffectorForce(magnitude, damping)
            self.force_target_body_id = self.model.body(link_name).id
            self.force_enabled = True
            print(f'External force interface initialized for body: {link_name}')
            self.external_force.print_instructions()
        except Exception as e:
            print(f'Error initializing external force for body {link_name}: {e}')
            available_bodies = [self.model.body(i).name for i in range(self.model.nbody)]
            print(f'Available wrist bodies: {available_bodies}')

    def set_external_force(self, force_vector, torque_vector=None):
        '''
        Directly set the external force and torque.

        Args:
            force_vector: 3D force vector in world frame (N)
            torque_vector: 3D torque vector in world frame (Nm), optional
        '''
        if self.external_force is not None:
            self.external_force.set_force(force_vector)
            if torque_vector is not None:
                self.external_force.torque = np.array(torque_vector)
            self.external_force.active = True

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw

        # pause / resume simulation loop
        if key == glfw.KEY_P:
            self.paused = not self.paused
            state = 'paused' if self.paused else 'resumed'
            print(f'Simulation {state}')

        # handle input for elastic band
        if self.elastic_band is not None:
            self.elastic_band.key_callback(key)
        # handle input for end-effector force
        if self.external_force is not None:
            self.external_force.key_callback(key)

    def eval_elastic_band(self):
        '''Evaluate and apply elastic band force if enabled'''
        if self.elastic_band is not None and self.elastic_band.enabled:
            # get x and v of pelvis joint (joint on the hip)
            x = self.data.qpos[:3]
            v = self.data.qvel[:3]
            # evaluate elastic band force
            f = self.elastic_band.evaluate_force(x, v)
            # apply force to the band-attached link
            self.data.xfrc_applied[self.band_attached_body_id, :3] = f

    def draw_elastic_band(self):
        '''Draw elastic band and anchor marker in the viewer.'''
        if self.viewer is None:
            return

        # reset per-frame geometry scene
        scn = self.viewer.user_scn
        scn.ngeom = 0

        if self.elastic_band is not None and self.elastic_band.enabled:
            anchor = self.elastic_band.point
            body_pos = self.data.xpos[self.band_attached_body_id]
            color = np.array([0.5, 0.6, 1.0, 0.8])

            # draw anchor point marker.
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.02, 0.0, 0.0]), # size (sphere radius in size[0])
                    anchor,                     # initial position (anchor)
                    np.eye(3).flatten(),        # orientation matrix (identity)
                    color,                      # RGBA color
                )
                scn.ngeom += 1

            # draw elastic band line from anchor to attached body.
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),         # size placeholder
                    np.zeros(3),         # position placeholder
                    np.eye(3).flatten(), # orientation matrix (identity)
                    color,               # RGBA color
                )
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom],          # geom to modify
                    mujoco.mjtGeom.mjGEOM_CAPSULE, # connector type
                    0.01,                          # capsule radius (band thickness)
                    anchor,                        # line start point
                    body_pos,                      # line end point
                )
                scn.ngeom += 1

    def eval_external_force(self):
        '''Apply external forces and torques for human interaction simulation'''
        if (self.external_force is not None and
            self.force_target_body_id is not None and
            self.external_force.active):

            # get current velocity of the target body for damping
            body_vel = self.data.cvel[self.force_target_body_id, :3]  # Linear velocity

            # apply damping to the force
            damped_force = self.external_force.apply_damping(body_vel)

            # apply force and torque to the target body
            self.data.xfrc_applied[self.force_target_body_id, :3] = damped_force
            self.data.xfrc_applied[self.force_target_body_id, 3:] = self.external_force.torque

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        # set camera position
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -29.7
        self.viewer.cam.lookat = [0.001, -0.103, 0.623]

    def sim_step(self):
        # evaluate elastic band
        self.eval_elastic_band()
        # evaluate external forces
        self.eval_external_force()
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # draw elastic band
        self.draw_elastic_band()
        # sync user input
        self.viewer.sync()

    def get_body_jacobian(self, body_id):
        '''
        Get the Jacobian of a given body.
        Return 3xN positional jacobian and 3xN rotational jacobian
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
        Return 3xN positional jacobian and 3xN rotational jacobian
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
        Get the body torque in world frame
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
        Get the body torque in world frame
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

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw

        if key == glfw.KEY_SPACE:
            self.enabled = not self.enabled
            print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
            return

        if not self.enabled:
            return

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

    def print_instructions(self):
        print('Elastic Band Controls:')
        print('  SPACE - Toggle elastic band on/off')
        print('  Arrow Keys - Move band anchor point (←→ for X, ↑↓ for Y)')
        print('  , / . - Decrease/increase band rest length')

class EndEffectorForce:
    def __init__(self, magnitude=10.0, damping=10.0):
        '''
        Initialize end-effector force controller for simulating human interaction

        Args:
            magnitude: Maximum force magnitude (N)
            damping: Damping coefficient for velocity-based force reduction
        '''
        self.magnitude = magnitude
        self.damping = damping
        self.force = np.zeros(3)  # Current applied force in world frame
        self.torque = np.zeros(3)  # Current applied torque in world frame
        self.active = False

        # force direction control
        self.force_direction = np.array([1.0, 0.0, 0.0])  # Current force direction

    def set_force(self, force_vector):
        '''Set the force vector directly'''
        self.force = np.array(force_vector)

    def set_force_magnitude_direction(self, magnitude, direction):
        '''Set force by magnitude and direction'''
        direction_normalized = np.array(direction) / np.linalg.norm(direction)
        self.force = magnitude * direction_normalized

    def apply_damping(self, velocity):
        '''Apply velocity-based damping to the force'''
        if np.linalg.norm(velocity) > 0:
            velocity_normalized = velocity / np.linalg.norm(velocity)
            damping_force = -self.damping * np.dot(velocity, velocity_normalized) * velocity_normalized
            return self.force + damping_force
        return self.force

    def key_callback(self, key):
        '''Handle keyboard input for force control'''
        glfw = mujoco.glfw.glfw

        # toggle force application
        if key == glfw.KEY_F:
            self.active = not self.active
            print(f'External force {"activated" if self.active else "deactivated"}')
            if not self.active:
                self.force = np.zeros(3)
                self.torque = np.zeros(3)

        if not self.active:
            return

        # force magnitude control
        if key == glfw.KEY_KP_ADD or key == glfw.KEY_EQUAL:  # + key
            self.magnitude = min(self.magnitude + 5.0, 200.0)
            print(f'Force magnitude: {self.magnitude:.1f} N')
        elif key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_MINUS:  # - key
            self.magnitude = max(self.magnitude - 5.0, 0.0)
            print(f'Force magnitude: {self.magnitude:.1f} N')

        # directional force control (Arrow keys + Page Up/Down for 3D)
        elif key == glfw.KEY_UP:  # Forward (+X)
            self.force = np.array([self.magnitude, 0, 0])
            print(f'Applying forward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_DOWN:  # Backward (-X)
            self.force = np.array([-self.magnitude, 0, 0])
            print(f'Applying backward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_LEFT:  # Left (-Y)
            self.force = np.array([0, -self.magnitude, 0])
            print(f'Applying left force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_RIGHT:  # Right (+Y)
            self.force = np.array([0, self.magnitude, 0])
            print(f'Applying right force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_PAGE_UP:  # Up (+Z)
            self.force = np.array([0, 0, self.magnitude])
            print(f'Applying upward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_PAGE_DOWN:  # Down (-Z)
            self.force = np.array([0, 0, -self.magnitude])
            print(f'Applying downward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_R:  # Reset force
            self.force = np.zeros(3)
            self.torque = np.zeros(3)
            print('Force reset to zero')

        # torque control (numpad keys)
        elif key == glfw.KEY_KP_4:  # Roll torque -
            self.torque = np.array([-self.magnitude * 0.1, 0, 0])
            print(f'Applying roll torque: {-self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_6:  # Roll torque +
            self.torque = np.array([self.magnitude * 0.1, 0, 0])
            print(f'Applying roll torque: {self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_8:  # Pitch torque +
            self.torque = np.array([0, self.magnitude * 0.1, 0])
            print(f'Applying pitch torque: {self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_2:  # Pitch torque -
            self.torque = np.array([0, -self.magnitude * 0.1, 0])
            print(f'Applying pitch torque: {-self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_7:  # Yaw torque +
            self.torque = np.array([0, 0, self.magnitude * 0.1])
            print(f'Applying yaw torque: {self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_9:  # Yaw torque -
            self.torque = np.array([0, 0, -self.magnitude * 0.1])
            print(f'Applying yaw torque: {-self.magnitude * 0.1:.1f} Nm')
        elif key == glfw.KEY_KP_5:  # Reset torque
            self.torque = np.zeros(3)
            print('Torque reset to zero')

    def print_instructions(self):
        print('External Force Controls:')
        print('  F - Toggle force application on/off')
        print('  + / - - Increase/decrease force magnitude')
        print('  Arrow Keys - Apply directional forces (←→ for Y, ↑↓ for X)')
        print('  Page Up/Down - Apply upward/downward force (Z axis)')
        print('  R - Reset force to zero')
        print('  Numpad 4/6 - Apply roll torque')
        print('  Numpad 8/2 - Apply pitch torque')
        print('  Numpad 7/9 - Apply yaw torque')
        print('  Numpad 5 - Reset torque to zero')
