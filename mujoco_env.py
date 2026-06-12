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
        self.force_target_body_ids = []
        self.force_enabled = False

        # set simulation parameters
        self.model.opt.timestep = 0.005
        self.timestep = self.model.opt.timestep

        # render decoupling: draw the viewer at ~render_hz instead of once PER physics step.
        # rendering every step is what throttled the twin to ~0.36x real-time; physics still steps
        # every iteration (band/forces/mj_step cadence UNCHANGED), only the screen refresh is rarer.
        self.render_hz = 30.0
        self._render_every = max(1, round((1.0 / self.render_hz) / self.timestep))
        self._render_i = 0

    def init_elastic_band(self, link_name='torso_link',
                          point=np.array([0, 0, 2.0]),
                          length=0, stiffness=1e3, damping=1e3,
                          tension_only=False, posture_kp=0.0, posture_kd=0.0,
                          release_fade=0.0):
        # initialize member variables
        self.elastic_band = ElasticBand(point, length, stiffness, damping,
                                        tension_only, posture_kp, posture_kd, release_fade)
        # attach band to model
        self.band_attached_body_id = self.model.body(link_name).id
        # posture assist acts on the band-attached body (torso). Keep gains MODERATE
        # (~300/60): stiff gains (1500/200) are explicit-integration UNSTABLE here -- wire
        # forensics showed kN*m oscillation, NaN explosions + MuJoCo auto-resets, and the
        # shaking tripped the safety estop 0.7 s after engagement. Hand the load to the
        # policy with release_fade, not with a stiffer hold.
        self.posture_body_id = self.band_attached_body_id

        # print instructions
        print('Elastic band initialized!')
        self.elastic_band.print_instructions()

    def init_external_force(self, link_names='torso_link',
                            magnitude=10.0):
        '''
        Initialize external force interface for human interaction simulation.

        Args:
            link_names: Names of bodies to apply forces to
            magnitude: Maximum force magnitude (N)
        '''
        try:
            self.external_force = EndEffectorForce(magnitude)
            if isinstance(link_names, str):
                link_names = [link_names]
            self.force_target_body_ids = [self.model.body(name).id for name in link_names]
            self.force_enabled = True
            print(f'External force interface initialized for bodies: {link_names}')
            self.external_force.print_instructions()
        except Exception as e:
            print(f'Error initializing external force for bodies {link_names}: {e}')
            available_bodies = [self.model.body(i).name for i in range(self.model.nbody)]
            print(f'Available wrist bodies: {available_bodies}')

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
            b = self.elastic_band
            # fade-out release: scale all support down over release_fade sim-seconds
            scale = 1.0
            if b.release_requested and b.release_fade > 0.0:
                if b._fade_t0 is None:
                    b._fade_t0 = self.data.time
                scale = 1.0 - (self.data.time - b._fade_t0) / b.release_fade
                if scale <= 0.0:
                    b.enabled = False
                    b.last_assist_mag = 0.0
                    b.fade_scale = 0.0
                    print('Harness fully released -- robot is free-standing')
                    return
                # quadratic ease-out: spend most of the fade at LOW support. While the
                # strap carries weight the planner cannot pre-center the body (its lean
                # commands fight an invisible rope), so the re-centering happens in the
                # low-support tail -- a linear fade left only ~1.7s of it (backward
                # topple exactly at fade end, 2026-06-12); squaring stretches the
                # under-100N tail to ~40% of the fade time.
                scale = scale * scale
            b.fade_scale = scale
            # get x and v of pelvis joint (joint on the hip)
            x = self.data.qpos[:3]
            v = self.data.qvel[:3]
            # evaluate elastic band force
            f = b.evaluate_force(x, v)
            # apply force to the band-attached link
            self.data.xfrc_applied[self.band_attached_body_id, :3] += scale * f
            # posture assist (harness mode): a point-force band cannot constrain TILT, so a
            # supported bring-up swings like a pendulum. When posture gains are set, also apply
            # a world-frame upright spring + angular damping on the attached body -- the
            # strap-stiffness / steadying-hands part of a real gantry harness. 0/0 = off.
            if b.posture_kp > 0.0 or b.posture_kd > 0.0:
                bid = self.posture_body_id
                R = self.data.xmat[bid].reshape(3, 3)
                vel6 = np.zeros(6)   # [angular(3); linear(3)], world frame (flg_local=0)
                mujoco.mj_objectVelocity(self.model, self.data,
                                         mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
                torque = scale * (b.posture_kp * np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
                                  - b.posture_kd * vel6[:3])
                self.data.xfrc_applied[bid, 3:] += torque
                b.last_assist_mag = float(np.linalg.norm(torque))

    def reset_geom_buffer(self):
        scn = self.viewer.user_scn
        scn.ngeom = 0

    def draw_elastic_band(self):
        '''Draw elastic band and anchor marker in the viewer.'''
        if self.viewer is None:
            return

        if self.elastic_band is not None and self.elastic_band.enabled:
            scn = self.viewer.user_scn
            anchor = self.elastic_band.point
            body_pos = self.data.xpos[self.band_attached_body_id]
            # tension-only harness: GREEN while slack (catch-only, exerting nothing),
            # blue while taut -- lets the operator SEE the strap release itself.
            slack = (self.elastic_band.tension_only and
                     np.linalg.norm(self.data.qpos[:3] - anchor) < self.elastic_band.length)
            color = (np.array([0.3, 1.0, 0.4, 0.8]) if slack
                     else np.array([0.5, 0.6, 1.0, 0.8]))

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
            self.force_target_body_ids and
            self.external_force.active):

            # apply force and torque to target bodies
            for body_id in self.force_target_body_ids:
                self.data.xfrc_applied[body_id, :3] += self.external_force.force
                self.data.xfrc_applied[body_id, 3:] += self.external_force.torque

    def draw_external_force(self):
        '''Draw external force vector as an arrow at the acting body'''
        if self.viewer is None:
            return

        if (self.external_force is not None and
            self.force_target_body_ids and
            self.external_force.active):
            scn = self.viewer.user_scn
            color = np.array([1.0, 0.2, 0.2, 0.8])

            # draw force arrow for each target body
            for body_id in self.force_target_body_ids:
                if scn.ngeom >= scn.maxgeom:
                    break
                origin = self.data.xpos[body_id]
                end = origin + (self.external_force.force + 1e-3) * 0.1

                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),         # size placeholder
                    np.zeros(3),         # position placeholder
                    np.eye(3).flatten(), # orientation matrix (identity)
                    color,               # RGBA color
                )
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom],        # geom to modify
                    mujoco.mjtGeom.mjGEOM_ARROW, # connector type
                    0.01,                        # arrow radius
                    origin,                      # arrow start point
                    end,                         # arrow end point
                )
                scn.ngeom += 1

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        # set camera position
        self.viewer.cam.azimuth = 88.650
        self.viewer.cam.distance = 5.269
        self.viewer.cam.elevation = -29.7
        self.viewer.cam.lookat = [0.001, -0.103, 0.623]

    def sim_step(self):
        # --- PHYSICS: every step ---
        # Clear last step's applied force so the band / external force apply a FRESH value each step.
        # eval_* use += (accumulate) and mj_step does NOT clear xfrc_applied, so without this the band
        # becomes an unstable INTEGRAL force that rockets the robot up (base_z -> thousands of m). It
        # builds up faster now that the twin runs real-time (more steps/sec). With this zero it is a
        # proper spring -> stable hang (~1.03-1.07 m). Also fixes the stale force left after SPACE-release.
        self.data.xfrc_applied[:] = 0.0
        # evaluate elastic band
        self.eval_elastic_band()
        # evaluate external forces
        self.eval_external_force()
        # step the simulator
        mujoco.mj_step(self.model, self.data)
        # --- RENDER: only ~render_hz, NOT every physics step (decoupled -> physics runs real-time) ---
        self._render_i += 1
        if self._render_i >= self._render_every:
            self._render_i = 0
            # reset custom geometry buffer for visualization
            self.reset_geom_buffer()
            # draw elastic band
            self.draw_elastic_band()
            # draw external force
            self.draw_external_force()
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
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100,
                 tension_only=False, posture_kp=0.0, posture_kd=0.0, release_fade=0.0):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.tension_only = tension_only   # a strap can PULL toward the anchor, never push
        self.posture_kp = posture_kp       # upright tilt spring (N*m/rad), applied by MujocoEnv
        self.posture_kd = posture_kd       # angular damping (N*m*s/rad)
        # release_fade > 0: releasing EASES the support off over this many sim-seconds instead
        # of cutting it -- an instant cut hands the policy whatever load the harness was still
        # carrying as a step disturbance (bench: topple in 1.4 s); a quasi-static fade lets the
        # planner track the shifting equilibrium. Double-SPACE forces an instant cut.
        self.release_fade = release_fade
        self.release_requested = False
        self._fade_t0 = None
        self.last_assist_mag = 0.0         # |posture torque| actually applied (N*m) -- the
                                           # "is the harness still load-bearing?" readout
        self.enabled = True

    def evaluate_force(self, x, v):
        # displacement
        displacement = np.linalg.norm(x - self.point)
        if displacement < 1e-9:
            return np.zeros(3)
        # direction
        direction = (x - self.point) / displacement
        # tangential velocity
        v_tan = np.dot(v, direction)
        # spring damper model (mag < 0 = pull toward anchor = rope tension)
        mag = -self.stiffness * (displacement - self.length) - self.damping * v_tan
        if self.tension_only and mag > 0.0:
            mag = 0.0   # slack (or compressed) rope exerts nothing
        f = mag * direction

        return f

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw

        if key == glfw.KEY_SPACE:
            # harness mode (fade configured): first SPACE eases the support off; a second
            # SPACE during the fade cuts it instantly (falls through to the toggle below)
            if self.release_fade > 0.0 and self.enabled and not self.release_requested:
                self.release_requested = True
                print(f'Harness release: easing support off over {self.release_fade:.0f}s '
                      '(SPACE again = instant cut)')
                return
            self.enabled = not self.enabled
            if self.enabled:   # re-enable resets a previous fade
                self.release_requested = False
                self._fade_t0 = None
            print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
            return

        if not self.enabled:
            return

        # gantry hoist: raise/lower the anchor like a real winch. Lower the limp robot
        # until the feet load and the knees fold -> a PHYSICALLY-reached crouch start
        # (no pose freeze, so the accelerometer stays live/healthy); hoist back up as
        # the robot rises, just like a real gantry operator. (P was taken by pause.)
        if key == glfw.KEY_U or key == glfw.KEY_L:
            self.point[2] += 0.05 if key == glfw.KEY_U else -0.05
            print(f"Hoist {'raised' if key == glfw.KEY_U else 'lowered'} to "
                  f"z={self.point[2]:.2f} (U=up L=lower, 5cm/press)")
            return
        # steadying-hands tilt assist, for supported bring-up from a hoisted crouch
        if key == glfw.KEY_T:
            if self.posture_kp > 0.0 or self.posture_kd > 0.0:
                self.posture_kp = self.posture_kd = 0.0
                print('Tilt assist OFF')
            else:
                self.posture_kp, self.posture_kd = 300.0, 60.0   # the proven-stable gains
                if self.release_fade <= 0.0:
                    self.release_fade = 3.0   # assist may carry load -> SPACE eases off, no step-cut
                print('Tilt assist ON (300/60 steadying hands) -- SPACE now eases support off over 3s')
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
        print('  SPACE - Toggle elastic band on/off (eases off over 3s when tilt assist is on)')
        print('  U / L - Hoist anchor Up / Lower (5cm per press) -- gantry-style crouch start:')
        print('          lower the limp robot until the knees bend (~0.7 rad), then press U')
        print('          once or twice to take up slack and HOLD it there (a limp robot keeps')
        print('          folding to the knee hard stop, and a joint AT its limit trips the')
        print('          safety estop = un-engageable). Engage, then hoist up as it rises.')
        print('  T - Toggle steadying tilt assist (use for bring-up from a hoisted crouch)')
        print('  Arrow Keys - Move band anchor point (←→ for X, ↑↓ for Y)')
        print('  , / . - Decrease/increase band rest length')

class EndEffectorForce:
    def __init__(self, magnitude=10.0):
        '''
        Initialize force controller for simulating external disturbance

        Args:
            magnitude: Force magnitude for keyboard control (N)
        '''
        self.magnitude = magnitude
        self.direction = np.array([1.0, 0.0, 0.0])
        self.torque = np.zeros(3)
        self.active = False
        self.step = 5.0

    @property
    def force(self):
        return self.magnitude * self.direction

    def set_force(self, force_vector):
        force = np.array(force_vector, dtype=np.float64)
        magnitude = np.linalg.norm(force)
        self.magnitude = magnitude
        if magnitude > 0:
            self.direction = force / magnitude

    def key_callback(self, key):
        '''Handle keyboard input for force control'''
        glfw = mujoco.glfw.glfw

        # toggle force application
        if key == glfw.KEY_F:
            self.active = not self.active
            print(f'External force {"activated" if self.active else "deactivated"}')
            # preset force direction and magnitude when activated
            # self.magnitude = 25.0
            # self.direction = np.array([0.0, 0.0, -1.0])
            if not self.active:
                self.magnitude = 0.0
                self.torque = np.zeros(3)

        if not self.active:
            return

        # force magnitude control
        if key == glfw.KEY_KP_ADD or key == glfw.KEY_EQUAL:  # + key
            self.magnitude += self.step
            print(f'Force magnitude: {self.magnitude:.1f} N')
        elif key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_MINUS:  # - key
            self.magnitude = max(self.magnitude - self.step, 0.0)
            print(f'Force magnitude: {self.magnitude:.1f} N')

        # directional force control (Arrow keys + Page Up/Down for 3D)
        elif key == glfw.KEY_UP:  # Forward (+X)
            self.direction = np.array([1.0, 0.0, 0.0])
            print(f'Applying forward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_DOWN:  # Backward (-X)
            self.direction = np.array([-1.0, 0.0, 0.0])
            print(f'Applying backward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_LEFT:  # Left (-Y)
            self.direction = np.array([0.0, -1.0, 0.0])
            print(f'Applying left force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_RIGHT:  # Right (+Y)
            self.direction = np.array([0.0, 1.0, 0.0])
            print(f'Applying right force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_PAGE_UP:  # Up (+Z)
            self.direction = np.array([0.0, 0.0, 1.0])
            print(f'Applying upward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_PAGE_DOWN:  # Down (-Z)
            self.direction = np.array([0.0, 0.0, -1.0])
            print(f'Applying downward force: {self.magnitude:.1f} N')
        elif key == glfw.KEY_R:  # Reset force
            self.magnitude = 0.0
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
