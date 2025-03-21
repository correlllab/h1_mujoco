import mink
import mujoco

class MinkInterface:
    def __init__(self):
        self.model_init = False
        self.task_init = False

    def init_model(self, model):
        self.model_init = True
        self.configuration = mink.Configuration(model)
        self.solver = 'quadprog'

        # use configuration copy of model & data
        self.model = self.configuration.model
        self.data = self.configuration.data
        return self.model, self.data

    def init_task(self, frame_name, frame_type,
                  position_cost=200.0, orientation_cost=0.0, lm_damping=1.0):
        self.task_init = True
        # set IK task
        self.task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping
        )

    def set_target(self, position):
        assert(position.shape == (3,))
        # set target position
        mocap_id = self.model.body('target').mocapid[0]
        self.data.mocap_pos[mocap_id] = position
        # update task target
        self.task.set_target(mink.SE3.from_translation(position))

    def solve_IK(self, damping=0.1):
        if not self.model_init or not self.task_init:
            print('Model or task not initialized')

        # set target position
        vel = mink.solve_ik(self.configuration, [self.task], self.model.opt.timestep, self.solver, damping)
        vel[0:14] = 0
        vel[33:] = 0
        # vel[25] = 0
        self.configuration.integrate_inplace(vel, self.model.opt.timestep)

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_LEFT:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[0] -= 0.1
            self.set_target(position)
        if key == glfw.KEY_RIGHT:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[0] += 0.1
            self.set_target(position)
        if key == glfw.KEY_UP:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[1] += 0.1
            self.set_target(position)
        if key == glfw.KEY_DOWN:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[1] -= 0.1
            self.set_target(position)
        if key == glfw.KEY_COMMA:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[2] += 0.1
            self.set_target(position)
        if key == glfw.KEY_PERIOD:
            mocap_id = self.model.body('target').mocapid[0]
            position = self.data.mocap_pos[mocap_id]
            position[2] -= 0.1
            self.set_target(position)
