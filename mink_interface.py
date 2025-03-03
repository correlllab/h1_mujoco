import mink

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

    def init_task(self, task_name, task_type,
                  position_cost=200.0, orientation_cost=0.0, lm_damping=1.0):
        self.task_init = True
        # set IK task
        self.task = mink.FrameTask(
            frame_name=task_name,
            frame_type=task_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping
        )
        return self.task

    def solve_IK(self, damping=0.1):
        if not self.model_init or not self.task_init:
            print('Model or task not initialized')

        # set target position
        self.task.set_target(mink.SE3.from_mocap_id(self.data, self.model.body(f'{self.task.frame_name}_target').mocapid[0]))
        vel = mink.solve_ik(self.configuration, [self.task], self.model.opt.timestep, self.solver, damping)
        vel[0:16] = 0
        vel[25] = 0
        self.configuration.integrate_inplace(vel, self.model.opt.timestep)
