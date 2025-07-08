import mujoco
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
TOPIC_HIGHSTATE = 'rt/sportmodestate'
TOPIC_HANDSTATE = 'rt/inspire/state'
# 27 motors on main body, 3 sensors on each motor
MOTOR_NUM = 27
MOTOR_SENSOR_NUM = 3

class SimInterface:
    def __init__(self, model, data):
        # record mujoco model & data
        self.model = model
        self.data = data

        # initialize state parameters
        self.num_motor = MOTOR_NUM
        self.num_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.dt = self.model.opt.timestep

        # check sensor
        self.have_imu = False
        self.have_frame_sensor = False
        for i in range(self.num_motor_sensor, self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == 'imu_quat':
                self.have_imu = True
            if name == 'frame_pos':
                self.have_frame_sensor = True

        # initialize channel
        ChannelFactoryInitialize(id=0)
        # ChannelFactoryInitialize(id=0, networkInterface='lo')
        # publish low state
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name='sim_lowstate'
        )
        self.lowStateThread.Start()

        # publish high state
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name='sim_highstate'
        )
        self.HighStateThread.Start()

        # subscribe to low command
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

    def PublishLowState(self):
        if self.data is not None:
            # write motor state
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.data.sensordata[i]
                self.low_state.motor_state[i].dq = self.data.sensordata[
                    i + self.num_motor
                ]
                self.low_state.motor_state[i].tau_est = self.data.sensordata[
                    i + 2 * self.num_motor
                ]
            if self.have_imu:
                # write IMU data
                self.low_state.imu_state.quaternion[0] = self.data.sensordata[
                    self.num_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.data.sensordata[
                    self.num_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.data.sensordata[
                    self.num_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.data.sensordata[
                    self.num_motor_sensor + 3
                ]
                # write gyroscope data
                self.low_state.imu_state.gyroscope[0] = self.data.sensordata[
                    self.num_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.data.sensordata[
                    self.num_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.data.sensordata[
                    self.num_motor_sensor + 6
                ]
                # write accelerometer data
                self.low_state.imu_state.accelerometer[0] = self.data.sensordata[
                    self.num_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.data.sensordata[
                    self.num_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.data.sensordata[
                    self.num_motor_sensor + 9
                ]
            # write to low state
            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):
        if self.data is not None:
            # write global posiiton
            self.high_state.position[0] = self.data.sensordata[
                self.num_motor_sensor + 10
            ]
            self.high_state.position[1] = self.data.sensordata[
                self.num_motor_sensor + 11
            ]
            self.high_state.position[2] = self.data.sensordata[
                self.num_motor_sensor + 12
            ]
            # write global velocity
            self.high_state.velocity[0] = self.data.sensordata[
                self.num_motor_sensor + 13
            ]
            self.high_state.velocity[1] = self.data.sensordata[
                self.num_motor_sensor + 14
            ]
            self.high_state.velocity[2] = self.data.sensordata[
                self.num_motor_sensor + 15
            ]
        # write to high state
        self.high_state_puber.Write(self.high_state)

    def LowCmdHandler(self, msg: LowCmd_):
        if self.data is not None:
            # apply control to each motor
            for i in range(self.num_motor):
                self.data.ctrl[i] = (
                    msg.motor_cmd[i].tau
                    + msg.motor_cmd[i].kp
                    * (msg.motor_cmd[i].q - self.data.sensordata[i])
                    + msg.motor_cmd[i].kd
                    * (
                        msg.motor_cmd[i].dq
                        - self.data.sensordata[i + self.num_motor]
                    )
                )

class ShadowInterface():
    def __init__(self, model, data):
        # record mujoco model & data
        self.model = model
        self.data = data

        # initialize state parameters
        self.num_motor = MOTOR_NUM
        self.num_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor

        # check sensor
        self.have_imu = False
        self.have_frame_sensor = False
        for i in range(self.num_motor_sensor, self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == 'imu_quat':
                self.have_imu = True
            if name == 'frame_pos':
                self.have_frame_sensor = True

        # variable tracking states
        self.motor_torque = np.zeros(self.num_motor)

        # initialize channel
        ChannelFactoryInitialize(id=0)
        # subscribe low state
        self.low_state_suber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.low_state_suber.Init(self.SubscribeLowState, 10)
        # subscribe hand state
        self.hand_state_suber = ChannelSubscriber(TOPIC_HANDSTATE, MotorStates_)
        self.hand_state_suber.Init(self.SubscribeHandState, 10)

    def SubscribeLowState(self, msg: LowState_):
        if self.data is not None:
            for i in range(self.num_motor):
                self.motor_torque[i] = msg.motor_state[i].tau_est
                self.data.ctrl[i] = (
                    msg.motor_state[i].tau_est
                    + msg.motor_state[i].kp
                    * (msg.motor_state[i].q - self.data.sensordata[i])
                    + msg.motor_state[i].kd
                    * (
                        msg.motor_state[i].dq
                        - self.data.sensordata[i + self.num_motor]
                    )
                )

    def SubscribeHandState(self, msg: MotorStates_):
        '''
        Finger state
        0: pinky
        1: ring
        2: middle
        3: index
        4: thumb
        5: thumb open
        '''
        # print(f'left hand: {[state.q for state in msg.states[:6]]}')
        # print(f'left hand ctrl: {self.data.ctrl[self.num_motor:self.num_motor + 12]}')
        # print(f'right hand: {[state.q for state in msg.states[6:12]]}')
        # print(f'right hand ctrl: {self.data.ctrl[self.num_motor + 12:self.num_motor + 24]}')
        finger_gain = 2.0
        if self.data is not None:
            # control pinky to index, 2 joints map to 1 control signal
            # target state [1, 0], sim joint state [0, 1.7]
            for i in range(4):
                # left hand
                self.data.ctrl[self.num_motor + i * 2] = (
                    finger_gain * (1 - msg.states[i].q -
                                   self.data.sensordata[self.num_motor_sensor + 16 + i * 2] / 1.7)
                )
                self.data.ctrl[self.num_motor + i * 2 + 1] = (
                    finger_gain * (1 - msg.states[i].q -
                                   self.data.sensordata[self.num_motor_sensor + 16 + i * 2 + 1] / 1.7)
                )
                # right hand
                self.data.ctrl[self.num_motor + i * 2 + 12] = (
                    finger_gain * (1 - msg.states[i + 6].q -
                                   self.data.sensordata[self.num_motor_sensor + 28 + i * 2] / 1.7)
                )
                self.data.ctrl[self.num_motor + i * 2 + 13] = (
                    finger_gain * (1 - msg.states[i + 6].q -
                                   self.data.sensordata[self.num_motor_sensor + 28 + i * 2 + 1] / 1.7)
                )

            # left thumb curl
            self.data.ctrl[self.num_motor + 8] = (
                finger_gain * (1 - msg.states[4].q -
                               (self.data.sensordata[self.num_motor_sensor + 16 + 8] + 0.1) / 0.7)
            ) # target state [1, 0], sim joint state [-0.1, 0.6]
            self.data.ctrl[self.num_motor + 9] = (
                finger_gain * (1 - msg.states[4].q -
                               self.data.sensordata[self.num_motor_sensor + 16 + 9] / 0.8)
            ) # target state [1, 0], sim joint state [0, 0.8]
            self.data.ctrl[self.num_motor + 10] = (
                finger_gain * (1 - msg.states[4].q -
                               self.data.sensordata[self.num_motor_sensor + 16 + 10] / 1.2)
            ) # target state [1, 0], sim joint state [0, 1.2]

            # left thumb open
            self.data.ctrl[self.num_motor + 11] = (
                finger_gain * (1 - msg.states[5].q -
                               (self.data.sensordata[self.num_motor_sensor + 16 + 11] + 0.1) / 1.4)
            ) # target state [1, 0], sim joint state [-0.1, 1.3]

            # right thumb curl
            self.data.ctrl[self.num_motor + 20] = (
                finger_gain * (1 - msg.states[10].q -
                               (self.data.sensordata[self.num_motor_sensor + 16 + 20] + 0.1) / 0.7)
            ) # target state [1, 0], sim joint state [-0.1, 0.6]
            self.data.ctrl[self.num_motor + 21] = (
                finger_gain * (1 - msg.states[10].q -
                               self.data.sensordata[self.num_motor_sensor + 16 + 21] / 0.8)
            ) # target state [1, 0], sim joint state [0, 0.8]
            self.data.ctrl[self.num_motor + 22] = (
                finger_gain * (1 - msg.states[10].q -
                               self.data.sensordata[self.num_motor_sensor + 16 + 22] / 1.2)
            ) # target state [1, 0], sim joint state [0, 1.2]

            # right thumb open
            self.data.ctrl[self.num_motor + 23] = (
                finger_gain * (1 - msg.states[11].q -
                               (self.data.sensordata[self.num_motor_sensor + 16 + 23] + 0.1) / 1.4)
            )

    def get_motor_torque(self):
        '''
        Get the mortor torque.
        '''
        return self.motor_torque
