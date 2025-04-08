import mujoco
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
TOPIC_HIGHSTATE = 'rt/sportmodestate'

MOTOR_SENSOR_NUM = 3

class SimInterface:
    def __init__(self, model, data):
        # record mujoco model & data
        self.model = model
        self.data = data

        # initialize state parameters
        self.num_motor = self.model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.dt = self.model.opt.timestep

        # check sensor
        self.have_imu = False
        self.have_frame_sensor = False
        for i in range(self.dim_motor_sensor, self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == 'imu_quat':
                self.have_imu = True
            if name == 'frame_pos':
                self.have_frame_sensor = True

        # initialize channel
        ChannelFactoryInitialize(id=0, networkInterface='lo')
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
            # wrote motor state
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.data.sensordata[i]
                self.low_state.motor_state[i].dq = self.data.sensordata[
                    i + self.num_motor
                ]
                self.low_state.motor_state[i].tau_est = self.data.sensordata[
                    i + 2 * self.num_motor
                ]

            if self.have_frame_sensor:
                # write IMU data
                self.low_state.imu_state.quaternion[0] = self.data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.data.sensordata[
                    self.dim_motor_sensor + 3
                ]
                # write gyroscope data
                self.low_state.imu_state.gyroscope[0] = self.data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.data.sensordata[
                    self.dim_motor_sensor + 6
                ]
                # write accelerometer data
                self.low_state.imu_state.accelerometer[0] = self.data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.data.sensordata[
                    self.dim_motor_sensor + 9
                ]
            # write to low state
            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):
        if self.data is not None:
            # write global posiiton
            self.high_state.position[0] = self.data.sensordata[
                self.dim_motor_sensor + 10
            ]
            self.high_state.position[1] = self.data.sensordata[
                self.dim_motor_sensor + 11
            ]
            self.high_state.position[2] = self.data.sensordata[
                self.dim_motor_sensor + 12
            ]
            # write global velocity
            self.high_state.velocity[0] = self.data.sensordata[
                self.dim_motor_sensor + 13
            ]
            self.high_state.velocity[1] = self.data.sensordata[
                self.dim_motor_sensor + 14
            ]
            self.high_state.velocity[2] = self.data.sensordata[
                self.dim_motor_sensor + 15
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
        self.num_motor = self.model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.dt = self.model.opt.timestep

        # check sensor
        self.have_imu = False
        self.have_frame_sensor = False
        for i in range(self.dim_motor_sensor, self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == 'imu_quat':
                self.have_imu = True
            if name == 'frame_pos':
                self.have_frame_sensor = True

        # initialize channel
        ChannelFactoryInitialize(id=0, networkInterface='lo')
        # subscribe low state
        self.low_state_suber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.low_state_suber.Init(self.SubscribeLowState, 10)

    def SubscribeLowState(self, msg: LowState_):
        if self.data is not None:
            # # read motor state
            # for i in range(self.num_motor):
            #     self.data.sensordata[i] = msg.motor_state[i].q
            #     self.data.sensordata[i + self.num_motor] = msg.motor_state[i].dq
            #     self.data.sensordata[i + 2 * self.num_motor] = msg.motor_state[i].tau_est

            # if self.have_frame_sensor:
            #     # read IMU data
            #     self.data.sensordata[self.dim_motor_sensor + 0] = msg.imu_state.quaternion[0]
            #     self.data.sensordata[self.dim_motor_sensor + 1] = msg.imu_state.quaternion[1]
            #     self.data.sensordata[self.dim_motor_sensor + 2] = msg.imu_state.quaternion[2]
            #     self.data.sensordata[self.dim_motor_sensor + 3] = msg.imu_state.quaternion[3]
            #     # read gyroscope data
            #     self.data.sensordata[self.dim_motor_sensor + 4] = msg.imu_state.gyroscope[0]
            #     self.data.sensordata[self.dim_motor_sensor + 5] = msg.imu_state.gyroscope[1]
            #     self.data.sensordata[self.dim_motor_sensor + 6] = msg.imu_state.gyroscope[2]
            #     # read accelerometer data
            #     self.data.sensordata[self.dim_motor_sensor + 7] = msg.imu_state.accelerometer[0]
            #     self.data.sensordata[self.dim_motor_sensor + 8] = msg.imu_state.accelerometer[1]
            #     self.data.sensordata[self.dim_motor_sensor + 9] = msg.imu_state.accelerometer[2]
            for i in range(self.num_motor):
                self.data.qpos[i] = msg.motor_state[i].q
                self.data.qvel[i] = msg.motor_state[i].dq
                self.data.qfrc_applied[i] = msg.motor_state[i].tau_est
