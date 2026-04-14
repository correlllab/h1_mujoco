import mujoco
import numpy as np
import time

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
        ChannelFactoryInitialize(id=1)
        # ChannelFactoryInitialize(id=0, networkInterface='lo')
        # publish low state
        self.low_state = LowState_default()
        self.low_state_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_publisher.Init()
        self.low_state_thread = RecurrentThread(
            interval=self.dt, target=self.publish_low_state, name='sim_lowstate'
        )
        self.low_state_thread.Start()

        # publish high state
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_publisher = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_publisher.Init()
        self.high_state_thread = RecurrentThread(
            interval=self.dt, target=self.publish_high_state, name='sim_highstate'
        )
        self.high_state_thread.Start()

        # subscribe to low command
        self.low_cmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_subscriber.Init(self.low_cmd_handler, 10)

        # timeout detection
        self.last_cmd_time = time.time()
        self.timeout = 0.1
        self.timeout_detected = False
        self.timeout_thread = RecurrentThread(
            interval=0.01, target=self.check_cmd_timeout, name='cmd_watchdog'
        )
        self.timeout_thread.Start()

    def publish_low_state(self):
        if self.data is not None:
            # write tick
            self.low_state.tick = int(self.data.time / self.dt)
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
            self.low_state_publisher.Write(self.low_state)

    def publish_high_state(self):
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
        self.high_state_publisher.Write(self.high_state)

    def low_cmd_handler(self, msg: LowCmd_):
        if self.data is not None:
            # update last command time
            self.last_cmd_time = time.time()
            # apply control to each motor
            for i in range(self.num_motor):
                if msg.motor_cmd[i].mode == 1:
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
                else:
                    self.data.ctrl[i] = 0.0

    def check_cmd_timeout(self):
        current_time = time.time()
        if (current_time - self.last_cmd_time) > self.timeout:
            if not self.timeout_detected:
                self.timeout_detected = True
                print('Command timeout! Resetting motor controls to zero.')
                # reset all motor controls to zero
                if self.data is not None:
                    for i in range(self.num_motor):
                        self.data.ctrl[i] = 0.0
        else:
            self.timeout_detected = False
