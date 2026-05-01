import mujoco
import numpy as np
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
# 27 motors on main body, 3 sensors on each motor
MOTOR_NUM = 27
MOTOR_SENSOR_NUM = 3

class SimInterface:
    def __init__(self, model, data, lock=None):
        # record mujoco model & data
        self.model = model
        self.data = data
        # Lock shared with the main sim loop to protect MjData access.
        # RecurrentThreads read sensordata and write ctrl from background
        # threads while the main loop calls mj_step — all must hold the
        # lock while touching data.
        self._lock = lock or threading.Lock()

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
        # publish low state
        self.low_state = LowState_default()
        self.low_state_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_publisher.Init()
        self.low_state_thread = RecurrentThread(
            interval=self.dt, target=self.publish_low_state, name='sim_lowstate'
        )
        self.low_state_thread.Start()

        # subscribe to low command
        self.low_cmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_subscriber.Init(self.low_cmd_handler, 10)

        # timeout detection + pose-hold fallback. When no command has been
        # received for `timeout` seconds, apply a stiff PD to the joint
        # positions captured at the moment the timeout first fires, so the
        # robot stays upright instead of collapsing into a NaN state.
        self.last_cmd_time = time.time()
        self.timeout = 0.1
        self.timeout_detected = False
        self.hold_q = None            # captured on first timeout, shape (num_motor,)
        self.hold_kp = 80.0
        self.hold_kd = 3.0
        self.timeout_thread = RecurrentThread(
            interval=0.01, target=self.check_cmd_timeout, name='cmd_watchdog'
        )
        self.timeout_thread.Start()

    @property
    def lock(self):
        return self._lock

    def publish_low_state(self):
        if self.data is not None:
            with self._lock:
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
            # write to low state (DDS publish is thread-safe, no lock needed)
            self.low_state_publisher.Write(self.low_state)

    def low_cmd_handler(self, msg: LowCmd_):
        if self.data is not None:
            with self._lock:
                # update last command time; drop any cached hold pose so the next
                # timeout re-snaps to wherever the robot is then.
                self.last_cmd_time = time.time()
                self.hold_q = None
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
                print('Command timeout! Holding current joint positions.')
                with self._lock:
                    if self.data is not None:
                        # Snapshot current joint positions for pose-hold.
                        self.hold_q = np.array([
                            self.data.sensordata[i] for i in range(self.num_motor)
                        ])
            # Apply stiff PD pose-hold every watchdog tick while timed out.
            if self.hold_q is not None and self.data is not None:
                with self._lock:
                    for i in range(self.num_motor):
                        q = self.data.sensordata[i]
                        dq = self.data.sensordata[i + self.num_motor]
                        self.data.ctrl[i] = (
                            self.hold_kp * (self.hold_q[i] - q)
                            - self.hold_kd * dq
                        )
        else:
            self.timeout_detected = False
