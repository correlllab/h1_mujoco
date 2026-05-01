"""ROS 2 publishers for MuJoCo head RGBD camera + downward half-sphere lidar.

Topics published (frame IDs match URDF link names):
  /clock                                                        rosgraph_msgs/Clock (sim time)
  /realsense/head/color/image_raw                               sensor_msgs/Image (rgb8)
  /realsense/head/color/image_raw/compressed                    sensor_msgs/CompressedImage (jpeg)
  /realsense/head/aligned_depth_to_color/image_raw              sensor_msgs/Image (16UC1 mm)
  /realsense/head/aligned_depth_to_color/image_raw/compressedDepth
                                                                sensor_msgs/CompressedImage (16UC1 png)
  /realsense/head/color/camera_info                             sensor_msgs/CameraInfo
  /livox/lidar                                                  sensor_msgs/PointCloud2
  /livox/imu                                                    sensor_msgs/Imu (co-located w/ lidar)

tick() is called once per sim step from the main loop. High-rate publishers
(clock, IMU) run at every call; slow publishers (camera, lidar) are
rate-throttled internally. MuJoCo's EGL renderer context is thread-affine,
so rendering must happen on the main thread.
"""
import io
import struct
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from PIL import Image as PILImage
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2, PointField


def _sim_time_to_msg(sim_time: float) -> TimeMsg:
    """Convert MuJoCo sim time (float seconds) to a ROS Time message."""
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def _build_camera_info(width: int, height: int, fovy_deg: float, frame_id: str) -> CameraInfo:
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info


class RosSensorBridge(Node):
    def __init__(
        self,
        model,
        data,
        cam_name: str = "head_cam",
        cam_frame: str = "camera_link",
        lidar_body: str = "livox_link",
        lidar_frame: str = "lidar_link",
        cam_width: int = 640,
        cam_height: int = 480,
        cam_rate_hz: float = 10.0,
        lidar_az_rays: int = 72,
        lidar_el_rays: int = 12,
        lidar_rate_hz: float = 5.0,
        lidar_max_range: float = 30.0,
        imu_quat_sensor: str = "livox_imu_quat",
        imu_gyro_sensor: str = "livox_imu_gyro",
        imu_acc_sensor: str = "livox_imu_acc",
        imu_frame: str = "lidar_link",
        imu_rate_hz: float = 100.0,
    ):
        super().__init__("mujoco_sensors")
        self.model = model
        self.data = data

        self.cam_name = cam_name
        self.cam_frame = cam_frame
        self.lidar_frame = lidar_frame

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise RuntimeError(f"camera '{cam_name}' not found in MJCF")
        self.lidar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_body)
        if self.lidar_body_id < 0:
            raise RuntimeError(f"body '{lidar_body}' not found in MJCF")
        # Filter self-hits during raycasting: walk the kinematic tree once so
        # we know which bodies share the lidar's root (the whole robot), and
        # skip them by re-casting with bodyexclude advancing past each hit.
        self.lidar_root_body = int(model.body_rootid[self.lidar_body_id])
        self.geom_root_body = model.body_rootid[model.geom_bodyid].astype(np.int32)
        self.lidar_geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        self.lidar_max_self_hits = 16  # safety cap on the per-ray bodyexclude loop

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_period = 1.0 / cam_rate_hz
        fovy = float(model.cam_fovy[self.cam_id])

        # Prime kinematics so site_xpos/site_xmat are valid before the first
        # lidar ray (otherwise rot is all zeros and mj_ray logs
        # "vector length is too small").
        mujoco.mj_forward(model, data)

        self.cam_info_msg = _build_camera_info(cam_width, cam_height, fovy, cam_frame)

        self.lidar_period = 1.0 / lidar_rate_hz
        self.lidar_max_range = lidar_max_range
        # Downward-facing half-sphere in the lidar's LOCAL frame.
        # el in [-pi/2, 0]: -pi/2 is straight down (-Z local), 0 is horizontal.
        az = np.linspace(0.0, 2.0 * np.pi, lidar_az_rays, endpoint=False, dtype=np.float64)
        el = np.linspace(-np.pi / 2.0, 0.0, lidar_el_rays, endpoint=False, dtype=np.float64)
        az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
        cos_el = np.cos(el_grid)
        self.lidar_local_dirs = np.stack(
            [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
            axis=-1,
        ).reshape(-1, 3)
        self.lidar_rays = self.lidar_local_dirs.shape[0]

        # /clock publisher — enables use_sim_time on subscriber nodes.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        self.pub_rgb = self.create_publisher(
            CompressedImage, "/realsense/head/color/image_raw/compressed", 10)
        self.pub_depth = self.create_publisher(
            CompressedImage, "/realsense/head/aligned_depth_to_color/image_raw/compressedDepth", 10)
        self.pub_rgb_raw = self.create_publisher(
            Image, "/realsense/head/color/image_raw", 10)
        self.pub_depth_raw = self.create_publisher(
            Image, "/realsense/head/aligned_depth_to_color/image_raw", 10)
        self.pub_info = self.create_publisher(
            CameraInfo, "/realsense/head/color/camera_info", 10)
        self.pub_lidar = self.create_publisher(PointCloud2, "/livox/lidar", 10)
        self.pub_imu = self.create_publisher(Imu, "/livox/imu", 50)

        self.imu_frame = imu_frame
        self.imu_period = 1.0 / imu_rate_hz
        self.imu_quat_adr, self.imu_quat_dim = self._sensor_adr(imu_quat_sensor)
        self.imu_gyro_adr, self.imu_gyro_dim = self._sensor_adr(imu_gyro_sensor)
        self.imu_acc_adr, self.imu_acc_dim = self._sensor_adr(imu_acc_sensor)

        # Lazy renderer — created on first tick() call, which runs on the
        # same (main) thread as the viewer and subsequent renders.
        self._renderer = None

        # Rate throttling uses sim time (data.time) not wall-clock, so topic
        # rates are consistent regardless of real-time factor.
        self._last_cam_sim_t = 0.0
        self._last_lidar_sim_t = 0.0
        self._last_imu_sim_t = 0.0

    def _sensor_adr(self, name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in MJCF")
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def shutdown(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def tick(self) -> None:
        """Call once per sim step from the main loop.

        /clock is published every call. TF and IMU are published at their
        target rates keyed off sim time. Camera and lidar are rate-throttled
        the same way but run less frequently (they involve rendering / ray
        casting on the main thread).
        """
        sim_t = float(self.data.time)
        stamp = _sim_time_to_msg(sim_t)

        # Always publish /clock so subscriber nodes can track sim time.
        clock_msg = Clock()
        clock_msg.clock = stamp
        self.pub_clock.publish(clock_msg)

        if sim_t - self._last_imu_sim_t >= self.imu_period:
            self._last_imu_sim_t = sim_t
            try:
                self._publish_imu(stamp)
            except Exception as e:
                self.get_logger().warn(f"imu publish failed: {e}")

        if sim_t - self._last_cam_sim_t >= self.cam_period:
            self._last_cam_sim_t = sim_t
            try:
                self._publish_camera_frame(stamp)
            except Exception as e:
                self.get_logger().warn(f"camera publish failed: {e}")

        if sim_t - self._last_lidar_sim_t >= self.lidar_period:
            self._last_lidar_sim_t = sim_t
            try:
                self._publish_lidar_scan(stamp)
            except Exception as e:
                self.get_logger().warn(f"lidar publish failed: {e}")

    def _publish_imu(self, stamp: TimeMsg) -> None:
        sd = self.data.sensordata
        # framequat sensor returns (w, x, y, z) world-frame; gyro/accel are site-local.
        q = sd[self.imu_quat_adr : self.imu_quat_adr + self.imu_quat_dim]
        w = sd[self.imu_gyro_adr : self.imu_gyro_adr + self.imu_gyro_dim]
        a = sd[self.imu_acc_adr : self.imu_acc_adr + self.imu_acc_dim]

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame
        msg.orientation.w = float(q[0])
        msg.orientation.x = float(q[1])
        msg.orientation.y = float(q[2])
        msg.orientation.z = float(q[3])
        msg.angular_velocity.x = float(w[0])
        msg.angular_velocity.y = float(w[1])
        msg.angular_velocity.z = float(w[2])
        msg.linear_acceleration.x = float(a[0])
        msg.linear_acceleration.y = float(a[1])
        msg.linear_acceleration.z = float(a[2])
        self.pub_imu.publish(msg)

    def _publish_camera_frame(self, stamp: TimeMsg) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)

        r = self._renderer
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_name)
        rgb = r.render()

        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_name)
        depth = r.render()

        rgb_u8 = np.ascontiguousarray(rgb, dtype=np.uint8)

        # RGB → JPEG CompressedImage
        buf = io.BytesIO()
        PILImage.fromarray(rgb_u8, mode="RGB").save(buf, format="JPEG", quality=80)
        rgb_msg = CompressedImage()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.cam_frame
        rgb_msg.format = "jpeg"
        rgb_msg.data = buf.getvalue()
        self.pub_rgb.publish(rgb_msg)

        # RGB → raw sensor_msgs/Image (rgb8)
        rgb_raw = Image()
        rgb_raw.header.stamp = stamp
        rgb_raw.header.frame_id = self.cam_frame
        rgb_raw.height = self.cam_height
        rgb_raw.width = self.cam_width
        rgb_raw.encoding = "rgb8"
        rgb_raw.is_bigendian = 0
        rgb_raw.step = self.cam_width * 3
        rgb_raw.data = rgb_u8.tobytes()
        self.pub_rgb_raw.publish(rgb_raw)

        # Depth (meters, float32) → 16UC1 mm → PNG → compressedDepth (12-byte
        # ConfigHeader + PNG). RealSense publishes aligned_depth_to_color in
        # 16UC1 mm, so this matches their format exactly; image_transport's
        # compressed_depth plugin decodes it directly.
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[~np.isfinite(depth) | (depth <= 0)] = 0
        buf = io.BytesIO()
        PILImage.fromarray(depth_mm, mode="I;16").save(buf, format="PNG")
        # ConfigHeader: compressionFormat (uint32, 0=PNG), depthQuantA (f32), depthQuantB (f32).
        # 16UC1 uses raw PNG with no quantization, so A=B=0.
        header = struct.pack("<Iff", 0, 0.0, 0.0)
        depth_msg = CompressedImage()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.cam_frame
        depth_msg.format = "16UC1; compressedDepth"
        depth_msg.data = header + buf.getvalue()
        self.pub_depth.publish(depth_msg)

        # Depth → raw sensor_msgs/Image (16UC1, mm)
        depth_raw = Image()
        depth_raw.header.stamp = stamp
        depth_raw.header.frame_id = self.cam_frame
        depth_raw.height = self.cam_height
        depth_raw.width = self.cam_width
        depth_raw.encoding = "16UC1"
        depth_raw.is_bigendian = 0
        depth_raw.step = self.cam_width * 2
        depth_raw.data = np.ascontiguousarray(depth_mm).tobytes()
        self.pub_depth_raw.publish(depth_raw)

        info = self.cam_info_msg
        info.header.stamp = stamp
        self.pub_info.publish(info)

    def _publish_lidar_scan(self, stamp: TimeMsg) -> None:
        origin = np.array(self.data.xpos[self.lidar_body_id], dtype=np.float64)
        rot = np.array(self.data.xmat[self.lidar_body_id], dtype=np.float64).reshape(3, 3)

        # Rotate local ray directions to world frame for mj_ray.
        # rot maps local→world (column vecs); for row-vec array: world = local @ rot.T
        world_dirs = self.lidar_local_dirs @ rot.T

        # Cast rays, collecting hit distances.  mj_ray is scalar so the loop
        # stays, but all per-ray math is deferred to the vectorized step below.
        hit_dists = np.full(self.lidar_rays, -1.0, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)

        for i in range(self.lidar_rays):
            world_dir = world_dirs[i]
            ray_origin = origin
            accumulated = 0.0
            exclude_body = -1
            for _ in range(self.lidar_max_self_hits):
                d = mujoco.mj_ray(
                    self.model, self.data,
                    ray_origin, world_dir,
                    self.lidar_geomgroup,
                    1,             # include static geoms
                    exclude_body,
                    geomid,
                )
                if d < 0.0 or geomid[0] < 0:
                    break
                if self.geom_root_body[geomid[0]] != self.lidar_root_body:
                    hit_dists[i] = accumulated + d
                    break
                step = d + 1e-4
                accumulated += step
                ray_origin = ray_origin + world_dir * step
                exclude_body = int(self.model.geom_bodyid[geomid[0]])

        # Points in lidar-local frame: local_dir * distance.
        valid = (hit_dists >= 0.0) & (hit_dists <= self.lidar_max_range)
        pts = (self.lidar_local_dirs[valid] * hit_dists[valid, np.newaxis]).astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.lidar_frame
        msg.height = 1
        msg.width = int(pts.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.pub_lidar.publish(msg)


def init_ros() -> bool:
    if not rclpy.ok():
        rclpy.init()
    return True


def shutdown_ros() -> None:
    if rclpy.ok():
        rclpy.shutdown()
