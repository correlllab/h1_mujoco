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
import os
import struct

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2, PointField


# Distance to step a ray past a self-hit before re-casting.
SELF_HIT_EPSILON = 1e-4
# Cap on self-hit re-casts per ray (multiRay first cast counts as one).
MAX_SELF_HITS = 16


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
        lidar_az_rays: int = 360,
        lidar_el_rays: int = 56,
        lidar_rate_hz: float = 10.0,
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
        # Lidar self-occlusion mask: ignore returns whose geom is attached to
        # `torso_link` (the body the lidar is mounted on — its outer shell
        # would otherwise show as a phantom near-zero return). Limbs hanging
        # off the torso (arms, head, legs) are deliberately NOT in this mask
        # so they occlude the scan like real obstacles.
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if torso_id < 0:
            raise RuntimeError("body 'torso_link' not found in MJCF")
        self.lidar_exclude_body_id = int(torso_id)
        self.geom_excluded = (model.geom_bodyid == self.lidar_exclude_body_id)
        # mj_multiRay / mj_ray require a (6, 1) column vector here.
        self.lidar_geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8).reshape(6, 1)

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_period = 1.0 / cam_rate_hz
        fovy = float(model.cam_fovy[self.cam_id])
        # Far-plane distance in world units. MuJoCo's depth buffer reports
        # zfar for rays that miss; mask those pixels to "no return" so
        # consumers don't treat the far plane as a real surface.
        self.zfar = float(model.vis.map.zfar * model.stat.extent)


        self.cam_info_msg = _build_camera_info(cam_width, cam_height, fovy, cam_frame)

        self.lidar_period = 1.0 / lidar_rate_hz
        self.lidar_max_range = lidar_max_range
        # Livox MID-360 FOV in WORLD coordinates is el ∈ [-7°, +52°] (mostly
        # above horizon). The sensor is mounted upside-down in the MJCF, so
        # local +Z = world -Z; flipping signs gives a LOCAL elevation range
        # of [-52°, +7°] that produces the correct world coverage.
        # Defaults of 360 az × 56 el rays at 10 Hz ≈ 200,000 pts/s, matching
        # the real sensor's per-second density. The uniform grid is a
        # simplification of the MID-360's non-repetitive scan pattern.
        az = np.linspace(0.0, 2.0 * np.pi, lidar_az_rays, endpoint=False, dtype=np.float64)
        el = np.linspace(-np.deg2rad(52.0), np.deg2rad(7.0), lidar_el_rays, endpoint=False, dtype=np.float64)
        az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
        cos_el = np.cos(el_grid)
        self.lidar_local_dirs = np.stack(
            [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
            axis=-1,
        ).reshape(-1, 3)
        self.lidar_rays = self.lidar_local_dirs.shape[0]
        # Preallocated mj_multiRay output buffers (overwritten each tick).
        # mujoco 3.3.1's pybind binding requires column-vector shapes
        # (m, 1) for the geomid / dist / vec / pnt / geomgroup args.
        self._lidar_dists = np.full((self.lidar_rays, 1), -1.0, dtype=np.float64)
        self._lidar_geomids = np.full((self.lidar_rays, 1), -1, dtype=np.int32)
        self._lidar_ray_geomid = np.zeros((1, 1), dtype=np.int32)

        # /clock stays on default reliable QoS so subscribers don't drop ticks.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        self.pub_rgb = self.create_publisher(
            CompressedImage, "/realsense/head/color/image_raw/compressed", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(
            CompressedImage, "/realsense/head/aligned_depth_to_color/image_raw/compressedDepth", qos_profile_sensor_data)
        self.pub_rgb_raw = self.create_publisher(
            Image, "/realsense/head/color/image_raw", qos_profile_sensor_data)
        self.pub_depth_raw = self.create_publisher(
            Image, "/realsense/head/aligned_depth_to_color/image_raw", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(
            CameraInfo, "/realsense/head/color/camera_info", qos_profile_sensor_data)
        self.pub_lidar = self.create_publisher(PointCloud2, "/livox/lidar", qos_profile_sensor_data)
        self.pub_imu = self.create_publisher(Imu, "/livox/imu", qos_profile_sensor_data)

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
        rgb_u8 = np.ascontiguousarray(r.render(), dtype=np.uint8)

        rgb_jpeg = io.BytesIO()
        PILImage.fromarray(rgb_u8, mode="RGB").save(rgb_jpeg, format="JPEG", quality=80)
        rgb_msg = CompressedImage()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.cam_frame
        rgb_msg.format = "jpeg"
        rgb_msg.data = rgb_jpeg.getvalue()
        self.pub_rgb.publish(rgb_msg)

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

        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_name)
        depth = r.render()

        # Mask "no hit" pixels (far-plane sentinel + non-finite/<=0) to 0
        # before quantizing to mm. RealSense uses 0 as the invalid value
        # in 16UC1 aligned-depth-to-color, which compressed_depth honors.
        invalid = ~np.isfinite(depth) | (depth <= 0) | (depth >= 0.999 * self.zfar)
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[invalid] = 0

        # 16UC1 mm → PNG → compressedDepth (12-byte ConfigHeader + PNG).
        # RealSense publishes aligned_depth_to_color in 16UC1 mm, so this
        # matches their format exactly; image_transport's compressed_depth
        # plugin decodes it directly.
        depth_png = io.BytesIO()
        PILImage.fromarray(depth_mm, mode="I;16").save(depth_png, format="PNG")
        # ConfigHeader: compressionFormat (uint32, 0=PNG), depthQuantA (f32), depthQuantB (f32).
        # 16UC1 uses raw PNG with no quantization, so A=B=0.
        depth_header = struct.pack("<Iff", 0, 0.0, 0.0)
        depth_msg = CompressedImage()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.cam_frame
        depth_msg.format = "16UC1; compressedDepth"
        depth_msg.data = depth_header + depth_png.getvalue()
        self.pub_depth.publish(depth_msg)

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
        origin = np.ascontiguousarray(self.data.xpos[self.lidar_body_id], dtype=np.float64).reshape(3, 1)
        rot = np.array(self.data.xmat[self.lidar_body_id], dtype=np.float64).reshape(3, 3)

        # Rotate local ray directions to world frame.
        # rot maps local→world (column vecs); for row-vec array: world = local @ rot.T
        world_dirs = np.ascontiguousarray(self.lidar_local_dirs @ rot.T, dtype=np.float64)

        # Fast path: one batched cast for all rays. Self-hits are resolved
        # in a follow-up per-ray pass below.
        dists = self._lidar_dists
        geomids = self._lidar_geomids
        dists.fill(-1.0)
        geomids.fill(-1)
        mujoco.mj_multiRay(
            self.model, self.data,
            origin, world_dirs.reshape(-1, 1),
            self.lidar_geomgroup,
            1,                  # include static geoms
            -1,                 # bodyexclude (none)
            geomids, dists,
            self.lidar_rays,
            self.lidar_max_range,
        )

        # Flatten (rays, 1) buffers back to 1D for the rest of the routine.
        dists_flat = dists.ravel()
        geomids_flat = geomids.ravel()
        hit_dists = dists_flat.copy()

        # Identify rays whose first hit was a torso-shell geom — the only body
        # we treat as self-occluding. Clamp -1 entries before indexing so the
        # masked comparison stays well-defined; the geomids>=0 guard keeps
        # the logic correct.
        safe_geomids = np.where(geomids_flat >= 0, geomids_flat, 0)
        excluded_hit_mask = (geomids_flat >= 0) & self.geom_excluded[safe_geomids]

        # Slow path: re-cast each excluded ray once with bodyexclude pinned
        # to the torso, so the next surface (limbs, scene geoms, or nothing)
        # becomes the reported return. The retry loop is kept generic in
        # case torso has multiple geoms separated in space.
        ray_geomid = self._lidar_ray_geomid
        for i in np.flatnonzero(excluded_hit_mask):
            world_dir = world_dirs[i].reshape(3, 1)
            accumulated = float(dists_flat[i]) + SELF_HIT_EPSILON
            ray_origin = origin + world_dir * accumulated
            hit_dists[i] = -1.0  # default to "no hit" until we exit the torso

            for _ in range(MAX_SELF_HITS - 1):
                d = mujoco.mj_ray(
                    self.model, self.data,
                    ray_origin, world_dir,
                    self.lidar_geomgroup,
                    1,
                    self.lidar_exclude_body_id,
                    ray_geomid,
                )
                if d < 0.0 or ray_geomid[0, 0] < 0:
                    break
                if not self.geom_excluded[ray_geomid[0, 0]]:
                    hit_dists[i] = accumulated + d
                    break
                step = d + SELF_HIT_EPSILON
                accumulated += step
                ray_origin = ray_origin + world_dir * step

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
    raw = os.getenv("ROS_DOMAIN_ID")
    if raw is None or raw == "":
        raise RuntimeError(
            "ROS_DOMAIN_ID is not set. Export a positive integer "
            "(domain 0 is reserved for the real robot), e.g. "
            "`export ROS_DOMAIN_ID=1`."
        )
    try:
        domain = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be an integer, got {raw!r}."
        ) from e
    if domain <= 0:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be a positive integer (>0), got {domain}. "
            "Domain 0 is reserved for the real robot."
        )
    if not rclpy.ok():
        rclpy.init()
    return True


def shutdown_ros() -> None:
    if rclpy.ok():
        rclpy.shutdown()
