"""ROS 2 publishers for MuJoCo head RGBD camera + 360 deg lidar.

Topics published (frame IDs match URDF link names):
  /head/color/image_raw       sensor_msgs/Image      (rgb8)
  /head/depth/image_raw       sensor_msgs/Image      (32FC1, meters)
  /head/color/camera_info     sensor_msgs/CameraInfo
  /lidar/points               sensor_msgs/PointCloud2

Single-threaded: both camera and lidar are driven by tick() from the main
sim loop. MuJoCo's EGL renderer context is thread-affine and can't be
created/used from a worker thread alongside the main-thread viewer, so we
do everything on the main thread and throttle by wall-clock rate.
"""
import time

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import TransformBroadcaster


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
        lidar_site: str = "lidar_site",
        lidar_frame: str = "lidar_link",
        cam_width: int = 640,
        cam_height: int = 480,
        cam_rate_hz: float = 15.0,
        lidar_rays: int = 360,
        lidar_rate_hz: float = 10.0,
        lidar_max_range: float = 30.0,
        tf_rate_hz: float = 50.0,
        world_frame: str = "world",
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
        self.lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lidar_site)
        if self.lidar_site_id < 0:
            raise RuntimeError(f"site '{lidar_site}' not found in MJCF")

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_period = 1.0 / cam_rate_hz
        fovy = float(model.cam_fovy[self.cam_id])

        # Prime kinematics so site_xpos/site_xmat are valid before the first
        # lidar ray (otherwise rot is all zeros and mj_ray logs
        # "vector length is too small").
        mujoco.mj_forward(model, data)

        self.cam_info_msg = _build_camera_info(cam_width, cam_height, fovy, cam_frame)

        self.lidar_rays = lidar_rays
        self.lidar_period = 1.0 / lidar_rate_hz
        self.lidar_max_range = lidar_max_range
        self.lidar_angles = np.linspace(0.0, 2.0 * np.pi, lidar_rays, endpoint=False, dtype=np.float64)

        self.pub_rgb = self.create_publisher(Image, "/head/color/image_raw", 10)
        self.pub_depth = self.create_publisher(Image, "/head/depth/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, "/head/color/camera_info", 10)
        self.pub_lidar = self.create_publisher(PointCloud2, "/lidar/points", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_period = 1.0 / tf_rate_hz
        self.world_frame = world_frame
        # MuJoCo body 0 is the worldbody itself — skip it. Publish every other
        # body as a flat world->body transform (xpos/xquat are world-frame).
        self.tf_body_ids = list(range(1, model.nbody))
        self.tf_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in self.tf_body_ids
        ]

        # Lazy renderer — created on first tick() call, which runs on the
        # same (main) thread as the viewer and subsequent renders.
        self._renderer = None

        self._last_cam_t = 0.0
        self._last_lidar_t = 0.0
        self._last_tf_t = 0.0

    def shutdown(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def tick(self) -> None:
        """Call once per sim step from the main loop. Rate-throttled internally."""
        now = time.time()
        if now - self._last_tf_t >= self.tf_period:
            self._last_tf_t = now
            try:
                self._publish_tf()
            except Exception as e:
                self.get_logger().warn(f"tf publish failed: {e}")
        if now - self._last_cam_t >= self.cam_period:
            self._last_cam_t = now
            try:
                self._publish_camera_frame()
            except Exception as e:
                self.get_logger().warn(f"camera publish failed: {e}")
        if now - self._last_lidar_t >= self.lidar_period:
            self._last_lidar_t = now
            try:
                self._publish_lidar_scan()
            except Exception as e:
                self.get_logger().warn(f"lidar publish failed: {e}")

    def _publish_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for body_id, name in zip(self.tf_body_ids, self.tf_body_names):
            pos = self.data.xpos[body_id]
            # MuJoCo quat is (w, x, y, z); ROS expects (x, y, z, w).
            quat = self.data.xquat[body_id]
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.world_frame
            t.child_frame_id = name
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.w = float(quat[0])
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            transforms.append(t)
        self.tf_broadcaster.sendTransform(transforms)

    def _publish_camera_frame(self) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)

        r = self._renderer
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_name)
        rgb = r.render()

        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_name)
        depth = r.render()

        stamp = self.get_clock().now().to_msg()

        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.cam_frame
        rgb_msg.height = self.cam_height
        rgb_msg.width = self.cam_width
        rgb_msg.encoding = "rgb8"
        rgb_msg.is_bigendian = 0
        rgb_msg.step = self.cam_width * 3
        rgb_msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        self.pub_rgb.publish(rgb_msg)

        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.cam_frame
        depth_msg.height = self.cam_height
        depth_msg.width = self.cam_width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = self.cam_width * 4
        depth_msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        self.pub_depth.publish(depth_msg)

        info = self.cam_info_msg
        info.header.stamp = stamp
        self.pub_info.publish(info)

    def _publish_lidar_scan(self) -> None:
        origin = np.array(self.data.site_xpos[self.lidar_site_id], dtype=np.float64)
        rot = np.array(self.data.site_xmat[self.lidar_site_id], dtype=np.float64).reshape(3, 3)

        pts_local = np.empty((self.lidar_rays, 3), dtype=np.float32)
        valid = np.zeros(self.lidar_rays, dtype=bool)
        geomid = np.zeros(1, dtype=np.int32)

        for i, angle in enumerate(self.lidar_angles):
            local_dir = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
            world_dir = rot @ local_dir
            dist = mujoco.mj_ray(
                self.model, self.data,
                origin, world_dir,
                None,   # geomgroup mask (all groups)
                1,      # include static geoms
                -1,     # exclude body: none
                geomid,
            )
            if 0.0 <= dist <= self.lidar_max_range:
                pts_local[i] = (local_dir * dist).astype(np.float32)
                valid[i] = True

        pts = pts_local[valid]

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
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
