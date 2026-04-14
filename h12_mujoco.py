import argparse
import time

import mujoco

from mujoco_env import MujocoEnv
from unitree_interface import SimInterface

try:
    from ros_bridge import RosSensorBridge, init_ros, shutdown_ros
    _ROS_AVAILABLE = True
except ImportError as _ros_err:
    _ROS_AVAILABLE = False
    _ROS_IMPORT_ERROR = _ros_err


ASSETS_DIR = "/home/code/assets"


def sim_loop(fixed=False, force_links=None, no_ros=False):
    """
    Run the MuJoCo H1-2 sim headless. Publishes lowstate on DDS domain 1 and,
    if rclpy is available, head RGBD + 360 deg lidar + TF on ROS 2. Visualize
    externally via RViz or Foxglove Studio.
    """
    if fixed:
        scene_path = f"{ASSETS_DIR}/scene_handless_pelvis_fixed.xml"
        mujoco_env = MujocoEnv(scene_path)
    else:
        scene_path = f"{ASSETS_DIR}/scene_handless.xml"
        mujoco_env = MujocoEnv(scene_path)
        mujoco_env.init_elastic_band("torso_link")

    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    if force_links:
        mujoco_env.init_external_force(force_links, magnitude=5.0)

    ros_bridge = None
    if not no_ros:
        if _ROS_AVAILABLE:
            init_ros()
            ros_bridge = RosSensorBridge(mujoco_env.model, mujoco_env.data)
            print("[h12_mujoco] ROS 2 sensor bridge started (camera + lidar + TF)")
        else:
            print(f"[h12_mujoco] rclpy not available ({_ROS_IMPORT_ERROR}); skipping ROS bridge")

    try:
        while True:
            start_time = time.time()
            mujoco_env.eval_elastic_band()
            mujoco_env.eval_external_force()
            mujoco.mj_step(mujoco_env.model, mujoco_env.data)
            if ros_bridge is not None:
                ros_bridge.tick()
            time.sleep(max(0, mujoco_env.timestep - (time.time() - start_time)))
    finally:
        if ros_bridge is not None:
            ros_bridge.shutdown()
            shutdown_ros()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1-2 MuJoCo simulation (headless)")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use pelvis-fixed scene without elastic band",
    )
    parser.add_argument(
        "--force",
        nargs="+",
        default=None,
        help="Enable external force interface for specified link names",
    )
    parser.add_argument(
        "--no-ros",
        dest="no_ros",
        action="store_true",
        help="Disable ROS 2 camera/lidar/TF publishers",
    )
    args = parser.parse_args()

    sim_loop(
        fixed=args.fixed,
        force_links=args.force,
        no_ros=args.no_ros,
    )
