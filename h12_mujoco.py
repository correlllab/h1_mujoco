import argparse
import threading
import time

import mujoco

from mujoco_env import MujocoEnv
from mujoco_ros_bridge import RosSensorBridge, init_ros, shutdown_ros
from unitree_interface import SimInterface


ASSETS_DIR = "/home/code/assets"


def sim_loop(fixed=False, force_links=None, viewer=True):
    """
    Run the MuJoCo H1-2 sim. Publishes lowstate on DDS domain 1 and head RGBD
    + downward half-sphere lidar + TF on ROS 2. The native passive viewer is
    on by default (requires MUJOCO_GL=glfw and an X display); set viewer=False
    to run headless and visualize externally via RViz or Foxglove Studio.
    """
    if fixed:
        scene_path = f"{ASSETS_DIR}/scene_handless_pelvis_fixed.xml"
        mujoco_env = MujocoEnv(scene_path)
    else:
        scene_path = f"{ASSETS_DIR}/scene_handless.xml"
        mujoco_env = MujocoEnv(scene_path)
        mujoco_env.init_elastic_band("torso_link")

    # Shared lock protects MjData access between the main sim thread
    # (mj_step + ros_bridge.tick) and the DDS RecurrentThreads
    # (publish_low_state, publish_high_state, low_cmd_handler).
    sim_lock = threading.Lock()

    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data, lock=sim_lock)

    if force_links:
        mujoco_env.init_external_force(force_links, magnitude=5.0)

    init_ros()
    ros_bridge = RosSensorBridge(mujoco_env.model, mujoco_env.data)
    print("[h12_mujoco] ROS 2 sensor bridge started (camera + lidar + TF + /clock)")

    if viewer:
        mujoco_env.launch_viewer()
        print("[h12_mujoco] MuJoCo passive viewer launched")

    try:
        while True:
            start_time = time.time()
            if viewer and not mujoco_env.viewer.is_running():
                break
            with sim_lock:
                mujoco_env.eval_elastic_band()
                mujoco_env.eval_external_force()
                mujoco.mj_step(mujoco_env.model, mujoco_env.data)
                ros_bridge.tick()
            if viewer:
                mujoco_env.reset_geom_buffer()
                mujoco_env.draw_elastic_band()
                mujoco_env.draw_external_force()
                mujoco_env.viewer.sync()
            time.sleep(max(0, mujoco_env.timestep - (time.time() - start_time)))
    finally:
        ros_bridge.shutdown()
        shutdown_ros()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1-2 MuJoCo simulation")
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
        "--headless",
        action="store_true",
        help="Run without the MuJoCo passive viewer (default: viewer on)",
    )
    args = parser.parse_args()

    sim_loop(
        fixed=args.fixed,
        force_links=args.force,
        viewer=not args.headless,
    )
