import argparse
import threading
import time

import mujoco
import rclpy.executors

from magpie_hand_bridge import MagpieHandBridge
from mujoco_env import MujocoEnv
from mujoco_ros_bridge import RosSensorBridge, init_ros, shutdown_ros
from scene_builder import build_kitchen_scene
from unitree_interface import SimInterface


def sim_loop(viewer=True):
    """
    Run the MuJoCo H1-2 sim in the Robocasa kitchen. Publishes lowstate on
    DDS domain 1 and head RGBD + downward half-sphere lidar + TF on ROS 2.
    The native passive viewer is on by default (requires MUJOCO_GL=glfw and
    an X display); set viewer=False to run headless and visualize externally
    via RViz or Foxglove Studio.
    """
    scene_path = build_kitchen_scene(layout_id=1, style_id=1)
    mujoco_env = MujocoEnv(scene_path)
    mujoco_env.init_elastic_band("torso_link")

    # Shared lock protects MjData access between the main sim thread
    # (mj_step + ros_bridge.tick) and the DDS RecurrentThreads
    # (publish_low_state, publish_high_state, low_cmd_handler).
    sim_lock = threading.Lock()
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data, lock=sim_lock)

    init_ros()
    ros_bridge = RosSensorBridge(
        mujoco_env.model,
        mujoco_env.data,
        elastic_band=mujoco_env.elastic_band,
        sim_lock=sim_lock,
    )
    print("[h12_mujoco] ROS 2 sensor bridge started (camera + lidar + TF + /clock)")

    # Magpie hand bridges. Each spins on a background MultiThreadedExecutor
    # because their service / timer / action callbacks must fire while the
    # main thread is busy in mj_step + ros_bridge.tick (sensor publishes).
    # The shared sim_lock serialises MjData access between the sim thread
    # and the bridge callbacks (open / close / set_position / state pub /
    # deligrasp loop).
    hand_bridge_right = MagpieHandBridge(
        mujoco_env.model, mujoco_env.data, side="right", sim_lock=sim_lock)
    hand_bridge_left = MagpieHandBridge(
        mujoco_env.model, mujoco_env.data, side="left", sim_lock=sim_lock)
    hand_executor = rclpy.executors.MultiThreadedExecutor()
    hand_executor.add_node(hand_bridge_right)
    hand_executor.add_node(hand_bridge_left)
    hand_executor.add_node(ros_bridge)
    hand_executor_thread = threading.Thread(
        target=hand_executor.spin, daemon=True, name="magpie_hand_executor")
    hand_executor_thread.start()
    print("[h12_mujoco] Magpie hand bridges started: /gripper/{left,right}/*")

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
                mujoco.mj_step(mujoco_env.model, mujoco_env.data)
                ros_bridge.tick()
            if viewer:
                mujoco_env.reset_geom_buffer()
                mujoco_env.draw_elastic_band()
                mujoco_env.viewer.sync()
            time.sleep(max(0, mujoco_env.timestep - (time.time() - start_time)))
    finally:
        hand_executor.shutdown()
        hand_bridge_right.destroy_node()
        hand_bridge_left.destroy_node()
        ros_bridge.shutdown()
        shutdown_ros()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1-2 MuJoCo simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo passive viewer (default: viewer on)",
    )
    args = parser.parse_args()
    sim_loop(viewer=not args.headless)
