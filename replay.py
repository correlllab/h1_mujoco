#!/usr/bin/env python3
import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


# -----------------------------------------------------------
# integrate obstacle trajectory (ballistic)
# -----------------------------------------------------------
def integrate_obstacle(start, vel0, gravity, T, dt):
    # Ensure shape (3,)
    start = np.asarray(start).reshape(3,)
    vel0  = np.asarray(vel0).reshape(3,)
    gravity = np.asarray(gravity).reshape(3,)

    pos = np.zeros((T, 3))
    vel = np.zeros((T, 3))

    p = start.copy()
    v = vel0.copy()

    for t in range(T):
        pos[t] = p
        vel[t] = v
        p = p + v * dt
        v = v + gravity * dt

    return pos, vel


# -----------------------------------------------------------
# MAIN replay function
# -----------------------------------------------------------
def replay(model, data, df, play_rate=1.0):

    dt = model.opt.timestep
    T = len(df)

    # Extract robot qpos/qvel (ignore obstacle DOFs)
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qvel_cols = [c for c in df.columns if c.startswith("qvel_")]
    qpos_robot = df[qpos_cols].to_numpy()
    qvel_robot = df[qvel_cols].to_numpy()

    # Extract launch parameters
    start = np.array([
        df["obst_px"][0],
        df["obst_py"][0],
        df["obst_pz"][0],
    ])
    vel0 = np.array([
        df["cmd_vx"][0],
        df["cmd_vy"][0],
        df["cmd_vz"][0],
    ])
    radius = df["obst_radius"][0]

    gravity = np.array(model.opt.gravity)
    obst_pos, obst_vel = integrate_obstacle(start, vel0, gravity, T, dt)

    # find obstacle body/joint indices
    try:
        obst_jid = model.joint("obstacle_freejoint").id
        obst_qpos_adr = model.jnt_qposadr[obst_jid]
    except:
        print("WARNING: obstacle freejoint not found in XML. Obstacle will not move.")
        obst_jid = None

    viewer = mujoco.viewer.launch_passive(model, data)

    for t in range(T):
        # load robot state
        data.qpos[:qpos_robot.shape[1]] = qpos_robot[t]
        data.qvel[:qvel_robot.shape[1]] = qvel_robot[t]

        # set obstacle state if exists
        if obst_jid is not None:
            # obstacle uses freejoint: qpos = [qw qx qy qz x y z]
            px, py, pz = obst_pos[t]
            qw, qx, qy, qz = 1, 0, 0, 0  # identity orientation
            data.qpos[obst_qpos_adr : obst_qpos_adr + 7] = [
                qw, qx, qy, qz, px, py, pz
            ]

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt / play_rate)

    viewer.close()
    print("Replay finished.")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--play_rate", type=float, default=1.0)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    replay(model, data, df, play_rate=args.play_rate)


if __name__ == "__main__":
    main()
