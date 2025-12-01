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
# Capacitance functions (Python mirror of C++)
# -----------------------------------------------------------

def compute_capacitance(sensor_pos, obst_pos, obst_radius, eps=1.0, sensing_radius=0.15):
    dx, dy, dz = sensor_pos - obst_pos
    d = np.sqrt(dx*dx + dy*dy + dz*dz)

    if d > sensing_radius + obst_radius:
        return -1.0

    effective_d = max(0.01, d - obst_radius)
    return eps / effective_d


def get_skin_site_ids(model):
    site_ids = []
    for i in range(model.nsite):
        name = model.site(i).name
        if name is not None and "sensor" in name:
            site_ids.append(i)
    return site_ids


# -----------------------------------------------------------
# MAIN replay
# -----------------------------------------------------------
def replay(model, data, df, play_rate=1.0, args=None):

    dt = model.opt.timestep
    T = len(df)

    # --- robot state arrays ---
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qvel_cols = [c for c in df.columns if c.startswith("qvel_")]
    qpos_robot = df[qpos_cols].to_numpy()
    qvel_robot = df[qvel_cols].to_numpy()

    # --- obstacle params ---
    radius = float(df["obst_radius"][0])

    if args.replay_logged_obst:
        # Load obstacle position directly from log
        print("Replaying LOGGED obstacle positions.")
        obst_pos_cols = ["obst_px", "obst_py", "obst_pz"]
        obst_pos = df[obst_pos_cols].to_numpy()
        
        # Velocity must be set to 0 for a passive replay of position, or approximated via finite difference.
        # We will use zeros for simplicity, as the position is being set explicitly on every step.
        obst_vel = np.zeros((T, 3))
    else:
        # Calculate obstacle ballistic trajectory
        print("Integrating BALLISTIC obstacle trajectory.")
        start = np.array([df["obst_px"][0], df["obst_py"][0], df["obst_pz"][0]])
        vel0  = np.array([df["cmd_vx"][0], df["cmd_vy"][0], df["cmd_vz"][0]])
        gravity = np.array(model.opt.gravity)
        obst_pos, obst_vel = integrate_obstacle(start, vel0, gravity, T, dt)

    # --- site ids + logged capacitance columns ---
    skin_site_ids = get_skin_site_ids(model)
    cap_cols = [c for c in df.columns if c.startswith("cap_site_")]
    cap_logged = df[cap_cols].to_numpy()

    if len(cap_cols) != len(skin_site_ids):
        print("WARNING: CSV has", len(cap_cols),
              "cap columns but model has", len(skin_site_ids),
              "skin sites.")

    # --- obstacle freejoint index ---
    # Try to detect obstacle freejoint even if unnamed
    obst_jid = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            # check whether this joint belongs to obstacle body
            body = model.jnt_bodyid[j]
            name = model.body(body).name
            if name == "obstacle":
                obst_jid = j
                break

    if obst_jid is None:
        print("WARNING: obstacle freejoint not found. Obstacle will not move.")
    else:
        obst_qpos_adr = model.jnt_qposadr[obst_jid]
        obst_qvel_adr = model.jnt_dofadr[obst_jid]
    # ---- Viewer ----
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 5.0
    # ---- Comparison error accumulators ----
    rel_errors = []

    # --------------------------------------------------------
    # Time loop
    # --------------------------------------------------------
    for t in range(T):

        # Load robot state
        data.qpos[:qpos_robot.shape[1]] = qpos_robot[t]
        data.qvel[:qvel_robot.shape[1]] = qvel_robot[t]

        # Load obstacle ballistic state
        if obst_jid is not None:
            px, py, pz = obst_pos[t]
            qw, qx, qy, qz = 1, 0, 0, 0  # identity rotation
            # qpos update (7 elements)
            data.qpos[obst_qpos_adr:obst_qpos_adr+7] = [px, py, pz, qw, qx, qy, qz]
            
            # qvel update (6 elements):
            # Use obst_qvel_adr, which points to the start of the 6-D velocity block
            data.qvel[obst_qvel_adr:obst_qvel_adr+6] = [
                0, 0, 0,  # Angular velocity (roll, pitch, yaw rates)
                obst_vel[t][0], obst_vel[t][1], obst_vel[t][2] # Linear velocity (vx, vy, vz)
            ]
        mujoco.mj_forward(model, data)

        # ------------------------------
        # Compute LIVE capacitances
        # ------------------------------
        live = []
        obst_xyz = obst_pos[t]

        for sid in skin_site_ids:
            spos = data.site_xpos[sid]  # (3,)
            cval = compute_capacitance(spos, obst_xyz, radius)
            live.append(cval)

        live = np.array(live)

        # ------------------------------
        # Compare with LOGGED capacitances
        # ------------------------------
        logged_row = cap_logged[t]

        eps = 1e-6
        rel = np.abs(live - logged_row) / np.maximum(np.abs(logged_row), eps)
        rel_errors.extend(rel.tolist())

        # Per-step logging
        if args.log_error:
            mean_err = np.mean(rel)
            print(f"[t={t}] mean %error = {100*mean_err:.2f}%")

        if args.print_caps:
            print(f"[t={t}] live={live}, logged={logged_row}")

        # ------------------------------
        # Show in viewer
        # ------------------------------
        viewer.sync()
        time.sleep(dt / play_rate)

    if not args.loop:
        viewer.close()

    # --------------------------------------------------------
    # Print evaluation summary
    # --------------------------------------------------------
    rel_errors = np.array(rel_errors)
    print("\nCapacitance Replay Validation")
    print("-----------------------------------------")
    print(f"Mean % error:     {100*np.mean(rel_errors):.2f}%")
    print(f"Median % error:   {100*np.median(rel_errors):.2f}%")
    print(f"Max % error:      {100*np.max(rel_errors):.2f}%")
    print("-----------------------------------------")
    print("Done.")

    return viewer

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--play_rate", type=float, default=1.0)
    parser.add_argument("--log_error",  default=False, action="store_true")
    parser.add_argument("--print_caps", default=False,  action="store_true")
    parser.add_argument("--replay_logged_obst", default=False, action="store_true",
                        help="If set, uses logged obstacle position (obst_px/py/pz) instead of integrating ballistic trajectory.")
    parser.add_argument("--loop", default=False, action="store_true",
                    help="If set, prevents the viewer from closing immediately after the trajectory finishes.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    replay(model, data, df, play_rate=args.play_rate, args=args)


if __name__ == "__main__":
    main()
