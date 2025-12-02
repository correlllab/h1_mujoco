#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import mujoco
from collections import OrderedDict

# -----------------------------------------------------------
# Same capacitance function as in replay
# -----------------------------------------------------------
def compute_capacitance(sensor_pos, obst_pos, obst_radius, eps=1.0, sensing_radius=0.15):
    dx, dy, dz = sensor_pos - obst_pos
    d = np.sqrt(dx*dx + dy*dy + dz*dz)

    if d > sensing_radius + obst_radius:
        return -1.0

    effective_d = max(0.01, d - obst_radius)
    return eps / effective_d


def get_skin_site_ids(model):
    ids = OrderedDict()
    for i in range(model.nsite):
        name = model.site(i).name
        if name and "sensor" in name:
            ids[i] = name
    return ids


# -----------------------------------------------------------
# Ballistic integration
# -----------------------------------------------------------
def integrate_ballistic(start, vel0, gravity, T, dt):
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
# Random sampling helpers
# -----------------------------------------------------------
def random_unit_vector():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)


def sample_random_obstacle_params(orig_start, orig_radius):
    """
    You can tune these ranges later.
    """
    # --- position perturbation (±40 cm cube)
    start = orig_start + np.random.uniform(-0.4, 0.4, size=3)

    # --- radius perturbation (±50%)
    radius = float(orig_radius) * np.random.uniform(0.5, 1.5)

    # --- speed: 1–10 m/s
    speed = np.random.uniform(1.0, 10.0)

    # --- direction aiming roughly toward torso:
    # Start by picking random
    direction = random_unit_vector()

    return start, radius, direction * speed


# -----------------------------------------------------------
# Collision sanity checks
# -----------------------------------------------------------
def passes_collision_checks(model, data, skin_site_ids, obst_xyz_traj, radius, max_dist=2.0):
    """
    Basic sanity checks:
      1. Obstacle must pass within some distance of torso or any skin site
      2. Obstacle must leave the close region eventually
    """
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_traj = data.xpos  # we'll update this manually per step

    ever_close = False
    eventually_far = False

    close_count = 0
    far_count = 0

    for t in range(len(obst_xyz_traj)):
        # temporarily set obstacle pos & forward
        data.qpos[:] = 0.0  # don't care about robot, only site positions
        data.qpos[0:3] = [1, 0, 0]  # valid quaternion
        data.xpos[3*torso_id: 3*torso_id+3] = 0  # ignore robot absolute

        # Compute closest distance to any skin site:
        dmin = 999.0
        for sid in skin_site_ids:
            spos = data.site_xpos[sid]
            d = np.linalg.norm(spos - obst_xyz_traj[t])
            if d < dmin:
                dmin = d

        if dmin < max_dist:
            close_count += 1
            ever_close = True
        else:
            far_count += 1
            if close_count > 20:  # was close earlier, now far
                eventually_far = True

    return ever_close and eventually_far


# -----------------------------------------------------------
# MAIN augmentation
# -----------------------------------------------------------
def augment_one(csv_path, xml_path, out_path):
    print(f"\n=== AUGMENTING {csv_path} ===")

    df   = pd.read_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    T = len(df)

    # ----- extract robot state -----
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qvel_cols = [c for c in df.columns if c.startswith("qvel_")]

    # ----- extract original obstacle params -----
    orig_start = np.array([df["obst_px"][0], df["obst_py"][0], df["obst_pz"][0]])
    orig_radius = float(df["obst_radius"][0])

    # ----- get skin sites -----
    skin_site_ids_dict = get_skin_site_ids(model)
    skin_site_ids = list(skin_site_ids_dict.keys())
    skin_site_names = list(skin_site_ids_dict.values())

    # ===========================================================
    # TRY RANDOM NEW OBSTACLE SETUP UNTIL COLLISION CONDITIONS
    # ===========================================================
    for attempt in range(40):

        start, radius, vel0 = sample_random_obstacle_params(orig_start, orig_radius)
        grav = np.array(model.opt.gravity)
        obst_pos, obst_vel = integrate_ballistic(start, vel0, grav, T, dt)

        # Evaluate collision sanity check
        if passes_collision_checks(model, data, skin_site_ids, obst_pos, radius):
            print("✓ Using obstacle sample on attempt", attempt+1)
            break
    else:
        print("WARNING: Could not find valid obstacle sample. Using original trajectory.")
        start = orig_start
        radius = orig_radius
        obst_pos = df[["obst_px", "obst_py", "obst_pz"]].to_numpy()

    # ===========================================================
    # RE-COMPUTE CAPACITANCE FOR AUGMENTED OBSTACLE TRAJ
    # ===========================================================
    cap_new = np.zeros((T, len(skin_site_ids)))

    for t in range(T):
        # robot pose does not matter for this computation — only site positions do
        # but we do need a correct forward pass
        for i, col in enumerate(qpos_cols):
            data.qpos[i] = df[col][t]

        mujoco.mj_forward(model, data)

        for k, sid in enumerate(skin_site_ids):
            spos = np.array(data.site_xpos[sid])
            cap_new[t, k] = compute_capacitance(spos, obst_pos[t], radius)

    # ===========================================================
    # BUILD AUGMENTED CSV
    # ===========================================================
    df_aug = df.copy()

    # overwrite obstacle columns
    df_aug["obst_px"] = obst_pos[:, 0]
    df_aug["obst_py"] = obst_pos[:, 1]
    df_aug["obst_pz"] = obst_pos[:, 2]
    df_aug["obst_radius"] = radius
    df_aug["cmd_vx"] = vel0[0]
    df_aug["cmd_vy"] = vel0[1]
    df_aug["cmd_vz"] = vel0[2]

    # overwrite capacitance columns
    cap_cols = [f"cap_site_{sid}_{name}" for sid, name in zip(skin_site_ids, skin_site_names)]
    for j, col in enumerate(cap_cols):
        df_aug[col] = cap_new[:, j]

    df_aug.to_csv(out_path, index=False)
    print(f"✓ Wrote augmented CSV -> {out_path}")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--out", default="augmented")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Take the first trajectory in traj_dir/good/
    trajs = sorted(glob.glob(os.path.join(args.traj_dir, "good", "*.csv")))
    if len(trajs) == 0:
        raise RuntimeError("No trajectories in good/ directory.")

    first = trajs[1]
    base = os.path.basename(first)
    out_path = os.path.join(args.out, "AUG_" + base)

    augment_one(first, args.xml, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
