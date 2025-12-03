#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import mujoco
from collections import OrderedDict
import json
import pickle

def compute_link_bounding_spheres(model):
    spheres = []
    for body_id in range(model.nbody):
        geoms = np.where(model.geom_bodyid == body_id)[0]
        if len(geoms) == 0:
            spheres.append(None)
            continue

        pts = []
        for g in geoms:
            xyz = model.geom_pos[g]
            size = model.geom_size[g]
            r = np.max(size) * 1.25
            pts.append((xyz, r))

        centers = np.array([p[0] for p in pts])
        center = np.mean(centers, axis=0)
        radius = np.max([np.linalg.norm(p[0] - center) + p[1] for p in pts])
        spheres.append((center, radius))
    return spheres

def is_robot_collision(model, data, qpos, obst_pos, obst_qpos_adr,
                       obst_body_id, spheres):
    """
    Efficient hybrid collision check:
    1) Broad phase: link bounding spheres
    2) Narrow phase: geom AABB
    3) Final real contact check (MuJoCo)
    """

    # --- Update robot and obstacle pose ---
    nq = qpos.shape[0]
    data.qpos[:nq] = qpos
    data.qpos[obst_qpos_adr:obst_qpos_adr+7] = [1,0,0,0,
                                                obst_pos[0], obst_pos[1], obst_pos[2]]
    mujoco.mj_forward(model, data)

    # -------------------------------
    # Layer 1: broad phase sphere test
    # -------------------------------
    for body_id, sph in enumerate(spheres):
        if sph is None:
            continue
        center_local, rad = sph
        center_world = data.xpos[body_id] + center_local
        d = np.linalg.norm(center_world - obst_pos)
        if d < rad:
            # Potential collision → continue to narrow phase
            break
    else:
        # No sphere overlap with ANY body → safe
        return False

    # -------------------------------
    # Layer 2: AABB geom tests
    # -------------------------------
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == obst_body_id:
            continue  # skip obstacle itself
        pos = data.geom_xpos[g]
        r = model.geom_size[g]
        d = np.maximum(np.abs(obst_pos - pos) - r, 0.0)
        if np.linalg.norm(d) < 0.01:
            # Very possible collision → go to final test
            break
    else:
        return False

    # -------------------------------
    # Layer 3: True contact test
    # -------------------------------
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 == obst_body_id or b2 == obst_body_id:
            if c.dist < 0.0:
                return True
    return False


# -----------------------------------------------------------
# Capacitance function
# -----------------------------------------------------------
def compute_capacitance(sensor_pos, obst_pos, obst_radius,
                        eps=1.0, sensing_radius=0.15):
    dx, dy, dz = sensor_pos - obst_pos
    d = np.sqrt(dx * dx + dy * dy + dz * dz)

    if d > sensing_radius + obst_radius:
        return -1.0

    effective_d = max(0.01, d - obst_radius)
    return eps / effective_d


# -----------------------------------------------------------
# Skin site retrieval
# -----------------------------------------------------------
def get_skin_site_ids(model):
    ids = OrderedDict()
    for i in range(model.nsite):
        name = model.site(i).name
        if name and "sensor" in name:
            ids[i] = name
    return ids


# -----------------------------------------------------------
# Ballistic integration (analytical)
# -----------------------------------------------------------
def integrate_ballistic(start, vel0, gravity, T, dt):
    start = np.asarray(start)
    vel0 = np.asarray(vel0)
    gravity = np.asarray(gravity)

    pos = np.zeros((T, 3))
    vel = np.zeros((T, 3))

    for t in range(T):
        tt = t * dt
        pos[t] = start + vel0 * tt + 0.5 * gravity * tt * tt
        vel[t] = vel0 + gravity * tt

    return pos, vel


def sample_structured_obstacle(df, model, data, qpos_robot_traj,
                               target_sids, orig_start, orig_vel0, t_lookup=0):
    """
    Improved structured sampling:
      - sample start within ~1 m of original obstacle start
      - target an upper torso skin site
      - preserve velocity *direction toward target* but keep norm near original
      - light noise to encourage variation but avoid nonsense trajectories
    """

    # -------------------------------
    # 1) Pick a torso target site
    # -------------------------------
    sid = np.random.choice(target_sids)

    nq = qpos_robot_traj.shape[1]
    data.qpos[:nq] = qpos_robot_traj[t_lookup]
    mujoco.mj_forward(model, data)

    site_xyz = np.array(data.site_xpos[sid])

    # -------------------------------
    # 2) Sample start near original (±1 m)
    # -------------------------------
    start = orig_start + np.random.uniform(-1.0, 1.0, size=3)

    # Don't allow it to spawn underground
    start[2] = max(start[2], 0.5)

    # -------------------------------
    # 3) Compute direction toward chosen site
    # -------------------------------
    direction = site_xyz - start
    dnorm = np.linalg.norm(direction)
    if dnorm < 1e-6:
        direction = np.array([1.0, 0.0, 0.2])  # fallback
        dnorm = 1.0
    direction /= dnorm

    # -------------------------------
    # 4) Preserve original speed norm
    # -------------------------------
    orig_speed = np.linalg.norm(orig_vel0)

    # keep speed within ±20%
    speed = orig_speed * np.random.uniform(0.8, 1.2)

    # -------------------------------
    # 5) Final velocity with small angular noise
    # -------------------------------
    noise = 0.15 * np.random.randn(3)
    vel0 = direction * speed + noise

    # -------------------------------
    # 6) Perturb radius slightly (but keep similar)
    # -------------------------------
    orig_radius = float(df["obst_radius"][0])
    radius = orig_radius * np.random.uniform(0.9, 1.1)

    return start, radius, vel0, sid


# -----------------------------------------------------------
# Proximity relevance filter
# -----------------------------------------------------------
def trajectory_is_relevant(model, data, qpos_robot_traj,
                           obst_pos, target_sid, radius):

    nq = qpos_robot_traj.shape[1]
    T = obst_pos.shape[0]

    min_dist = 999
    end_dist = None

    for t in range(T):
        data.qpos[:nq] = qpos_robot_traj[t]
        data.qpos[-7:-4] = obst_pos[t]
        mujoco.mj_forward(model, data)

        site_xyz = np.array(data.site_xpos[target_sid])
        d = np.linalg.norm(site_xyz - obst_pos[t])
        if d < min_dist:
            min_dist = d

    # distance at final timestep
    data.qpos[:nq] = qpos_robot_traj[-1]
    data.qpos[-7:-4] = obst_pos[-1]
    mujoco.mj_forward(model, data)
    end_dist = np.linalg.norm(
        np.array(data.site_xpos[target_sid]) - obst_pos[-1]
    )

    # close enough at some point + far enough at the end
    close_thresh = radius + 0.15
    return (min_dist < close_thresh) and (end_dist > 0.3)

# -----------------------------------------------------------
# Main augmentation for one trajectory
# -----------------------------------------------------------
def augment_one(csv_path, xml_path, out_dir, num_samples, tracker, tracker_data):

    print(f"\n=== AUGMENTING {csv_path} ===")

    df = pd.read_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    gravity = np.array(model.opt.gravity)

    # ---------------------------------
    # Extract robot trajectory
    # ---------------------------------
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qpos_robot_traj = df[qpos_cols].to_numpy()
    T = len(df)

    # ---------------------------------
    # Skin sites
    # ---------------------------------
    skin_sites = get_skin_site_ids(model)
    skin_sids = list(skin_sites.keys())
    skin_names = list(skin_sites.values())

    # ---------------------------------
    # Obstacle freejoint info
    # ---------------------------------
    obst_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obstacle"
    )
    obst_jid = model.body_jntadr[obst_body_id]
    obst_qpos_adr = model.jnt_qposadr[obst_jid]

    # ---------------------------------
    # Augmentation loop
    # ---------------------------------
    samples_created = 0
    attempts = 0
    MAX_ATTEMPTS = num_samples * 4000
    # Initialize/Resume based on tracker
    csv_name = os.path.basename(csv_path)
    initial_samples_created = tracker_data.get(csv_name, 0)
    if initial_samples_created == 0 and csv_name in tracker_data and num_samples > 0:
        # If the tracker says 0 samples created AND we are looking to create more, skip it.
        # If num_samples is 0, we still process to update the tracker state.
        tracker[csv_name] = initial_samples_created
        print(f"Skipping {csv_name}: Previously recorded 0 successful augmentations.")
        return
        
    samples_created = initial_samples_created
    tracker[csv_name] = initial_samples_created

    # ---------------------------------
    # Robot collision check
    # ---------------------------------
    spheres = compute_link_bounding_spheres(model)
    


    while samples_created < num_samples and attempts < MAX_ATTEMPTS:
        attempts += 1
        if attempts % 1000 == 0:
            print(f"  Attempt {attempts}/{MAX_ATTEMPTS}...")
        
        if attempts > 10000 and samples_created == 0:
            print("  Too many attempts without successful sample creation, stopping augmentation for this file.")
            break

        # 1) Sample obstacle parameters
        orig_start = np.array([df["obst_px"][0], df["obst_py"][0], df["obst_pz"][0]])
        orig_vel0 = np.array([df["cmd_vx"][0], df["cmd_vy"][0], df["cmd_vz"][0]])
        start, radius, vel0, target_sid = sample_structured_obstacle(
            df, model, data, qpos_robot_traj, skin_sids, orig_start, orig_vel0
        )

        # 2) Integrate ballistic traj
        obst_pos, obst_vel = integrate_ballistic(
            start, vel0, gravity, T, dt
        )

        # 3) Relevance check
        if not trajectory_is_relevant(
            model, data, qpos_robot_traj,
            obst_pos, target_sid, radius
        ):
            continue

        # 4) Compute capacitance
        cap_new = np.zeros((T, len(skin_sids)))

        coll = False

        for t in range(T):
            nq = qpos_robot_traj.shape[1]
            data.qpos[:nq] = qpos_robot_traj[t]

            if is_robot_collision(model, data,
                                  qpos_robot_traj[t],
                                  obst_pos[t],
                                  obst_qpos_adr,
                                  obst_body_id,
                                  spheres):
                # print("  ✗ Collision detected, resampling...")
                coll = True
                break
            else:
                pass

            # obstacle pose
            data.qpos[obst_qpos_adr:obst_qpos_adr + 7] = [
                1, 0, 0, 0,
                obst_pos[t][0],
                obst_pos[t][1],
                obst_pos[t][2],
            ]
            mujoco.mj_forward(model, data)

            for k, sid in enumerate(skin_sids):
                spos = np.array(data.site_xpos[sid])
                cap_new[t, k] = compute_capacitance(
                    spos, obst_pos[t], radius
                )

        # 5) Build augmented CSV
        if not coll:
            df_aug = df.copy()

            df_aug["obst_px"] = obst_pos[:, 0]
            df_aug["obst_py"] = obst_pos[:, 1]
            df_aug["obst_pz"] = obst_pos[:, 2]
            df_aug["obst_radius"] = radius

            df_aug["obst_vx"] = obst_vel[:, 0]
            df_aug["obst_vy"] = obst_vel[:, 1]
            df_aug["obst_vz"] = obst_vel[:, 2]

            df_aug["cmd_vx"] = vel0[0]
            df_aug["cmd_vy"] = vel0[1]
            df_aug["cmd_vz"] = vel0[2]

            # Capacitance columns — consistent naming
            cap_cols = [
                f"cap_site_{sid}_{skin_sites[sid]}"
                for sid in skin_sids
            ]
            for j, col in enumerate(cap_cols):
                df_aug[col] = cap_new[:, j]

            # Save file
            base = os.path.basename(csv_path)
            fname = f"AUG_{samples_created:02d}_{base}"
            out_path = os.path.join(out_dir, fname)
            df_aug.to_csv(out_path, index=False)

            print(f"✓ Wrote {fname}")
            samples_created += 1
            tracker[csv_name] += 1
            print(f"Created {samples_created}/{num_samples} augmentations so far.")

    print(f"Done. Created {samples_created}/{num_samples} augmentations.")


# -----------------------------------------------------------
# Entry
# -----------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--out", default="augmented")
    parser.add_argument("--samples_per_file", type=int, default=100)
    parser.add_argument("--tracker_path", type=str, default=None,
                    help="Path to previous augmentation_tracker.json/pkl to resume work.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    trajs = sorted(glob.glob(os.path.join(args.traj_dir, "*.csv")))
    if len(trajs) == 0:
        raise RuntimeError(f"No CSV files found in {args.traj_dir}")

    # 1. Load existing tracker data if provided
    tracker_data = {}
    if args.tracker_path:
        tracker_path = args.tracker_path
        if tracker_path.endswith('.json') and os.path.exists(tracker_path):
            with open(tracker_path, 'r') as f:
                tracker_data = json.load(f)
            print(f"Loaded existing tracker data from {tracker_path}.")
        elif tracker_path.endswith('.pkl') and os.path.exists(tracker_path):
            with open(tracker_path, 'rb') as f:
                tracker_data = pickle.load(f)
            print(f"Loaded existing tracker data from {tracker_path}.")
        else:
            print(f"Warning: Tracker file not found at {tracker_path}. Starting fresh.")
    
    # Ensure all file names in tracker data are base names
    tracker_data = {os.path.basename(k): v for k, v in tracker_data.items()}

    tracker = {}
    tracker_json = os.path.join(args.out, "augmentation_tracker.json")
    for traj_path in trajs:
        augment_one(traj_path, args.xml, args.out, args.samples_per_file, tracker, tracker_data)
        with open(tracker_json, 'w') as f:
            json.dump(tracker, f, indent=4)
        # also write as pkl
        tracker_pkl = os.path.join(args.out, "augmentation_tracker.pkl")
        with open(tracker_pkl, 'wb') as f:
            pickle.dump(tracker, f)
    print(tracker)


if __name__ == "__main__":
    main()
