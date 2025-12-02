#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import mujoco
from collections import OrderedDict

# -----------------------------------------------------------
# Capacitance function (Kept from replay.py)
# -----------------------------------------------------------
def compute_capacitance(sensor_pos, obst_pos, obst_radius, eps=1.0, sensing_radius=0.15):
    dx, dy, dz = sensor_pos - obst_pos
    d = np.sqrt(dx*dx + dy*dy + dz*dz)

    if d > sensing_radius + obst_radius:
        return -1.0

    # Ensure effective distance is at least 0.01 to prevent division by zero / high values
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
# Ballistic integration (Analytical Solution - High Accuracy)
# -----------------------------------------------------------
def integrate_ballistic_analytical(start, vel0, gravity, T, dt):
    """
    Calculates the exact ballistic position and velocity at each timestep
    using the analytical solution: P(t) = P0 + V0*t + 0.5*g*t^2
    This replaces the less accurate Euler integration.
    """
    start = np.asarray(start).reshape(1, 3)
    vel0  = np.asarray(vel0).reshape(1, 3)
    gravity = np.asarray(gravity).reshape(1, 3)
    
    pos = np.zeros((T, 3))
    vel = np.zeros((T, 3))

    for t_step in range(T):
        t = t_step * dt
        
        # Position: P(t) = P0 + V0*t + 0.5*g*t^2
        pos[t_step] = start + vel0 * t + 0.5 * gravity * t**2
        
        # Velocity: V(t) = V0 + g*t
        vel[t_step] = vel0 + gravity * t
        
    return pos.squeeze(), vel.squeeze()


# -----------------------------------------------------------
# Random sampling helpers
# -----------------------------------------------------------
def random_unit_vector():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)


def sample_random_obstacle_params(orig_start, orig_radius):
    """
    Samples perturbed obstacle start state, radius, and velocity.
    """
    # --- position perturbation (±50 cm cube around original start)
    start = orig_start + np.random.uniform(-0.5, 0.5, size=3)

    # --- radius perturbation (±25%)
    radius = float(orig_radius) * np.random.uniform(0.75, 1.25)

    # --- speed: 1.0–8.0 m/s
    speed = np.random.uniform(1.0, 8.0)

    # --- random direction (normalized)
    direction = random_unit_vector()

    return start, radius, direction * speed


# -----------------------------------------------------------
# Collision sanity checks (FIXED)
# -----------------------------------------------------------
def passes_collision_checks(model, data, initial_qpos_robot, skin_site_ids, obst_xyz_traj, radius, sensing_radius=0.15):
    """
    Ensures the obstacle trajectory is 'relevant' by checking if it enters the
    robot's sensory sphere at its initial position, and then leaves it.
    
    The check is performed against the robot's INITIAL/HOME posture (t=0)
    to ensure site positions are valid and fixed for the check.
    """
    
    # Set robot state to initial posture from the log (t=0)
    qpos_cols = [c for c in initial_qpos_robot.index if c.startswith("qpos_")]
    for i, col in enumerate(qpos_cols):
        data.qpos[i] = initial_qpos_robot[col]
        
    # Run forward kinematics ONCE to get correct, fixed site positions
    mujoco.mj_forward(model, data)
    
    relevance_threshold = radius + sensing_radius + 0.2  # Obstacle must come within 0.2m of sensing range
    
    ever_close = False
    eventually_far = False
    close_count = 0
    far_count = 0

    for t in range(len(obst_xyz_traj)):
        obst_pos = obst_xyz_traj[t]
        
        # Compute closest distance to any skin site (dmin)
        dmin = 999.0
        for sid in skin_site_ids:
            spos = data.site_xpos[sid] # Site positions are now fixed by the initial mj_forward call
            d = np.linalg.norm(spos - obst_pos)
            if d < dmin:
                dmin = d

        if dmin < relevance_threshold:
            close_count += 1
            ever_close = True
        else:
            far_count += 1
            # If it was close earlier (at least 10 steps), and is now far, it's a valid trajectory
            if close_count > 10: 
                eventually_far = True

    # Trajectory is valid if it came close to the initial robot posture AND moved past it
    return ever_close and eventually_far


# -----------------------------------------------------------
# MAIN augmentation
# -----------------------------------------------------------
def augment_one(csv_path, xml_path, out_dir, num_samples):
    print(f"\n=== AUGMENTING {csv_path} ===")

    df = pd.read_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    T = len(df)
    
    # Store initial robot QPOS for the collision check function
    initial_qpos_robot = df.iloc[0]

    # ----- extract column names -----
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qvel_cols = [c for c in df.columns if c.startswith("qvel_")]

    # ----- extract original obstacle params -----
    orig_start = np.array([df["obst_px"][0], df["obst_py"][0], df["obst_pz"][0]])
    orig_radius = float(df["obst_radius"][0])
    orig_vel0 = np.array([df["cmd_vx"][0], df["cmd_vy"][0], df["cmd_vz"][0]])

    # ----- get skin sites -----
    skin_site_ids_dict = get_skin_site_ids(model)
    skin_site_ids = list(skin_site_ids_dict.keys())
    skin_site_names = list(skin_site_ids_dict.values())

    # ===========================================================
    # AUGMENTATION LOOP
    # ===========================================================
    
    samples_created = 0
    
    while samples_created < num_samples:
        
        # 1. SAMPLE AND INTEGRATE
        # Use original params for the first sample (0), then perturb
        if samples_created == 0:
            start, radius, vel0 = orig_start, orig_radius, orig_vel0
            print("INFO: Sample 1 is the original trajectory with analytical integration.")
        else:
            start, radius, vel0 = sample_random_obstacle_params(orig_start, orig_radius)
            
        grav = np.array(model.opt.gravity)
        # Use the analytical solution for the trajectory
        obst_pos, obst_vel = integrate_ballistic_analytical(start, vel0, grav, T, dt)

        # 2. SANITY CHECK (Skip check for original trajectory sample 0)
        if samples_created > 0 and not passes_collision_checks(model, data, initial_qpos_robot, skin_site_ids, obst_pos, radius):
            print(f"Attempt {samples_created + 1} failed relevance check. Retrying...")
            continue # Try again with new random sample


        # 3. RE-COMPUTE CAPACITANCE FOR AUGMENTED OBSTACLE TRAJ
        cap_new = np.zeros((T, len(skin_site_ids)))
        
        for t in range(T):
            # Set robot qpos for time t
            for i, col in enumerate(qpos_cols):
                data.qpos[i] = df[col][t]

            # Set obstacle qpos (position only) for time t. QVEL is not needed for capacity.
            # Assuming QPOS is [qw, qx, qy, qz, px, py, pz]
            # We skip setting QPOS/QVEL for obstacle's freejoint to avoid index complexity here, 
            # as only the robot's qpos affects its site_xpos. We manually pass obst_pos[t] to compute_capacitance.
            
            # Run forward kinematics
            mujoco.mj_forward(model, data) 

            # Compute capacity
            for k, sid in enumerate(skin_site_ids):
                spos = np.array(data.site_xpos[sid])
                cap_new[t, k] = compute_capacitance(spos, obst_pos[t], radius)

        # 4. BUILD AUGMENTED CSV
        df_aug = df.copy()

        # overwrite obstacle columns (using calculated trajectory)
        df_aug["obst_px"] = obst_pos[:, 0]
        df_aug["obst_py"] = obst_pos[:, 1]
        df_aug["obst_pz"] = obst_pos[:, 2]
        df_aug["obst_radius"] = radius
        df_aug["obst_vx"] = obst_vel[:, 0] # Save instantaneous velocity
        df_aug["obst_vy"] = obst_vel[:, 1]
        df_aug["obst_vz"] = obst_vel[:, 2]
        
        # Overwrite commanded velocity (set once at launch)
        df_aug["cmd_vx"] = vel0[0]
        df_aug["cmd_vy"] = vel0[1]
        df_aug["cmd_vz"] = vel0[2]

        # overwrite capacitance columns
        cap_cols = [f"cap_site_{sid}_{name}" for sid, name in zip(skin_site_ids, skin_site_names)]
        for j, col in enumerate(cap_cols):
            # Ensure the column exists or create it, though typically logs have placeholders
            if col not in df_aug.columns:
                 df_aug[col] = 0.0 
            df_aug[col] = cap_new[:, j]

        # Save file
        base_name = os.path.basename(csv_path)
        new_name = f"AUG_{samples_created:02d}_{base_name}"
        out_path = os.path.join(out_dir, new_name)
        
        df_aug.to_csv(out_path, index=False)
        print(f"✓ Wrote augmented CSV -> {new_name}")
        
        samples_created += 1


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="Path to the MuJoCo XML file (e.g., avoid_h12.xml).")
    parser.add_argument("--traj_dir", required=True, help="Path to the parent directory (e.g., traj_logs).")
    parser.add_argument("--out", default="augmented", help="Output directory for augmented files.")
    parser.add_argument("--samples_per_file", type=int, default=5, help="Number of augmented samples to create per source file.")
    args = parser.parse_args()

    # Define the input directory for 'good' trajectories
    input_dir = os.path.join(args.traj_dir,"")
    os.makedirs(args.out, exist_ok=True)

    # Find all good trajectories
    trajs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if len(trajs) == 0:
        raise RuntimeError(f"No trajectories found in {input_dir}.")

    print(f"Found {len(trajs)} source trajectories. Creating {args.samples_per_file} augmentations for each.")

    for traj_path in trajs:
        augment_one(traj_path, args.xml, args.out, args.samples_per_file)

    print("\nBatch augmentation complete.")


if __name__ == "__main__":
    main()