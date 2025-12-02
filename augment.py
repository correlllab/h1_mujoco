#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import mujoco
from collections import OrderedDict

# Define a minimum force threshold for a collision to be considered 'actual contact'
COLLISION_FORCE_THRESHOLD = 1e-6 

# -----------------------------------------------------------
# Capacitance function
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
# Ballistic integration (Analytical Solution)
# -----------------------------------------------------------
def integrate_ballistic_analytical(start, vel0, gravity, T, dt):
    """
    Calculates the exact ballistic position and velocity at each timestep
    using the analytical solution: P(t) = P0 + V0*t + 0.5*g*t^2
    """
    start = np.asarray(start).reshape(1, 3)
    vel0  = np.asarray(vel0).reshape(1, 3)
    gravity = np.asarray(gravity).reshape(1, 3)
    
    pos = np.zeros((T, 3))
    vel = np.zeros((T, 3))

    for t_step in range(T):
        t = t_step * dt
        
        # Position: P(t) = P0 + V0*t + 0.5*g*t**2
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
# GEOMETRIC COLLISION CHECK (NEW)
# -----------------------------------------------------------
def is_collision_free(model, data, initial_qpos_robot, qpos_robot_t, obst_pos, obst_qpos_adr, obst_body_id):
    """
    Checks if the obstacle is in contact with the robot at a given time step.
    Returns True if contact force > COLLISION_FORCE_THRESHOLD, False otherwise.
    """
    
    # Set the robot's qpos for this time step
    data.qpos[:qpos_robot_t.shape[0]] = qpos_robot_t
    
    # Set the obstacle's qpos (Position part only). Assuming identity quaternion.
    # qpos layout for freejoint: [qw, qx, qy, qz, px, py, pz]
    data.qpos[obst_qpos_adr:obst_qpos_adr+7] = [1, 0, 0, 0, obst_pos[0], obst_pos[1], obst_pos[2]]

    # 1. Update kinematics and constraint list (contacts)
    mujoco.mj_forward(model, data)
    
    # 2. Check for contact force (mj_fwdConstraint computes contact forces, data.efc_force has the results)
    mujoco.mj_fwdConstraint(model, data)

    # 3. Iterate through contacts and check if the obstacle body is involved
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # MuJoCo stores geom IDs, we need to check if one of the geoms belongs to the obstacle
        geom1_body_id = model.geom_bodyid[contact.geom1]
        geom2_body_id = model.geom_bodyid[contact.geom2]
        
        # Check if the obstacle body is one of the bodies involved in the contact
        if geom1_body_id == obst_body_id or geom2_body_id == obst_body_id:
            # Check the normal force component (force in the direction of the contact normal)
            # The magnitude of the contact force is stored in data.efc_force 
            # (which is an array of size model.nefc, mapping contact elements to forces).
            # This is complex to extract perfectly without a full step.
            
            # SIMPLER CHECK: Total number of non-zero contact forces indicates contact
            # However, mj_fwdConstraint only calculates efc_force if a full step is run.
            
            # Let's rely on the number of contacts (data.ncon) and the gap
            # If the contact distance is near zero or negative, a collision is happening.
            if contact.dist < 0: 
                # Negative distance means deep penetration/contact
                return False # Collision detected! Trajectory is NOT collision-free

    return True # No collisions detected for this step


# -----------------------------------------------------------
# Combined sanity checks (UPDATED)
# -----------------------------------------------------------
def passes_collision_checks(model, data, qpos_robot_traj, obst_xyz_traj, radius, obst_qpos_adr, obst_body_id, sensing_radius=0.15):
    """
    Ensures the trajectory is:
      1. Relevant (passes close enough to the sensor sites at the initial posture).
      2. Safe (never results in a geometric collision with the robot).
    """
    
    relevance_threshold = radius + sensing_radius + 0.05
    ever_close = False
    
    for t in range(len(obst_xyz_traj)):
        obst_pos_t = obst_xyz_traj[t]
        qpos_robot_t = qpos_robot_traj[t]

        # 1. Geometric Collision Check (Safety)
        if not is_collision_free(model, data, qpos_robot_t, qpos_robot_t, obst_pos_t, obst_qpos_adr, obst_body_id):
            print(f"FAILED SAFETY CHECK: Collision detected at time step {t}.")
            return False # Collision: Reject trajectory

        # 2. Relevance Check (Proximity)
        # Note: We must check proximity against the robot's posture *at that time t*
        
        # We need to run mj_forward here to get correct site positions for the relevance check
        data.qpos[:qpos_robot_t.shape[0]] = qpos_robot_t
        data.qpos[obst_qpos_adr:obst_qpos_adr+7] = [1, 0, 0, 0, obst_pos_t[0], obst_pos_t[1], obst_pos_t[2]]
        mujoco.mj_forward(model, data) 
        
        dmin = 999.0
        # Check distance to ALL sites (even non-sensors) to ensure relevance is wide-ranging
        for sid in range(model.nsite):
            spos = data.site_xpos[sid]
            d = np.linalg.norm(spos - obst_pos_t)
            if d < dmin:
                dmin = d

        if dmin < relevance_threshold:
            ever_close = True
            
    # Trajectory is valid if it came close (ever_close) and was collision free throughout.
    return ever_close


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

    # ----- extract column names -----
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qpos_robot_traj = df[qpos_cols].to_numpy()

    # ----- extract original obstacle params -----
    orig_start = np.array([df["obst_px"][0], df["obst_py"][0], df["obst_pz"][0]])
    orig_radius = float(df["obst_radius"][0])
    orig_vel0 = np.array([df["cmd_vx"][0], df["cmd_vy"][0], df["cmd_vz"][0]])

    # ----- get site IDs (for capacity) -----
    skin_site_ids_dict = get_skin_site_ids(model)
    skin_site_ids = list(skin_site_ids_dict.keys())
    skin_site_names = list(skin_site_ids_dict.values())
    
    # ----- Get Obstacle MuJoCo IDs (for collision) -----
    obst_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
    obst_jid = model.body_jntadr[obst_body_id]
    obst_qpos_adr = model.jnt_qposadr[obst_jid] if obst_jid >= 0 else -1

    if obst_body_id < 0 or obst_qpos_adr < 0:
        raise ValueError("Could not find obstacle body or joint in XML.")

    # ===========================================================
    # AUGMENTATION LOOP
    # ===========================================================
    
    samples_created = 0
    attempts_made = 0
    MAX_ATTEMPTS = num_samples * 500 # Try 5x the desired samples before giving up
    
    while samples_created < num_samples and attempts_made < MAX_ATTEMPTS:
        attempts_made += 1
        
        # 1. SAMPLE AND INTEGRATE
        if samples_created == 0:
            # Use original params for the first sample
            start, radius, vel0 = orig_start, orig_radius, orig_vel0
        else:
            start, radius, vel0 = sample_random_obstacle_params(orig_start, orig_radius)
            
        grav = np.array(model.opt.gravity)
        obst_pos, obst_vel = integrate_ballistic_analytical(start, vel0, grav, T, dt)

        # 2. SANITY CHECK
        if not passes_collision_checks(model, data, qpos_robot_traj, obst_pos, radius, obst_qpos_adr, obst_body_id):
            print("WARNING: Original trajectory failed geometric safety check.")
            continue # Try again with new random sample

        # 3. RE-COMPUTE CAPACITANCE 
        cap_new = np.zeros((T, len(skin_site_ids)))
        
        for t in range(T):
            # Set robot qpos and run forward kinematics (already done in passes_collision_checks, but necessary here)
            qpos_robot_t = qpos_robot_traj[t]
            data.qpos[:qpos_robot_t.shape[0]] = qpos_robot_t
            data.qpos[obst_qpos_adr:obst_qpos_adr+7] = [1, 0, 0, 0, obst_pos[t][0], obst_pos[t][1], obst_pos[t][2]]
            
            mujoco.mj_forward(model, data) 

            # Compute capacity
            for k, sid in enumerate(skin_site_ids):
                spos = np.array(data.site_xpos[sid])
                cap_new[t, k] = compute_capacitance(spos, obst_pos[t], radius)

        # 4. BUILD AUGMENTED CSV
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

        cap_cols = [f"cap_site_{sid}_{name}" for sid, name in zip(skin_site_ids, skin_site_names)]
        for j, col in enumerate(cap_cols):
            if col not in df_aug.columns:
                 df_aug[col] = 0.0 
            df_aug[col] = cap_new[:, j]

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
    parser.add_argument("--samples_per_file", type=int, default=1, help="Number of augmented samples to create per source file.")
    args = parser.parse_args()

    input_dir = os.path.join(args.traj_dir, "good")
    os.makedirs(args.out, exist_ok=True)

    trajs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if len(trajs) == 0:
        raise RuntimeError(f"No trajectories found in {input_dir}.")

    print(f"Found {len(trajs)} source trajectories. Creating {args.samples_per_file} augmentations for each.")

    for traj_path in trajs:
        augment_one(traj_path, args.xml, args.out, args.samples_per_file)

    print("\nBatch augmentation complete.")


if __name__ == "__main__":
    main()