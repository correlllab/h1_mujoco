#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

LOG_DIR = "/home/wxie/workspace/mujoco_mpc/traj_logs"
OUT_DIR = "/home/wxie/workspace/h1_mujoco/traj_logs"

def dedupe_csv(path):
    df = pd.read_csv(path)
    print("Processing:", path)
    # remove first row of data because it is stale
    df = df.iloc[1:]
    
    # Identify robot-only qpos and qvel columns
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    qvel_cols = [c for c in df.columns if c.startswith("qvel_")]

    robot_state = df[qpos_cols + qvel_cols].to_numpy()

    # Remove rows where robot state matches the previous row
    keep = [True]
    for i in range(1, len(df)):
        same = np.allclose(robot_state[i], robot_state[i-1], atol=1e-9)
        keep.append(not same)
    

    df2 = df[keep]

    out = path.replace(".csv", "_dedup.csv")
    out = path.replace(LOG_DIR, OUT_DIR)
    df2.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} → {len(df2)} rows)")


if __name__ == "__main__":
    for fn in os.listdir(LOG_DIR):
        if fn.endswith(".csv"):
            dedupe_csv(os.path.join(LOG_DIR, fn))
