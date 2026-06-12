import time
import argparse
import numpy as np
import pyvista as pv
import mujoco

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface


def sim_loop(fixed=False, force_links=None, handless=False, crouch=0.0):
    """
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    Includes end-effector force interface for human interaction simulation
    """
    # initialize mujoco environment
    if fixed:
        if handless:
            scene_path = "unitree_robots/h1_2/scene_handless_pelvis_fixed.xml"
        else:
            scene_path = "unitree_robots/h1_2/scene_pelvis_fixed.xml"
        mujoco_env = MujocoEnv(scene_path)
    else:
        if handless:
            scene_path = "unitree_robots/h1_2/scene_handless.xml"
        else:
            scene_path = "unitree_robots/h1_2/scene.xml"
        mujoco_env = MujocoEnv(scene_path)
        if crouch > 0:
            # ---- bring-up rehearsal: start CROUCHED on the floor, like the real robot's
            # off-home power-on. Flat feet + upright torso: the pitch angles cancel along
            # the leg chain (real robot's observed crouch signs: hipP -0.36 knee +0.96 ankP -0.55).
            k = crouch
            for hp, kn, an in ((8, 10, 11), (14, 16, 17)):   # L/R hip_pitch, knee, ankle_pitch
                mujoco_env.data.qpos[hp] = -k / 2
                mujoco_env.data.qpos[kn] = k
                mujoco_env.data.qpos[an] = -k / 2
            mujoco.mj_forward(mujoco_env.model, mujoco_env.data)
            ankle_ids = [mujoco_env.model.body(n).id
                         for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
            low = min(float(mujoco_env.data.xpos[b][2]) for b in ankle_ids)
            mujoco_env.data.qpos[2] -= (low - 0.047)   # ankle-body height at floor contact (FK calib)
            mujoco.mj_forward(mujoco_env.model, mujoco_env.data)
            # Harness, not bungee: TAUT strap anchored above the CROUCHED torso (zero preload --
            # it auto-slackens as the robot rises, turning GREEN = catch-only) + tilt assist
            # (the strap-stiffness/steadying-hands part). SPACE after standing = full release.
            torso0 = np.array(mujoco_env.data.xpos[mujoco_env.model.body("torso_link").id])
            # assist 300/60 = the PROVEN-STABLE gains (stiffer oscillates/NaNs -- see
            # mujoco_env note). The rise tilts ~10 deg and the policy recovers to ~4 deg
            # leaning lightly on the assist; the 3 s release FADE hands that load over
            # quasi-statically (an instant cut is a ~25 N*m step -- bench: topple in 1.4 s).
            mujoco_env.init_elastic_band(
                "torso_link", point=torso0 + np.array([0.0, 0.0, 0.8]), length=0.8,
                stiffness=3000, damping=300, tension_only=True,
                posture_kp=300, posture_kd=60, release_fade=3.0)
            print(f"[crouch rehearsal] start pose: knee=+{k:.2f} hipP/ankP=-{k/2:.2f} "
                  f"base_z={float(mujoco_env.data.qpos[2]):.3f}")
            print("[crouch rehearsal] pose FROZEN until the node's first stiff command; "
                  "harness slackens as it rises (band turns GREEN); SPACE once standing "
                  "eases the support off over 3s")
        else:
            # Suspension anchor: torso hangs at z = anchor_z - M*g/stiffness, so the
            # hang height tracks anchor_z 1:1 (independent of mass). Headless-measured
            # foot-sole vs floor: anchor 1.688 = feet just touching (0 cm), and the
            # slope is exactly 1:1, so 1.695 puts the feet ~0.5 cm above the floor ->
            # a near-zero drop on SPACE-release. (was 1.73 ~= 4-5 cm; default [0,0,2.0]
            # ~= 8 cm. To raise the gap, raise anchor_z by the same cm; lower to lower.)
            mujoco_env.init_elastic_band("torso_link", point=np.array([0.0, 0.0, 1.695]))
    # initialize sdk interface
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    # crouch rehearsal: hold the pose (no gravity integration) until the node's first STIFF
    # command -- limp zero-kp joints would crumple the crouch before the chain is up. The
    # safety layer's idle 500 Hz publisher sends kp=0 zeros, so gate on kp > 1 via a second
    # rt/lowcmd reader (DDS allows many; same proven pattern as the headless bench).
    _stiff = {"kp": 0.0}
    if crouch > 0 and not fixed:
        from unitree_sdk2py.core.channel import ChannelSubscriber as _CS
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as _LC

        def _kp_sniff(msg):
            try:
                kmax = max(m.kp for m in msg.motor_cmd[:27])
                if kmax > _stiff["kp"]:
                    _stiff["kp"] = kmax
            except Exception:
                pass
        _sniff_sub = _CS("rt/lowcmd", _LC)
        _sniff_sub.Init(_kp_sniff, 10)

    # initialize external force interface if link names are provided
    if force_links:
        mujoco_env.init_external_force(force_links, magnitude=5.0)

    # # define body name & id for wrench calculation (optional)
    # body_name = 'left_wrist_yaw_link'
    # body_id = mujoco_env.model.body(body_name).id

    # # pyvista visualization (optional)
    # pv_interface = PVInterface(mujoco_env.model, mujoco_env.data)
    # pv_interface.track_body(body_name)

    print("Press P to pause/resume the simulation loop")

    # launch viewer
    mujoco_env.launch_viewer()
    # main simulation loop -- paced to real-time CUMULATIVELY: sleep until n_steps*timestep of
    # wall-clock has elapsed since the anchor, so the occasional render frame's cost is absorbed by
    # the slack of the cheap physics-only steps (true ~1.0x, no per-step drift). Physics still steps
    # exactly once per iteration, so the sim trajectory is unchanged -- only wall-clock spacing moves.
    sim_t0 = time.time()
    n_steps = 0
    crouch_frozen = crouch > 0 and not fixed
    hold_qpos = np.array(mujoco_env.data.qpos).copy()
    frz_i = 0
    while mujoco_env.viewer.is_running():
        if mujoco_env.paused:
            # keep viewer responsive while paused; rebase the clock so we don't sprint on resume
            mujoco_env.viewer.sync()
            time.sleep(1.0 / 60.0)
            sim_t0 = time.time()
            n_steps = 0
            continue

        if crouch_frozen:
            if _stiff["kp"] > 1.0:
                crouch_frozen = False
                print(f"[crouch rehearsal] first STIFF command (kp={_stiff['kp']:.0f}) "
                      "-- physics live, node ramp rising to home")
                sim_t0 = time.time()
                n_steps = 0
            else:
                mujoco_env.data.qpos[:] = hold_qpos
                mujoco_env.data.qvel[:] = 0.0
                # freeze the POSE, not the CLOCK: the lowstate tick must keep advancing or a
                # node on --use_twin_time plans against a frozen clock and lurches on the jump
                mujoco_env.data.time += mujoco_env.timestep
                mujoco.mj_forward(mujoco_env.model, mujoco_env.data)
                frz_i += 1
                if frz_i % mujoco_env._render_every == 0:   # throttled draw (software GL is slow)
                    mujoco_env.viewer.sync()
                time.sleep(mujoco_env.timestep)
                continue

        mujoco_env.sim_step()
        n_steps += 1

        # # optional: get and display wrench on end-effector
        # force, torque = mujoco_env.get_body_wrench(body_id)
        # print(f'Force: {force}, Torque: {torque}')

        # # optional: update pyvista visualization
        # pv_interface.update_vector(force)
        # pv_interface.pv_render()

        ahead = (sim_t0 + n_steps * mujoco_env.timestep) - time.time()
        if ahead > 0:
            time.sleep(ahead)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1-2 Mujoco simulation")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use pelvis-fixed scene without elastic band",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--handless",
        dest="handless",
        action="store_true",
        help="Load handless model scene files (default)",
    )
    model_group.add_argument(
        "--inspire",
        dest="handless",
        action="store_false",
        help="Load inspire-hand model scene files",
    )
    # Default to handless model when no model option is provided.
    parser.set_defaults(handless=True)
    parser.add_argument(
        "--force",
        nargs="+",
        default=None,
        help="Enable external force interface for specified link names",
    )
    parser.add_argument(
        "--crouch",
        type=float,
        default=0.0,
        help="bring-up rehearsal: start CROUCHED on the floor (knee=+K rad), pose frozen "
             "until the node engages, taut catch-only harness + tilt assist instead of the "
             "hang band; SPACE releases the harness once standing. e.g. --crouch 0.7",
    )
    args = parser.parse_args()

    sim_loop(fixed=args.fixed, force_links=args.force, handless=args.handless,
             crouch=args.crouch)
