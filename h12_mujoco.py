import time
import argparse
import numpy as np
import pyvista as pv
import mujoco

from mujoco_env import MujocoEnv
from pv_interface import PVInterface
from unitree_interface import SimInterface


def _auto_harness_loop(mujoco_env, stiff, headless=False, duration=300.0,
                       sim_dt=0.0, slow_factor=1.0, trace_out=None,
                       trace_hz=25.0, stop_on_fall=0.0):
    """Automated gantry-harness bring-up + release — the T1 twin-control port of
    the RoboCasa bench harness (HAMS/h1_robocasa/h12_mujoco.py --harness), so the
    IDENTICAL procedure can be compared on the twin, where the stand is known-good:
      spawn joint-hold + stiff xy pin + boosted posture grip
      -> engagement (first STIFF lowcmd, kp-sniffed: the safety layer idles kp=0)
      -> 15 s holdout through the node's scripted bring-up
      -> 10 s grip fade (operator lightening grip)
      -> CHUNKED payout ('.'-style +10 cm presses, quiet-gated, 6 s settle)
      -> honest gate (strap slack + hands <=15 N + assist <=1.6 Nm sustained 2 s)
      -> 3 s quadratic release fade.
    Self-contained: does NOT use MujocoEnv's ElasticBand/sim_step (sim_step clears
    xfrc_applied); steps physics directly at 1x realtime pacing. Constants mirror
    the RoboCasa bench verbatim. Logs use the same format for diffability.
    """
    model, data = mujoco_env.model, mujoco_env.data
    torso = model.body("torso_link").id
    if sim_dt > 0.0:
        # T2 delta-walk: run the twin at a RoboCasa-style timestep. NOTE: must
        # be set before SimInterface reads model.opt.timestep (done in sim_loop);
        # here we just keep the env pacing attribute consistent.
        mujoco_env.timestep = model.opt.timestep
    mujoco.mj_forward(model, data)   # xpos is all-zeros before the first forward
    anchor = np.array(data.xpos[torso]); anchor[2] += 0.8
    K, C = 3000.0, 300.0            # strap (tension-only webbing)
    PK, PKD = 300.0, 60.0           # posture assist, operating gains
    XY_KP, XY_KD = 1000.0, 200.0    # xy pin
    HK, HD = 80.0, 8.0              # pre-engagement joint hold
    BOOST = 5.0                     # posture-grip boost through bring-up
    HOLDOUT, HFADE = 15.0, 10.0
    CHUNK, SETTLE, QUIET = 0.10, 6.0, 35.0
    # REL_TQ recalibrated 1.6 -> 15 (T1b, 2026-07-04): the twin FREE-STANDING
    # (tension 0 N, z 1.01, 218 s) still reads a 7-13 Nm posture-assist floor
    # (300 Nm/rad x ~2 deg natural sway) -- the old 1.6 bar never fires even on
    # the known-good plant. 15 sits above the standing floor and far below the
    # 40+ Nm fight range.
    MIN_T, REL_TQ, SUSTAIN, RFADE = 25.0, 15.0, 2.0, 3.0
    PAY_MAX = 1.5
    rope = 0.8                      # slack 0: taut winch at spawn, twin-rig style
    hold_q0 = np.array(data.qpos[7:34])
    t_cmd0 = None; holdout_done = False; fade_t0 = None
    pay_last = None; pay_quiet_t0 = None
    slack_t0 = None; rel_t0 = None; released = False
    log_t = -1e9
    print(f"[twin-harness] AUTO-HARNESS on torso: anchor {anchor.round(3)}, "
          f"slack 0cm, hands kp/kd {PK:.0f}/{PKD:.0f} (boost x{BOOST:.0f}), "
          f"holdout {HOLDOUT:.0f}s + {HFADE:.0f}s grip fade, chunk "
          f"{CHUNK*100:.0f}cm/{SETTLE:.0f}s, fade {RFADE:.1f}s", flush=True)
    sim_t0 = time.time(); n = 0
    # opt-in qpos trace for render_twin_trace.py -- the PLANT's own ground truth
    # (not the node's belief, not the planner's own-sim). trace_out=None => inert.
    _tr = {"t": [], "qpos": [], "qvel": [], "next": 0.0} if trace_out else None
    _tr_dt = 1.0 / max(trace_hz, 1e-6)
    _fall_since = [None]   # sim time the robot first read FALLEN, for --stop-on-fall
    viewer = mujoco_env.viewer if not headless else None
    while True:
        if viewer is not None and not viewer.is_running():
            break
        if headless and data.time >= duration:
            print(f"[twin-harness] duration {duration:.0f}s reached — done", flush=True)
            break
        if t_cmd0 is None and stiff["kp"] > 1.0:
            t_cmd0 = data.time
            print(f"[twin-harness] harness: controller engaged at sim "
                  f"t={t_cmd0:.1f}s — stiff hands HELD through bring-up (holdout "
                  f"{HOLDOUT:.0f}s + {HFADE:.0f}s grip fade); release gate opens "
                  f"at t={t_cmd0 + MIN_T:.1f}s", flush=True)
        data.xfrc_applied[torso, :] = 0.0
        scale = 1.0
        if rel_t0 is not None and not released:
            s = 1.0 - (data.time - rel_t0) / RFADE
            if s <= 0.0:
                released = True
                print(f"[twin-harness] harness FULLY RELEASED at sim "
                      f"t={data.time:.1f}s — robot is free-standing", flush=True)
            else:
                scale = s * s
        if released:
            # post-release ground truth (the node's view can be clock-offset;
            # this is the twin's own verdict on the free stand)
            if data.time - log_t >= 5.0:
                log_t = data.time
                x = np.array(data.xpos[torso])
                R = data.xmat[torso].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
                fallen = x[2] < 0.5
                print(f"[twin-harness] free-stand: t={data.time:.1f}s "
                      f"z={x[2]:.2f} tilt={tilt:.1f}deg"
                      + ("  *** FALLEN ***" if fallen else ""), flush=True)
                # opt-in early exit: once the robot is DOWN, the rest of the run is
                # dead air (a fallen run otherwise burns the remaining duration, up to
                # ~160 s of the 210 s cap, and every later ladder stage is meaningless
                # anyway -- lean_bound_metrics reports them as "n/a (already down)").
                # Break out of the LOOP rather than killing the process, so the qpos
                # trace still gets written below. 0 = off => byte-identical.
                if stop_on_fall > 0.0:
                    if fallen:
                        if _fall_since[0] is None:
                            _fall_since[0] = data.time
                        elif data.time - _fall_since[0] >= stop_on_fall:
                            print(f"[twin-harness] FALLEN for {stop_on_fall:.0f}s "
                                  f"(since t={_fall_since[0]:.1f}s) — aborting run "
                                  f"early at t={data.time:.1f}s", flush=True)
                            break
                    else:
                        _fall_since[0] = None
        if not released:
            x = np.array(data.xpos[torso]); v = np.array(data.qvel[0:3])
            delta = anchor - x; d = float(np.linalg.norm(delta))
            u = delta / max(d, 1e-9)
            ddot = -float(u @ v)
            mag = K * (d - rope) + C * ddot
            tension = max(0.0, mag) if d >= rope else 0.0
            f = tension * u
            R = data.xmat[torso].reshape(3, 3)
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     torso, vel6, 0)
            grip = 1.0
            if t_cmd0 is None or not holdout_done:
                gp = 1.0
                if fade_t0 is not None:
                    gp = max(0.0, 1.0 - (data.time - fade_t0) / HFADE)
                grip = 1.0 + (BOOST - 1.0) * gp
            tq = (grip * PK * np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
                  - min(grip, 2.0) * PKD * vel6[:3])
            if t_cmd0 is not None and not holdout_done:
                if fade_t0 is None and data.time - t_cmd0 >= HOLDOUT:
                    fade_t0 = data.time
                    print(f"[twin-harness] harness: holdout done at sim "
                          f"t={data.time:.1f}s — fading xy-pin gain to zero over "
                          f"{HFADE:.0f}s", flush=True)
                if fade_t0 is not None and data.time - fade_t0 >= HFADE:
                    holdout_done = True
                    print(f"[twin-harness] harness: HANDS OFF at sim "
                          f"t={data.time:.1f}s — payout + honest gate armed",
                          flush=True)
            if t_cmd0 is None or not holdout_done:
                pin = 1.0
                if fade_t0 is not None:
                    pin = max(0.0, 1.0 - (data.time - fade_t0) / HFADE)
                fxy = (pin * XY_KP * (anchor[:2] - x[:2]) - XY_KD * vel6[3:5])
                if t_cmd0 is None:
                    data.ctrl[:27] = (HK * (hold_q0 - data.qpos[7:34])
                                      - HD * data.qvel[6:33])
            else:
                fxy = -XY_KD * vel6[3:5]
            data.xfrc_applied[torso, 0] = scale * (f[0] + fxy[0])
            data.xfrc_applied[torso, 1] = scale * (f[1] + fxy[1])
            data.xfrc_applied[torso, 2] = scale * f[2]
            data.xfrc_applied[torso, 3:] = scale * tq
            assist = float(np.linalg.norm(tq)); hands_f = float(np.linalg.norm(fxy))
            if data.time - log_t >= 1.0:
                log_t = data.time
                print(f"[twin-harness] harness: t={data.time:.1f}s "
                      f"tension={tension:.0f}N assist={assist:.1f}Nm "
                      f"hands={hands_f:.0f}N scale={scale:.2f} rope={rope:.2f} "
                      f"z={x[2]:.2f}", flush=True)
            if (rel_t0 is None and t_cmd0 is not None and holdout_done
                    and data.time - t_cmd0 >= MIN_T and tension > 5.0
                    and rope < PAY_MAX):
                if assist <= QUIET:
                    if pay_quiet_t0 is None:
                        pay_quiet_t0 = data.time
                    elif (data.time - pay_quiet_t0 >= 1.0
                          and (pay_last is None
                               or data.time - pay_last >= SETTLE)):
                        rope = min(rope + CHUNK, PAY_MAX)
                        pay_last = data.time; pay_quiet_t0 = None
                        print(f"[twin-harness] harness: PAYOUT chunk "
                              f"+{CHUNK*100:.0f}cm at sim t={data.time:.1f}s -> "
                              f"rope {rope:.2f} (tension was {tension:.0f}N; "
                              f"settling {SETTLE:.0f}s)", flush=True)
                else:
                    pay_quiet_t0 = None
            if (rel_t0 is None and t_cmd0 is not None and holdout_done
                    and data.time - t_cmd0 >= MIN_T and tension <= 5.0
                    and hands_f <= 15.0 and assist <= REL_TQ):
                if slack_t0 is None:
                    slack_t0 = data.time
                elif data.time - slack_t0 >= SUSTAIN:
                    rel_t0 = data.time
                    print(f"[twin-harness] harness RELEASING at sim "
                          f"t={data.time:.1f}s (slack + assist {assist:.1f}Nm <= "
                          f"{REL_TQ:.1f}Nm for {SUSTAIN:.1f}s) — easing off over "
                          f"{RFADE:.1f}s", flush=True)
            elif rel_t0 is None:
                slack_t0 = None
        mujoco.mj_step(model, data)
        if _tr is not None and data.time >= _tr["next"]:
            _tr["next"] = data.time + _tr_dt
            _tr["t"].append(float(data.time))
            _tr["qpos"].append(np.array(data.qpos, float))
            _tr["qvel"].append(np.array(data.qvel, float))
        n += 1
        if viewer is not None and n % mujoco_env._render_every == 0:
            viewer.sync()
        # slow_factor < 1 = T3 delta-walk: run the plant DELIBERATELY below
        # realtime (RTF = slow_factor) at its native dt, isolating the
        # real-time-factor variable from the timestep variable (T2 conflated
        # them: dt 0.002 also dropped the twin to ~0.6x RTF).
        ahead = (sim_t0 + n * mujoco_env.timestep / slow_factor) - time.time()
        if ahead > 0:
            time.sleep(ahead)
    if _tr is not None and _tr["t"]:
        # render_twin_trace.py schema: t + base_pos/base_quat/q sliced from qpos.
        Q = np.asarray(_tr["qpos"]); V = np.asarray(_tr["qvel"])
        np.savez_compressed(trace_out, t=np.asarray(_tr["t"]),
                            qpos_full=Q, base_pos=Q[:, 0:3],
                            base_quat=Q[:, 3:7], q=Q[:, 7:34],
                            qvel_full=V, base_vel=V[:, 0:6], dq=V[:, 6:33],
                            meta=np.array([f"source=twin_plant_qpos",
                                           f"dt={model.opt.timestep}"], dtype=object))
        print(f"[twin-harness] trace: {len(_tr['t'])} frames -> {trace_out}", flush=True)


def sim_loop(fixed=False, force_links=None, handless=False, crouch=0.0, magpie=False,
             auto_harness=False, headless=False, duration=300.0, sim_dt=0.0,
             slow_factor=1.0, box_floor=False, scene=None, trace_out=None,
             stop_on_fall=0.0):
    """
    Simulating the robot in mujoco.
    Publishing low state and high state.
    Subscribing to low command.
    Includes end-effector force interface for human interaction simulation

    magpie=True loads the handless model with the two fixed ~0.506 kg magpie
    grippers at the wrists, so the twin matches the MJPC Lean_H12_Magpie planner
    model and the real robot when the grippers are mounted. nu/nq are unchanged
    (mass-only); the grippers are NOT actuated here.
    """
    # initialize mujoco environment
    if fixed:
        if handless:
            scene_path = ("unitree_robots/h1_2/scene_handless_pelvis_fixed_magpie.xml"
                          if magpie else
                          "unitree_robots/h1_2/scene_handless_pelvis_fixed.xml")
        else:
            scene_path = "unitree_robots/h1_2/scene_pelvis_fixed.xml"
        mujoco_env = MujocoEnv(scene_path)
    else:
        if handless:
            scene_path = ("unitree_robots/h1_2/scene_handless_magpie.xml"
                          if magpie else
                          "unitree_robots/h1_2/scene_handless.xml")
            if magpie and box_floor:
                # bench-only parity probe: RoboCasa-kitchen thin-box floor
                # instead of the plane (see the boxfloor scene XML header)
                scene_path = "unitree_robots/h1_2/scene_handless_magpie_boxfloor.xml"
        else:
            scene_path = "unitree_robots/h1_2/scene.xml"
        if scene:
            # explicit scene override (e.g. the TABLE scene the forearm-brace
            # strategy needs). Opt-in: scene=None => byte-identical selection.
            scene_path = scene
            print(f"[h12_mujoco] SCENE OVERRIDE: {scene_path}", flush=True)
        mujoco_env = MujocoEnv(scene_path)
        if magpie and box_floor:
            # RoboCasa contact regime: box-mesh narrowphase yields ONE contact
            # per foot (no ankle moment) unless multiccd is on; the bench runs
            # --multiccd, so the faithful probe does too (4 pts/foot, probed).
            mujoco_env.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
            print("[h12_mujoco] BOX-FLOOR parity probe: kitchen thin-box floor + "
                  "multiccd (RoboCasa regime)")
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
            # tension_only: a gantry strap pulls, never pushes. With rest length 0 the
            # strap is ALWAYS taut (1000 N/m x distance-to-anchor -- it hauled a standing
            # robot up by ~25% bodyweight -> feet lifted, ankle-tau estop, 2026-06-12 run).
            # Release gesture = pay out rope with '.' (+10cm/press) until GREEN/slack;
            # tension-only makes the paid-out rope a pure catch instead of a downward shove.
            # release_fade: SPACE eases the band off over 6s instead of an instant cut.
            # Run-5 forensics: the strap carried ~560 of 660 N during the "stand" and
            # SPACE dropped it in one tick -> 80%-bodyweight step -> ankle tau estop.
            # The PRIMARY release is still '.' payout (keeps the rope as a catch);
            # the fade just makes a premature SPACE survivable.
            if not auto_harness:
                mujoco_env.init_elastic_band("torso_link",
                                             point=np.array([0.0, 0.0, 1.695]),
                                             tension_only=True, release_fade=10.0)
    # AUTO-HARNESS T2 delta-walk: override the plant timestep BEFORE SimInterface
    # reads model.opt.timestep (its publish interval + tick=int(t/dt) follow it;
    # the node must be passed the matching --twin_dt).
    if auto_harness and sim_dt > 0.0:
        mujoco_env.model.opt.timestep = sim_dt
        mujoco_env.timestep = sim_dt
        print(f"[twin-harness] plant timestep OVERRIDDEN to {sim_dt} "
              f"(T2 delta-walk; pass --twin_dt {sim_dt} to the node)")

    # initialize sdk interface
    sim_interface = SimInterface(mujoco_env.model, mujoco_env.data)

    # AUTO-HARNESS (T1 twin-control, 2026-07-04): run the RoboCasa bench's
    # automated gantry procedure against the twin — self-contained loop, no
    # ElasticBand (the band was intentionally NOT initialized above for this
    # mode). Engagement is kp-sniffed like the crouch path: the safety layer
    # idles kp=0 zeros on rt/lowcmd, so "first message" != engagement.
    if auto_harness:
        from unitree_sdk2py.core.channel import ChannelSubscriber as _CS2
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as _LC2
        _ah_stiff = {"kp": 0.0}

        def _ah_kp_sniff(msg):
            try:
                kmax = max(m.kp for m in msg.motor_cmd[:27])
                if kmax > _ah_stiff["kp"]:
                    _ah_stiff["kp"] = kmax
            except Exception:
                pass
        _ah_sub = _CS2("rt/lowcmd", _LC2)
        _ah_sub.Init(_ah_kp_sniff, 10)
        if not headless:
            mujoco_env.launch_viewer()
        _auto_harness_loop(mujoco_env, _ah_stiff, headless=headless,
                           duration=duration, sim_dt=sim_dt,
                           slow_factor=slow_factor, trace_out=trace_out,
                           stop_on_fall=stop_on_fall)
        return

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
        # strap readout (~1/s): the band's rest length is 0 by default, so it is a
        # permanently-taut winch unless rope is paid out with '.' -- a taut strap on a
        # standing robot hauls it up (feet lift, ankle/shoulder estops; 2026-06-12 runs).
        # Static estimate (damping omitted): tension = k * (dist - rest_length).
        b = mujoco_env.elastic_band
        if b is not None and b.enabled and n_steps % 200 == 0:
            # measure from the pelvis free-joint position -- the SAME point
            # evaluate_force uses, so the printed N == the applied N
            d = float(np.linalg.norm(np.asarray(mujoco_env.data.qpos[:3]) - b.point))
            assist = (f" | tilt-assist {b.last_assist_mag:4.1f} Nm"
                      if (b.posture_kp > 0 or b.posture_kd > 0) else "")
            if d < b.length:
                print(f"[strap] SLACK/GREEN (pure catch -- robot is free){assist}")
            else:
                # show the APPLIED tension (fade-scaled) -- during a release the
                # geometric tension stays high while the applied force ramps to zero
                tension = (b.stiffness * (d - b.length)
                           * getattr(b, 'fade_scale', 1.0))
                fading = ' (FADING)' if b.release_requested else ''
                need = max(1, int(np.ceil((d - b.length) / 0.1)))
                print(f"[strap] TAUT hauling ~{tension:4.0f} N{fading} -- "
                      f"'.' x{need} pays out to slack{assist}")

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
    parser.add_argument(
        "--auto-harness", dest="auto_harness", action="store_true",
        help="T1 twin-control: run the RoboCasa bench's AUTOMATED gantry "
             "procedure (joint-hold + boosted grip -> kp-sniffed engagement -> "
             "15s holdout + 10s grip fade -> chunked '.'-style payout -> honest "
             "slack gate -> 3s release fade) instead of the manual hang band. "
             "Constants mirror HAMS/h1_robocasa/h12_mujoco.py --harness verbatim.")
    parser.add_argument(
        "--headless", action="store_true",
        help="(with --auto-harness) run without the viewer; ends after --duration")
    parser.add_argument(
        "--duration", type=float, default=300.0,
        help="(with --auto-harness --headless) sim seconds to run before exiting")
    parser.add_argument(
        "--slow-factor", type=float, default=1.0,
        help="(with --auto-harness) deliberately pace the plant BELOW realtime "
             "(RTF = this factor) at its native dt -- T3 delta-walk isolating "
             "real-time-factor from timestep. e.g. 0.5")
    parser.add_argument(
        "--sim-dt", type=float, default=0.0,
        help="(with --auto-harness) override the plant timestep, e.g. 0.002 for "
             "the RoboCasa delta-walk (T2). 0 = keep the twin's 0.005. The "
             "lowstate tick + publish interval follow it; pass the same value "
             "as --twin_dt to the node.")
    parser.add_argument(
        "--magpie",
        action="store_true",
        help="Load the handless model WITH the two fixed magpie grippers (~0.506 kg each "
             "at the wrists), matching the MJPC --task 'Lean H12 Magpie' planner model and "
             "the real robot when the grippers are mounted. Mass-only, nu/nq unchanged. "
             "Only affects the handless scenes (ignored with --inspire).",
    )
    parser.add_argument(
        "--box-floor", action="store_true",
        help="(bench parity probe, --magpie only) replace the plane floor with "
             "the RoboCasa kitchen's thin-box walking surface (3.86x3.088x0.02, "
             "top at z=0) -- isolates box-vs-plane foot contact narrowphase.")
    parser.add_argument(
        "--scene", default=None,
        help="explicit scene XML (h1_mujoco-relative or absolute), overriding the "
             "--handless/--magpie selection. Use for the forearm-brace bench: "
             "unitree_robots/h1_2/scene_handless_magpie_table.xml")
    parser.add_argument(
        "--trace-out", dest="trace_out", default=None,
        help="(with --auto-harness) write a qpos trace npz of the PLANT's own "
             "ground truth for render_twin_trace.py / twin_mindmap.py")
    parser.add_argument(
        "--stop-on-fall", dest="stop_on_fall", type=float, default=0.0,
        metavar="SEC",
        help="(with --auto-harness) end the run early once the robot has read "
             "FALLEN continuously for SEC seconds, instead of playing out the rest "
             "of --duration as dead air (a fallen 210s run wasted up to ~160s, and "
             "every later ladder stage is meaningless anyway). The qpos trace is "
             "still written. 0 = off, byte-identical.")
    args = parser.parse_args()

    sim_loop(fixed=args.fixed, force_links=args.force, handless=args.handless,
             crouch=args.crouch, magpie=args.magpie,
             auto_harness=args.auto_harness, headless=args.headless,
             duration=args.duration, sim_dt=args.sim_dt,
             slow_factor=args.slow_factor, box_floor=args.box_floor,
             scene=args.scene, trace_out=args.trace_out,
             stop_on_fall=args.stop_on_fall)
