import builtins
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_env import MujocoEnv


MODEL_XML = '''
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="torso_link" pos="0.1 -0.2 0.7">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="child" pos="0.2 0 0">
        <joint name="hinge" type="hinge" ref="0.2"/>
        <geom type="sphere" size="0.05" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
'''


@pytest.fixture
def model():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    # Scene composition may set the initial free-body pose after compilation.
    model.qpos0[:3] = [0.25, -0.15, 1.5]
    return model


def test_constructor_requires_exactly_one_source(model):
    with pytest.raises(ValueError, match='exactly one'):
        MujocoEnv()
    with pytest.raises(ValueError, match='exactly one'):
        MujocoEnv('unused.xml', model=model)


def test_legacy_xml_timestep(tmp_path):
    xml_path = tmp_path / 'model.xml'
    xml_path.write_text(MODEL_XML)

    env = MujocoEnv(str(xml_path))

    assert env.timestep == pytest.approx(0.005)
    assert env.model.opt.timestep == pytest.approx(0.005)
    assert env.viewer is None
    np.testing.assert_allclose(env.data.qpos, env.model.qpos0)
    np.testing.assert_allclose(env.data.body('torso_link').xpos,
                               env.model.qpos0[:3])


def test_compiled_model_preserves_timestep_and_initial_pose(model):
    env = MujocoEnv(model=model)

    assert env.model is model
    assert env.timestep == pytest.approx(0.002)
    assert model.opt.timestep == pytest.approx(0.002)
    assert env.viewer is None
    np.testing.assert_allclose(env.data.qpos, model.qpos0)
    np.testing.assert_allclose(env.data.body('torso_link').xpos, model.qpos0[:3])


def test_reset_restores_model_defaults_and_clears_forces(model):
    env = MujocoEnv(model=model)
    env.init_external_force('torso_link')
    env.external_force.active = True
    env.external_force.torque[:] = 2
    env.data.qpos[:3] = 4
    env.data.qvel[:] = 3
    env.data.ctrl[:] = 2
    env.data.qfrc_applied[:] = 5
    env.data.xfrc_applied[:] = 6
    env.data.time = 7
    env.paused = True

    env.reset()

    np.testing.assert_allclose(env.data.qpos, model.qpos0)
    np.testing.assert_allclose(env.data.body('torso_link').xpos, model.qpos0[:3])
    np.testing.assert_array_equal(env.data.qvel, 0)
    np.testing.assert_array_equal(env.data.ctrl, 0)
    np.testing.assert_array_equal(env.data.qfrc_applied, 0)
    np.testing.assert_array_equal(env.data.xfrc_applied, 0)
    assert env.data.time == 0
    assert not env.paused
    assert not env.external_force.active
    assert env.external_force.magnitude == 0
    np.testing.assert_array_equal(env.external_force.torque, 0)
    env.sim_step()
    np.testing.assert_array_equal(env.data.xfrc_applied, 0)


def test_reset_key_resets_simulation(model):
    from mujoco.glfw import glfw

    env = MujocoEnv(model=model)
    env.data.time = 1
    env.paused = True

    env.key_callback(glfw.KEY_R)

    assert env.data.time == 0
    assert not env.paused


def test_headless_steps_preserve_caller_forces_without_accumulating(model):
    env = MujocoEnv(model=model)
    env.init_external_force('torso_link', magnitude=2)
    env.external_force.active = True
    env.init_elastic_band('torso_link', stiffness=1, damping=1)
    env.data.xfrc_applied[1, 1] = 0.5
    caller_forces = env.data.xfrc_applied.copy()

    for _ in range(10):
        env.sim_step()
        np.testing.assert_array_equal(env.data.xfrc_applied, caller_forces)

    assert env.data.time == pytest.approx(10 * env.timestep)
    assert np.isfinite(env.data.qpos).all()
    assert np.isfinite(env.data.qvel).all()
    assert np.isfinite(env.data.qacc).all()
    assert env.viewer is None


@pytest.mark.parametrize('args,expected_code,message', [
    (['--help'], 0, '--headless-steps'),
    (['--scene', 'scene.yaml', '--handless'], 2, 'cannot be used with --handless'),
    (['--headless-steps', '-1'], 2, 'must be non-negative'),
])
def test_cli_validation_needs_no_third_party_packages(args, expected_code, message):
    script = Path(__file__).resolve().parents[1] / 'h12_mujoco.py'
    result = subprocess.run([sys.executable, '-S', str(script), *args],
                            capture_output=True, text=True, check=False)

    assert result.returncode == expected_code
    assert message in result.stdout + result.stderr


@pytest.mark.parametrize('fixed', [False, True])
def test_scene_headless_cli_uses_builder_without_sdk(model, monkeypatch, capsys, fixed):
    import h12_mujoco

    calls = []
    config = object()

    def load_scene_config(scene):
        calls.append(('load', scene))
        return config

    def build_magpie_scene(loaded_config, *, fixed):
        calls.append(('build', loaded_config, fixed))
        return model

    monkeypatch.setitem(sys.modules, 'scene_builder', SimpleNamespace(
        load_scene_config=load_scene_config, build_magpie_scene=build_magpie_scene))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(('unitree', 'pyvista', 'pv_interface', 'mujoco.viewer'))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)

    h12_mujoco.sim_loop(scene='scene.yaml', fixed=fixed, headless_steps=3)

    assert calls == [('load', 'scene.yaml'), ('build', config, fixed)]
    output = capsys.readouterr().out
    assert 'steps=3, time=0.006000, finite=True' in output
    assert ('Elastic band initialized!' in output) == (not fixed)


def test_headless_cli_rejects_mujoco_automatic_instability_reset(model, monkeypatch, tmp_path):
    import h12_mujoco

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, 'scene_builder', SimpleNamespace(
        load_scene_config=lambda _: None,
        build_magpie_scene=lambda *args, **kwargs: model))
    original_step = MujocoEnv.sim_step

    def bad_step(env):
        env.data.qpos[0] = np.nan
        original_step(env)

    monkeypatch.setattr(MujocoEnv, 'sim_step', bad_step)
    previous_warning = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: None)
    try:
        with pytest.raises(RuntimeError, match='instability: mjWARN_BADQPOS'):
            h12_mujoco.sim_loop(scene='test', fixed=True, headless_steps=1)
    finally:
        mujoco.set_mju_user_warning(previous_warning)


def test_headless_cli_detects_unexpected_time_reset(model, monkeypatch):
    import h12_mujoco

    monkeypatch.setitem(sys.modules, 'scene_builder', SimpleNamespace(
        load_scene_config=lambda _: None,
        build_magpie_scene=lambda *args, **kwargs: model))
    monkeypatch.setattr(MujocoEnv, 'sim_step', lambda env: env.reset())
    with pytest.raises(RuntimeError, match='time reset'):
        h12_mujoco.sim_loop(scene='test', fixed=True, headless_steps=2)


def test_viewer_sync_and_overlays_are_limited_without_skipping_physics(model, monkeypatch):
    import mujoco_env

    clock = [10.0]
    calls = []
    monkeypatch.setattr(mujoco_env.time, 'monotonic', lambda: clock[0])
    env = MujocoEnv(model=model)
    env.viewer = SimpleNamespace(lock=nullcontext, sync=lambda: calls.append('sync'))
    for method in ('reset_geom_buffer', 'draw_elastic_band', 'draw_external_force'):
        monkeypatch.setattr(env, method, lambda name=method: calls.append(name))
    for step in range(100):
        clock[0] = 10.0 + step * 0.001
        env.sim_step()

    assert env.data.time == pytest.approx(100 * model.opt.timestep)
    assert calls.count('sync') == 6
    assert calls == ['reset_geom_buffer', 'draw_elastic_band', 'draw_external_force', 'sync'] * 6


def test_viewer_sync_remains_responsive_while_paused_and_after_reset(model, monkeypatch):
    import mujoco_env

    clock = [10.0]
    sync_times = []
    monkeypatch.setattr(mujoco_env.time, 'monotonic', lambda: clock[0])
    env = MujocoEnv(model=model)
    env.viewer = SimpleNamespace(lock=nullcontext, user_scn=SimpleNamespace(ngeom=0),
                                 sync=lambda: sync_times.append(clock[0]))
    env.data.time = 100
    env.sync_viewer()
    env.paused = True
    clock[0] += 0.02
    env.sync_viewer()
    env.reset()
    assert env.data.time == 0
    clock[0] += 0.02
    env.sync_viewer()
    assert len(sync_times) == 3
