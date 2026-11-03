"""World-fixed fixtures using measured pelvis translation and ideal XYZ axes."""

import mujoco
import numpy as np
import pytest

from scene_builder import build_magpie_scene, load_scene_config


# Live DDS snapshot on 2026-11-03 at tick 122542; not the XML's initial height.
PELVIS_POSITION = np.array([-0.0740056829810623, -0.027101960939199664, 0.9551722399174063])
# Deliberately ignore the measured rotation for ideal workspace calibration.
PELVIS_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module", params=["left", "right"])
def fixture_scene(request):
    side = request.param
    config = load_scene_config(f"magpie_peg_in_hole_{side}")
    model = build_magpie_scene(config)
    data = mujoco.MjData(model)
    root_joint = model.body("pelvis").jntadr[0]
    adr = model.jnt_qposadr[root_joint]
    data.qpos[adr:adr + 3] = PELVIS_POSITION
    data.qpos[adr + 3:adr + 7] = PELVIS_QUATERNION
    mujoco.mj_forward(model, data)
    return side, config, model, data


def _pelvis_relative(data, world_position):
    pelvis = data.body("pelvis")
    return pelvis.xmat.reshape(3, 3).T @ (world_position - pelvis.xpos)


def test_single_cylinder_top_near_workspace_lower_bound(fixture_scene):
    side, config, model, data = fixture_scene
    assert len(config.board["pegs"]) == 1
    assert config.board["pegs"][0]["hole"] == "cylinder_1"
    tip = data.site("task_peg_cylinder_1_tip").xpos
    expected_tip = [0.45, 0.225 if side == "left" else -0.225, 0.003]
    np.testing.assert_allclose(_pelvis_relative(data, tip), expected_tip, atol=1e-10, rtol=0)
    np.testing.assert_allclose(_pelvis_relative(data, tip + [0, 0, 0.05]),
                               [*expected_tip[:2], 0.053], atol=1e-10, rtol=0)


def test_grasp_end_stays_in_easy_region_during_extraction(fixture_scene):
    side, config, model, data = fixture_scene
    hole = config.catalog["holes"]["cylinder_1"]
    entry = data.site("task_" + hole["entry_site"]).xpos
    tip = data.site("task_peg_cylinder_1_tip").xpos
    # Source cylinder is 50 mm long, with the tip at local z=0.
    free_end = tip + data.body("task_peg_cylinder_1_peg").xmat.reshape(3, 3) @ [0, 0, 0.05]
    lower = [0.4, 0.125 if side == "left" else -0.325, 0.05]
    upper = [0.5, 0.325 if side == "left" else -0.125, 0.225]
    free_end_at_entry = free_end + entry - tip
    assert _pelvis_relative(data, entry)[2] == pytest.approx(0.025)
    for position in (free_end, free_end_at_entry):
        relative = _pelvis_relative(data, position)
        assert np.all(relative >= lower)
        assert np.all(relative <= upper)


def test_level_board_is_supported_and_table_stays_forward(fixture_scene):
    side, config, model, data = fixture_scene
    np.testing.assert_allclose(data.body("task_board").xmat.reshape(3, 3), np.eye(3), atol=1e-12)
    table = config.table
    assert table["position"][2] == 0
    assert table["position"][0] - table["size"][0] / 2 - PELVIS_POSITION[0] > 0.35
    np.testing.assert_allclose(np.array(table["size"]) - config.catalog["board"]["footprint"],
                               [0.04, 0.04], atol=1e-12)
    assert data.body("task_board").xpos[2] == pytest.approx(table["height"])
    assert data.site("task_peg_cylinder_1_tip").xpos[2] == pytest.approx(table["height"] + 0.003)


def test_lower_table_leaves_extraction_and_planning_headroom(fixture_scene):
    side, config, model, data = fixture_scene
    tabletop_relative_z = config.table["height"] - PELVIS_POSITION[2]
    assert tabletop_relative_z == pytest.approx(0.0, abs=1e-10)
    assert tabletop_relative_z <= 1e-10
    tip_z = _pelvis_relative(data, data.site("task_peg_cylinder_1_tip").xpos)[2]
    seated_top_z = tip_z + 0.05
    extracted_top_z = seated_top_z + config.catalog["holes"]["cylinder_1"]["seated_depth"]
    assert seated_top_z == pytest.approx(0.053)
    assert extracted_top_z == pytest.approx(0.075)
    assert 0.225 - extracted_top_z == pytest.approx(0.150)
