from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import yaml

import scene_builder as builder
from mujoco_env import MujocoEnv


SCENES = ("magpie_table", "magpie_peg_in_hole")
SHAPES = ("l", "rectangle", "triangle", "hex", "cylinder")


def _raw_scene(name):
    return yaml.safe_load((builder.SCENE_ROOT / f"{name}.yaml").read_text())


def _all_peg_placements():
    return [
        {"id": f"{shape}_{size}", "shape": shape, "size": size, "hole": f"{shape}_{size}"}
        for shape in SHAPES for size in range(4)
    ]


def _write_config(tmp_path, config):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(config))
    return path.resolve()


@pytest.fixture
def table_config():
    return _raw_scene("magpie_table")


@pytest.fixture
def peg_config():
    config = _raw_scene("magpie_peg_in_hole")
    config["board"]["pegs"] = _all_peg_placements()
    return config


@pytest.fixture(scope="session")
def baseline_models():
    return {
        fixed: mujoco.MjModel.from_xml_path(str(builder.ASSET_ROOT / "mujoco_assets" / filename))
        for fixed, filename in (
            (False, "scene_h1_2_magpie.xml"),
            (True, "scene_h1_2_magpie_pelvis_fixed.xml"),
        )
    }


@pytest.fixture(scope="session")
def composed_scene():
    cache = {}

    def get(name, fixed=False):
        key = name, fixed
        if key not in cache:
            config = builder.load_scene_config(name)
            if name == "magpie_peg_in_hole":
                pegs = _all_peg_placements()
                for peg in pegs:
                    peg["insertion_depth"] = config.catalog["holes"][peg["hole"]]["seated_depth"]
                config = replace(config, board={**config.board, "pegs": pegs})
            cache[key] = config, builder.build_magpie_scene(config, fixed=fixed)
        return cache[key]

    return get


def test_default_peg_scene_has_one_seated_cylinder():
    config = builder.load_scene_config("magpie_peg_in_hole")
    assert config.board["pegs"] == [{
        "id": "cylinder_1",
        "shape": "cylinder",
        "size": 1,
        "hole": "cylinder_1",
        "insertion_depth": config.catalog["holes"]["cylinder_1"]["seated_depth"],
    }]


@pytest.mark.parametrize("name", SCENES)
@pytest.mark.parametrize("form", ["bare", "extension", "root_relative", "dot_relative", "absolute", "absolute_bare"])
def test_scene_resolution_is_independent_of_cwd(tmp_path, monkeypatch, name, form):
    expected = (builder.SCENE_ROOT / f"{name}.yaml").resolve()
    choices = {
        "bare": name,
        "extension": f"{name}.yaml",
        "root_relative": f"scene/{name}.yaml",
        "dot_relative": f"./scene/{name}",
        "absolute": expected,
        "absolute_bare": expected.with_suffix(""),
    }
    # A same-named cwd file must not shadow the project configuration.
    (tmp_path / f"{name}.yaml").write_text("not: the project scene\n")
    monkeypatch.chdir(tmp_path)

    assert builder.resolve_scene_path(choices[form]) == expected


@pytest.mark.parametrize("value", ["", "scene/", "scene/example.yml", "example.xml", "example.YAML"])
def test_scene_resolution_rejects_invalid_filenames(value):
    with pytest.raises(ValueError, match="filename|extension|yaml"):
        builder.resolve_scene_path(value)


def test_scene_resolution_reports_missing_absolute_and_bare_paths(tmp_path):
    for value in ("scene_builder_missing_scene", tmp_path / "missing.yaml"):
        with pytest.raises(FileNotFoundError, match="resolved to missing file"):
            builder.resolve_scene_path(value)
    directory = tmp_path / "directory.yaml"
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        builder.resolve_scene_path(directory)


def test_asset_resolution_uses_only_asset_root(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    asset = nested / "component.xml"
    asset.write_text("<mujoco/>")
    (root / "alias.xml").symlink_to(asset)
    monkeypatch.setattr(builder, "ASSET_ROOT", root)
    monkeypatch.chdir(nested)

    assert builder.resolve_asset_path("nested/component.xml") == asset.resolve()
    assert builder.resolve_asset_path("./nested/component.xml") == asset.resolve()
    assert builder.resolve_asset_path("alias.xml") == asset.resolve()
    with pytest.raises(FileNotFoundError, match="custom.asset: asset not found"):
        builder.resolve_asset_path("missing.xml", "custom.asset")
    with pytest.raises(FileNotFoundError):
        builder.resolve_asset_path("nested")


@pytest.mark.parametrize("value", [None, 1, True, "", "../outside.xml", "nested/../../outside.xml", "/tmp/outside.xml"])
def test_asset_resolution_rejects_absolute_and_traversal_paths(value):
    with pytest.raises(ValueError, match="asset-root-relative|CL_Assets-relative"):
        builder.resolve_asset_path(value)


def test_asset_resolution_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.xml"
    outside.write_text("<mujoco/>")
    (root / "escape.xml").symlink_to(outside)
    (root / "escape_directory").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(builder, "ASSET_ROOT", root)

    for path in ("escape.xml", "escape_directory/outside.xml"):
        with pytest.raises(ValueError, match="asset escapes CL_Assets"):
            builder.resolve_asset_path(path)


@pytest.mark.parametrize("tag", ["include", "attach", "replicate", "model"])
def test_components_cannot_load_indirect_external_assets(tmp_path, tag):
    path = tmp_path / "component.xml"
    path.write_text(f'<mujoco><{tag} file="outside.xml"/></mujoco>')
    with pytest.raises(ValueError, match="components must be standalone"):
        builder._component(path)


@pytest.mark.parametrize("tag,attribute", [("mesh", "file"), ("texture", "file"), ("texture", "fileleft")])
def test_component_file_attributes_cannot_escape_asset_root(tmp_path, monkeypatch, tag, attribute):
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.asset"
    outside.write_text("not accessible from components")
    (root / "escape.asset").symlink_to(outside)
    path = root / "component.xml"
    path.write_text(f'<mujoco><asset><{tag} name="external" {attribute}="escape.asset"/></asset></mujoco>')
    monkeypatch.setattr(builder, "ASSET_ROOT", root)
    with pytest.raises(ValueError, match="asset escapes CL_Assets"):
        builder._component(path)


@pytest.mark.parametrize("tag", ["body", "site"])
@pytest.mark.parametrize("attribute,value", [
    ("euler", "0 0 1"), ("axisangle", "0 0 1 1"),
    ("xyaxes", "1 0 0 0 1 0"), ("zaxis", "0 0 1"),
])
def test_component_orientations_require_explicit_quaternions(tmp_path, tag, attribute, value):
    path = tmp_path / "component.xml"
    path.write_text(f'<mujoco><worldbody><{tag} name="oriented" {attribute}="{value}"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="orientations must use explicit quaternions"):
        builder._component(path)


def test_tip_pose_is_relative_to_peg_not_its_old_world_pose():
    spec = mujoco.MjSpec.from_string('''
        <mujoco><worldbody><body name="peg" pos="0.1 0.2 0.3" quat="0 0 0 1">
          <site name="tip" pos="0.02 0.03 0.04" quat="0 1 0 0"/>
        </body></worldbody></mujoco>
    ''')
    pos, quat = builder._site_pose(spec, "tip", "peg", relative_to_body=True)
    np.testing.assert_allclose(pos, [0.02, 0.03, 0.04])
    np.testing.assert_allclose(quat, [0, 1, 0, 0])
    world_pos, _ = builder._site_pose(spec, "tip", "peg")
    np.testing.assert_allclose(world_pos, [0.08, 0.17, 0.34])


@pytest.mark.parametrize("xml", [
    '<body name="peg"><body name="child"><site name="tip"/></body></body>',
    '<body name="parent"><body name="peg"><site name="tip"/></body></body>',
    '<body name="peg"><frame pos="1 0 0"><site name="tip"/></frame></body>',
    '<body name="peg"/>',
])
def test_component_sites_must_be_directly_on_top_level_body(xml):
    spec = mujoco.MjSpec.from_string(f"<mujoco><worldbody>{xml}</worldbody></mujoco>")
    with pytest.raises(ValueError, match="expected a site directly on component body"):
        builder._site_pose(spec, "tip", "peg")


def test_freejoint_must_belong_to_named_peg(tmp_path, monkeypatch, peg_config):
    peg_config["board"]["pegs"] = [peg_config["board"]["pegs"][0]]
    config = builder.load_scene_config(_write_config(tmp_path, peg_config))
    board_path = builder.resolve_asset_path(
        config.catalog["board"]["collision_xmls"][config.board["collision"]])

    def component(path):
        if path == config.scene_xml:
            return mujoco.MjSpec.from_string("<mujoco/>")
        if path == board_path:
            spec = mujoco.MjSpec()
            body = spec.worldbody.add_body(name="board")
            for hole in config.catalog["holes"].values():
                body.add_site(name=hole["entry_site"])
                body.add_site(name=hole["target_site"])
            return spec
        return mujoco.MjSpec.from_string('''
            <mujoco><worldbody>
              <body name="peg"><site name="tip"/></body>
              <body name="wrong_owner"><freejoint/><geom type="sphere" size="0.01"/></body>
            </worldbody></mujoco>
        ''')

    monkeypatch.setattr(builder, "_component", component)
    with pytest.raises(ValueError, match="free joint must belong to body 'peg'"):
        builder.build_magpie_scene(config)


@pytest.mark.parametrize("config", [None, [], "not a mapping", {1: "not a string key"}])
def test_yaml_requires_mapping(tmp_path, config):
    with pytest.raises(ValueError, match="mapping with string keys"):
        builder.load_scene_config(_write_config(tmp_path, config))


def test_malformed_yaml_is_not_silently_accepted(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("version: 1\ntable: [\n")
    with pytest.raises(yaml.YAMLError):
        builder.load_scene_config(path.resolve())


@pytest.mark.parametrize("version", [None, False, True, 0, 2, 1.0, "1"])
def test_schema_version_is_exact_integer_one(tmp_path, table_config, version):
    table_config["version"] = version
    with pytest.raises(ValueError, match="version: expected scene schema version 1"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("section", [None, "scene", "simulation", "table", "board"])
def test_unknown_yaml_keys_are_rejected(tmp_path, table_config, section):
    target = table_config if section is None else table_config[section]
    target["typo"] = 1
    with pytest.raises(ValueError, match="unknown fields: typo"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("section", ["scene", "simulation", "table", "board"])
@pytest.mark.parametrize("value", [None, [], "enabled"])
def test_sections_require_mappings(tmp_path, table_config, section, value):
    table_config[section] = value
    with pytest.raises(ValueError, match=rf"{section}: expected a mapping"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "timestep", 0),
    ("simulation", "timestep", -0.001),
    ("simulation", "timestep", True),
    ("simulation", "timestep", float("inf")),
    ("simulation", "timestep", float("nan")),
    ("table", "position", [0, 0]),
    ("table", "position", "0 0 0"),
    ("table", "position", [0, True, 0]),
    ("table", "position", [0, float("nan"), 0]),
    ("table", "quaternion", [1, 0, 0]),
    ("table", "quaternion", [0, 0, 0, 0]),
    ("table", "quaternion", [2, 0, 0, 0]),
    ("table", "quaternion", [1, 0, 0, float("inf")]),
    ("table", "size", [0.8]),
    ("table", "size", [0, 0.7]),
    ("table", "size", [-0.8, 0.7]),
    ("table", "size", [0.11, 0.7]),
    ("table", "height", 0),
    ("table", "top_thickness", 0),
    ("table", "top_thickness", 1.05),
    ("table", "leg_width", -0.05),
    ("table", "leg_width", True),
    ("table", "leg_inset", -0.01),
    ("table", "rgba", [0.5, 0.5, 1.1, 1]),
    ("table", "rgba", [0.5, 0.5, -0.1, 1]),
    ("table", "rgba", [0.5, 0.5, 1]),
    ("table", "friction", [0.8, -0.01, 0]),
    ("table", "friction", [0.8, 0.01]),
    ("table", "friction", [0.8, float("inf"), 0]),
    ("board", "position_on_table", [0, 0]),
    ("board", "position_on_table", [0, 0, False]),
    ("board", "quaternion", [0, 0, 0, 0]),
    ("board", "quaternion", [1, 0, 0]),
    ("board", "collision", "approximate"),
    ("board", "collision", None),
    ("board", "collision", True),
])
def test_invalid_physical_parameters(tmp_path, table_config, section, key, value):
    table_config[section][key] = value
    with pytest.raises(ValueError, match=rf"{section}(\.|:)"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("section", ["table", "board"])
@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_enabled_requires_yaml_boolean(tmp_path, table_config, section, value):
    table_config[section]["enabled"] = value
    with pytest.raises(ValueError, match=rf"{section}.enabled: expected true or false"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


def test_minimal_configuration_defaults_and_nondefault_values(tmp_path):
    config = builder.load_scene_config(_write_config(tmp_path, {"version": 1}))
    assert not config.table["enabled"]
    assert not config.board["enabled"]
    assert config.catalog is None
    assert config.timestep == pytest.approx(0.005)
    assert config.scene_xml == builder.resolve_asset_path("mujoco_assets/manipulation_net/workcell.xml")

    raw = _raw_scene("magpie_table")
    raw["simulation"]["timestep"] = 0.002
    raw["table"]["quaternion"] = [-1, 0, 0, 0]
    raw["table"]["friction"] = [0, 0, 0]
    raw["table"]["leg_inset"] = 0
    config = builder.load_scene_config(_write_config(tmp_path, raw))
    assert config.timestep == pytest.approx(0.002)
    assert config.table["quaternion"] == [-1, 0, 0, 0]
    assert config.table["friction"] == [0, 0, 0]


def test_yaml_asset_references_are_not_relative_to_yaml(tmp_path, table_config):
    (tmp_path / "workcell.xml").write_text("<mujoco/>")
    table_config["scene"]["xml"] = "workcell.xml"
    with pytest.raises(FileNotFoundError, match="scene.xml: asset not found"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("value", ["../workcell.xml", "/tmp/workcell.xml"])
def test_yaml_rejects_unsafe_asset_reference(tmp_path, table_config, value):
    table_config["scene"]["xml"] = value
    with pytest.raises(ValueError, match="scene.xml: use a CL_Assets-relative path"):
        builder.load_scene_config(_write_config(tmp_path, table_config))


@pytest.mark.parametrize("case,message", [
    ("no_table", "requires an enabled table"),
    ("disabled", "cannot place pegs on a disabled board"),
    ("not_list", "expected a list"),
])
def test_board_enablement_and_peg_list(tmp_path, peg_config, case, message):
    if case == "no_table":
        peg_config["table"]["enabled"] = False
    elif case == "disabled":
        peg_config["board"]["enabled"] = False
    else:
        peg_config["board"]["pegs"] = {}
    with pytest.raises(ValueError, match=message):
        builder.load_scene_config(_write_config(tmp_path, peg_config))


@pytest.mark.parametrize("key,value,message", [
    ("id", "", "letters, digits or underscores"),
    ("id", "1peg", "letters, digits or underscores"),
    ("id", "peg-with-dash", "letters, digits or underscores"),
    ("id", None, "letters, digits or underscores"),
    ("shape", "circle", "expected shape"),
    ("size", True, "expected shape"),
    ("size", 0.0, "expected shape"),
    ("size", -1, "expected shape"),
    ("size", 4, "expected shape"),
    ("hole", "missing", "unknown hole"),
    ("hole", None, "unknown hole"),
    ("hole", "rectangle_0", "incompatible"),
    ("insertion_depth", -0.001, "must be at least"),
    ("insertion_depth", 0.023, "exceeds hole depth"),
    ("insertion_depth", True, "expected a finite number"),
    ("insertion_depth", float("nan"), "expected a finite number"),
    ("insertion_depth", float("inf"), "expected a finite number"),
    ("typo", 0, "unknown fields"),
])
def test_invalid_peg_placements(tmp_path, peg_config, key, value, message):
    peg = deepcopy(peg_config["board"]["pegs"][0])
    peg[key] = value
    peg_config["board"]["pegs"] = [peg]
    with pytest.raises(ValueError, match=message):
        builder.load_scene_config(_write_config(tmp_path, peg_config))


@pytest.mark.parametrize("case,message", [
    ("duplicate", "duplicate peg id"),
    ("occupied", "already occupied"),
    ("not_mapping", "expected a mapping"),
])
def test_duplicate_or_malformed_peg_placements(tmp_path, peg_config, case, message):
    peg = deepcopy(peg_config["board"]["pegs"][0])
    other = deepcopy(peg)
    if case == "occupied":
        other["id"] = "different_id"
    elif case == "not_mapping":
        other = None
    peg_config["board"]["pegs"] = [peg, other]
    with pytest.raises(ValueError, match=message):
        builder.load_scene_config(_write_config(tmp_path, peg_config))


@pytest.mark.parametrize("depth", [0, 0.01, 0.022])
def test_insertion_depth_inclusive_boundaries(tmp_path, peg_config, depth):
    peg_config["board"]["pegs"] = [peg_config["board"]["pegs"][0]]
    peg_config["board"]["pegs"][0]["insertion_depth"] = depth
    config = builder.load_scene_config(_write_config(tmp_path, peg_config))
    assert config.board["pegs"][0]["insertion_depth"] == pytest.approx(depth)


@pytest.mark.parametrize("name", SCENES)
@pytest.mark.parametrize("fixed", [False, True])
def test_robot_indices_actuators_sensors_and_equalities_are_preserved(composed_scene, baseline_models, name, fixed):
    config, model = composed_scene(name, fixed)
    baseline = baseline_models[fixed]
    count = len(config.board["pegs"])
    assert (baseline.nq, baseline.nv, baseline.nu, baseline.nsensor) == (46, 45, 31, 101)
    assert (model.nq, model.nv, model.nu, model.nsensor) == (46 + 7 * count, 45 + 6 * count, 31, 101)
    assert model.opt.timestep == pytest.approx(config.timestep)
    np.testing.assert_allclose(model.qpos0[:baseline.nq], baseline.qpos0)
    for i in range(baseline.njnt):
        assert model.joint(i).name == baseline.joint(i).name
        assert model.jnt_qposadr[i] == baseline.jnt_qposadr[i]
        assert model.jnt_dofadr[i] == baseline.jnt_dofadr[i]
        assert model.jnt_type[i] == baseline.jnt_type[i]
    for i in range(baseline.nu):
        assert model.actuator(i).name == baseline.actuator(i).name
    for field in ("actuator_trnid", "actuator_trntype", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
                  "sensor_type", "sensor_objtype", "sensor_objid", "sensor_adr", "sensor_dim"):
        np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
    for i in range(baseline.nsensor):
        assert model.sensor(i).name == baseline.sensor(i).name
    assert model.neq == baseline.neq
    # This includes Magpie's four-bar connections, not just joint counts.
    for field in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_active0", "eq_solref", "eq_solimp"):
        np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
    welds = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    assert len(welds) == int(fixed)
    if fixed:
        weld = welds[0]
        assert model.eq_obj1id[weld] == model.body("pelvis").id
        assert model.eq_obj2id[weld] == 0


@pytest.mark.parametrize("name", SCENES)
def test_table_dimensions_friction_and_mount(composed_scene, name):
    config, model = composed_scene(name)
    table = config.table
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    width, length = table["size"]
    height, thickness, leg = (table[key] for key in ("height", "top_thickness", "leg_width"))
    top = model.geom("task_table_top")
    assert top.type == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(top.size, [width / 2, length / 2, thickness / 2])
    np.testing.assert_allclose(top.pos, [0, 0, height - thickness / 2])
    np.testing.assert_allclose(top.friction, table["friction"])
    np.testing.assert_allclose(top.rgba, table["rgba"], atol=1e-7)
    np.testing.assert_allclose(data.body("task_table").xpos, table["position"])
    np.testing.assert_allclose(data.site("task_tabletop").xpos, np.array(table["position"]) + [0, 0, height])
    for ix, sign_x in enumerate((-1, 1)):
        for iy, sign_y in enumerate((-1, 1)):
            geom = model.geom(f"task_table_leg_{ix}_{iy}")
            np.testing.assert_allclose(geom.size, [leg / 2, leg / 2, (height - thickness) / 2])
            np.testing.assert_allclose(geom.pos, [sign_x * (width / 2 - table["leg_inset"] - leg / 2),
                                                sign_y * (length / 2 - table["leg_inset"] - leg / 2),
                                                (height - thickness) / 2])
            np.testing.assert_allclose(geom.friction, table["friction"])


def _assert_peg_poses(config, model, data):
    for index, peg in enumerate(config.board["pegs"]):
        prefix = f"task_peg_{peg['id']}_"
        joint = model.joint(prefix + "freejoint")
        assert joint.type == mujoco.mjtJoint.mjJNT_FREE
        assert joint.qposadr == 46 + 7 * index
        assert joint.dofadr == 45 + 6 * index
        assert model.body(prefix + "peg").parentid == 0
        hole = config.catalog["holes"][peg["hole"]]
        entry = data.site("task_" + hole["entry_site"])
        tip = data.site(prefix + "tip")
        rotation = entry.xmat.reshape(3, 3)
        local_tip = rotation.T @ (tip.xpos - entry.xpos)
        np.testing.assert_allclose(local_tip, [0, 0, -peg["insertion_depth"]], atol=1e-10)
        np.testing.assert_allclose(tip.xmat, entry.xmat, atol=1e-10)


def test_catalog_matches_calibrated_board_and_all_clearance_rows(composed_scene):
    config, model = composed_scene("magpie_peg_in_hole")
    catalog = config.catalog
    assert catalog["version"] == 1
    np.testing.assert_allclose(catalog["board"]["footprint"], [0.32, 0.25])
    assert len(config.board["pegs"]) == 20
    assert {(peg["shape"], peg["size"]) for peg in config.board["pegs"]} == {
        (shape, size) for shape in SHAPES for size in range(4)
    }
    variants = {}
    for shape in SHAPES:
        family_dimensions = []
        for size in range(4):
            hole = catalog["holes"][f"{shape}_{size}"]
            assert (hole["shape"], hole["size"]) == (shape, size)
            assert hole["max_insertion_depth"] == pytest.approx(0.022)
            assert hole["seated_depth"] == pytest.approx(0.022)
            entry = model.site("task_" + hole["entry_site"])
            target = model.site("task_" + hole["target_site"])
            assert entry.pos[2] == pytest.approx(0.025)
            assert target.pos[2] == pytest.approx(0.003)
            np.testing.assert_allclose(entry.pos[:2], target.pos[:2])
            np.testing.assert_allclose(entry.quat, target.quat)
            path = builder.resolve_asset_path(catalog["variants"][shape][str(size)]["xml"])
            if path not in variants:
                variants[path] = builder._component(path).compile()
            peg = variants[path]
            family_dimensions.append((peg.body_mass.copy(), peg.geom_size.copy(), peg.mesh_vert.copy()))
        # Size denotes clearance, so the physical peg must not be scaled by row.
        for dimensions in family_dimensions[1:]:
            for actual, expected in zip(dimensions, family_dimensions[0]):
                np.testing.assert_allclose(actual, expected)


def test_collision_modes_keep_initial_state_and_hole_frames(composed_scene):
    config, functional = composed_scene("magpie_peg_in_hole")
    assert config.board["collision"] == "functional"
    detailed_config = replace(config, board={**config.board, "collision": "detailed"})
    detailed = builder.build_magpie_scene(detailed_config)
    assert detailed.ngeom - functional.ngeom == 7890
    np.testing.assert_array_equal(functional.qpos0, detailed.qpos0)
    for hole in config.catalog["holes"].values():
        for field in ("entry_site", "target_site"):
            name = "task_" + hole[field]
            np.testing.assert_array_equal(functional.site(name).pos, detailed.site(name).pos)
            np.testing.assert_array_equal(functional.site(name).quat, detailed.site(name).quat)


def test_all_twenty_pegs_initialize_at_authored_site_poses(composed_scene):
    config, model = composed_scene("magpie_peg_in_hole")
    assert (model.nq, model.nv, model.nu, model.nsensor) == (186, 165, 31, 101)
    data = mujoco.MjData(model)
    np.testing.assert_array_equal(data.qpos, model.qpos0)
    np.testing.assert_array_equal(data.qvel, 0)
    for peg in config.board["pegs"]:
        assert peg["insertion_depth"] == config.catalog["holes"][peg["hole"]]["seated_depth"]
    mujoco.mj_forward(model, data)
    _assert_peg_poses(config, model, data)
    assert data.time == 0
    # Exact contact at the blind-hole floor is valid; deep initial overlaps are not.
    for contact in data.contact:
        bodies = [model.body(model.geom_bodyid[geom]).name for geom in contact.geom]
        if any(name.startswith("task_peg_") for name in bodies):
            assert contact.dist >= -1e-5, (bodies, contact.dist)
    assert np.isfinite(data.qacc).all()


@pytest.mark.parametrize("reset", ["mujoco", "environment"])
def test_all_twenty_pegs_reset_to_configured_defaults(composed_scene, reset):
    config, model = composed_scene("magpie_peg_in_hole")
    env = MujocoEnv(model=model)
    env.data.qpos[46:] = 0.3
    env.data.qvel[:] = 2
    env.data.ctrl[:] = 3
    env.data.time = 7
    if reset == "environment":
        env.reset()
    else:
        mujoco.mj_resetData(model, env.data)
        mujoco.mj_forward(model, env.data)
    np.testing.assert_array_equal(env.data.qpos, model.qpos0)
    np.testing.assert_array_equal(env.data.qvel, 0)
    np.testing.assert_array_equal(env.data.ctrl, 0)
    assert env.data.time == 0
    _assert_peg_poses(config, model, env.data)


def _rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def test_transformed_table_board_and_mixed_depths(tmp_path, peg_config):
    # Noncommuting rotations catch reversed pose multiplication and world-Z depth.
    table_angle, board_angle = 0.31, -0.47
    peg_config["table"]["position"] = [0.72, -0.18, 0.07]
    peg_config["table"]["size"] = [0.92, 0.61]
    peg_config["table"]["height"] = 1.12
    peg_config["table"]["top_thickness"] = 0.055
    peg_config["table"]["quaternion"] = [np.cos(table_angle / 2).item(), np.sin(table_angle / 2).item(), 0, 0]
    peg_config["board"]["position_on_table"] = [0.04, -0.03, 0.012]
    peg_config["board"]["quaternion"] = [np.cos(board_angle / 2).item(), 0, 0, np.sin(board_angle / 2).item()]
    peg_config["board"]["pegs"] = [deepcopy(peg_config["board"]["pegs"][i]) for i in (0, 5, 19)]
    for peg, depth in zip(peg_config["board"]["pegs"], (0, 0.011, 0.022)):
        peg["insertion_depth"] = depth
    config = builder.load_scene_config(_write_config(tmp_path, peg_config))
    model = builder.build_magpie_scene(config)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    table_rotation = _rotation(config.table["quaternion"])
    expected_position = np.asarray(config.table["position"]) + table_rotation @ (
        np.asarray(config.board["position_on_table"]) + [0, 0, config.table["height"]])
    expected_rotation = table_rotation @ _rotation(config.board["quaternion"])
    np.testing.assert_allclose(model.geom("task_table_top").size, [0.46, 0.305, 0.0275])
    np.testing.assert_allclose(data.body("task_board").xpos, expected_position, atol=1e-10)
    np.testing.assert_allclose(data.body("task_board").xmat.reshape(3, 3), expected_rotation, atol=1e-10)
    np.testing.assert_allclose(data.site("task_tabletop").xpos,
                               np.asarray(config.table["position"]) + table_rotation @ [0, 0, config.table["height"]], atol=1e-10)
    _assert_peg_poses(config, model, data)
