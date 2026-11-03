"""Source geometry and headless contact probes, independent of robot scenes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from shapely import MultiPoint, Polygon, union_all
from shapely.affinity import rotate, scale, translate


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "submodules" / "CL_Assets"
CATALOG = json.loads((ASSETS / "mujoco_assets/manipulation_net/manipulation_catalog.json").read_text())
SHAPES = ("l", "rectangle", "triangle", "hex", "cylinder")


def component(path):
    spec = mujoco.MjSpec.from_file(str(ASSETS / path))
    for mesh in spec.meshes:
        if mesh.file:
            assert not mesh.file.startswith("../")
            mesh.file = str(ASSETS / mesh.file)
    return spec


@pytest.fixture(scope="module", params=("functional", "detailed"))
def board_spec(request):
    return component(CATALOG["board"]["collision_xmls"][request.param])


@pytest.fixture(scope="module")
def models(board_spec):
    result = {}
    for shape in SHAPES:
        board = board_spec.copy()
        peg = component(CATALOG["variants"][shape]["0"]["xml"])
        board.attach(peg, frame=board.worldbody.add_frame(), prefix="probe_")
        board.option.gravity = [0, 0, 0]
        board.option.timestep = 0.001
        result[shape] = board.compile()
    return result


@pytest.fixture(scope="module")
def guided_models(board_spec):
    """Test jig, not a shipped peg variant: vertical motion is force-driven."""
    result = {}
    for shape in SHAPES:
        board = board_spec.copy()
        variant = CATALOG["variants"][shape]["3"]
        hole = CATALOG["holes"][f"{shape}_3"]
        peg = component(variant["xml"])
        peg.delete(peg.joint("freejoint"))
        body = peg.body("peg")
        body.pos = [*hole["entry"][:2], 0]
        body.quat = hole["quat"]
        body.add_joint(name="vertical_guide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                       axis=[0, 0, 1], damping=100 * variant["mass"])
        board.attach(peg, frame=board.worldbody.add_frame(), prefix="guided_")
        board.option.gravity = [0, 0, -9.81]
        result[shape] = board.compile()
    return result


def place(model, data, hole, z, offset=(0, 0)):
    data.qpos[:3] = [hole["entry"][0] + offset[0], hole["entry"][1] + offset[1], z]
    data.qpos[3:7] = hole["quat"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def penetrations(data):
    return [c.dist for c in data.contact if c.dist < -1e-7]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("size", range(4))
def test_aligned_peg_insertion_and_withdrawal(models, shape, size):
    """Prescribed probes traverse the whole blind bore in both directions."""
    model = models[shape]
    data = mujoco.MjData(model)
    hole = CATALOG["holes"][f"{shape}_{size}"]
    path = np.linspace(0.026, 0.0031, 25)
    for z in np.r_[path, path[::-1]]:
        place(model, data, hole, z)
        assert not penetrations(data), (shape, size, z, penetrations(data))
    # The actual source has a floor, not a through-hole.
    place(model, data, hole, 0.0028)
    assert min(penetrations(data)) < -0.0001


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("size", range(4))
def test_offset_peg_contacts_during_insertion_and_withdrawal(models, shape, size):
    model = models[shape]
    data = mujoco.MjData(model)
    hole = CATALOG["holes"][f"{shape}_{size}"]
    for path in (np.linspace(0.024, 0.004, 7), np.linspace(0.004, 0.024, 7)):
        for z in path:
            place(model, data, hole, z, offset=(0.004, 0))
            assert penetrations(data), (shape, size, z)
    place(model, data, hole, 0.027, offset=(0.004, 0))
    assert not penetrations(data)


@pytest.mark.parametrize("shape", SHAPES)
def test_dynamic_contact_free_entry_and_exit(models, shape):
    """Free pegs move into and back out of tightest bores without pose forcing."""
    model = models[shape]
    data = mujoco.MjData(model)
    hole = CATALOG["holes"][f"{shape}_0"]
    place(model, data, hole, 0.03)
    data.qvel[2] = -0.02
    for _ in range(1000):
        mujoco.mj_step(model, data)
        assert not penetrations(data)
    assert data.qpos[2] == pytest.approx(0.01, abs=1e-8)
    data.qvel[2] = 0.02
    for _ in range(1000):
        mujoco.mj_step(model, data)
        assert not penetrations(data)
    assert data.qpos[2] == pytest.approx(0.03, abs=1e-8)
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("timestep", (0.001, 0.005))
def test_force_driven_extraction_and_gravity_reinsertion(guided_models, shape, timestep, record_property):
    """A vertical jig guides extraction and gravity insertion without pose forcing.

    These blunt source pegs are not self-aligning; an entirely unconstrained
    falling peg can tip on contact. The test jig holds XY and orientation, with
    viscous damping for ~0.1 m/s terminal speed. Shipped pegs retain freejoints.
    """
    model = guided_models[shape]
    original_timestep = model.opt.timestep
    model.opt.timestep = timestep
    try:
        data = mujoco.MjData(model)
        data.qpos[0] = 0.0031
        mujoco.mj_forward(model, data)
        for _ in range(round(0.5 / timestep)):
            mujoco.mj_step(model, data)
        initial_floor_penetration = max(0, 0.003 - data.site("guided_tip").xpos[2])
        assert initial_floor_penetration < 0.000005
        weight = model.body("guided_peg").mass[0] * 9.81
        data.qfrc_applied[0] = 2 * weight
        for _ in range(round(0.6 / timestep)):
            mujoco.mj_step(model, data)
            if data.site("guided_tip").xpos[2] > 0.027:
                break
        else:
            pytest.fail("Upward force did not extract the peg")
        data.qfrc_applied[:] = 0
        deepest_contact = 0
        for _ in range(round(1.2 / timestep)):
            mujoco.mj_step(model, data)
            deepest_contact = min([deepest_contact, *(c.dist for c in data.contact)])
        final_floor_penetration = max(0, 0.003 - data.site("guided_tip").xpos[2])
        record_property("settled_floor_penetration_m", final_floor_penetration)
        record_property("peak_reinsertion_contact_penetration_m", -deepest_contact)
        assert final_floor_penetration < 0.000005
        assert abs(data.site("guided_tip").xpos[2] - 0.003) < 0.000005
        assert deepest_contact > -0.00025
        assert np.linalg.norm(data.qvel) < 0.001
        assert not any(w.number for w in data.warning)
        assert np.isfinite(data.qpos).all()
    finally:
        model.opt.timestep = original_timestep


def test_l_compound_preserves_concave_notch():
    variant = CATALOG["variants"]["l"]["0"]
    profile = Polygon(variant["profile_xy"])
    assert profile.convex_hull.area > profile.area * 1.2
    spec = component(variant["xml"])
    # Select a sphere wholly within the notch, but within the invalid convex hull.
    missing = profile.convex_hull.difference(profile)
    point = missing.representative_point()
    radius = min(point.distance(profile), point.distance(profile.convex_hull.boundary)) / 3
    assert radius > 0.0001
    spec.worldbody.add_geom(name="notch_probe", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            pos=[point.x, point.y, 0.025], size=[radius, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_catalog_and_component_contracts(models, board_spec):
    assert CATALOG["version"] == 1
    assert CATALOG["board"]["footprint"] == [0.32, 0.25]
    mode = "functional" if board_spec.modelname.endswith("_functional") else "detailed"
    collision = CATALOG["board"]["collision_modes"][mode]
    assert len(CATALOG["holes"]) == 20
    for shape in SHAPES:
        variants = CATALOG["variants"][shape]
        assert len({v["xml"] for v in variants.values()}) == 1
        source = variants["0"]["source"]
        assert hashlib.sha256((ASSETS / source["file"]).read_bytes()).hexdigest() == source["sha256"]
        assert source["nonmanifold_edge_count_at_10nm_weld"] == 0
        model = models[shape]
        assert np.count_nonzero(model.geom_bodyid == model.body("board").id) == collision["collision_geom_count"] + 1
        assert model.nq == 7 and model.nv == 6 and model.nu == 0
        assert model.body("probe_peg").mass[0] == pytest.approx(variants["0"]["mass"])
        assert np.all(model.body("probe_peg").inertia > 0)
        assert np.allclose(model.site("probe_tip").pos, 0)
        for size in range(4):
            hole = CATALOG["holes"][f"{shape}_{size}"]
            assert hole["entry"][2] == 0.025 and hole["target"][2] == 0.003
            assert hole["max_insertion_depth"] == pytest.approx(0.022)
            assert hole["minimum_wall_clearance_at_site_pose"] > 1.4e-5
            for key, expected in (("entry_site", "entry"), ("target_site", "target")):
                site = model.site(hole[key])
                assert np.allclose(site.pos, hole[expected], atol=1e-12)
                assert np.allclose(site.quat, hole["quat"], atol=1e-12)
            peg_profile = Polygon(variants[str(size)]["profile_xy"])
            angle = 2 * np.arctan2(hole["quat"][3], hole["quat"][0])
            placed = translate(rotate(peg_profile, angle, origin=(0, 0), use_radians=True), *hole["entry"][:2])
            assert Polygon(hole["profile_xy"]).covers(placed)
    board_source = CATALOG["board"]["source"]
    assert board_source["nonmanifold_edge_count_at_10nm_weld"] == 0
    assert hashlib.sha256((ASSETS / board_source["file"]).read_bytes()).hexdigest() == board_source["sha256"]


@pytest.fixture(scope="module")
def preparer():
    spec = importlib.util.spec_from_file_location("prepare_manipulation_assets", ROOT / "utility/prepare_manipulation_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collision_sections(root):
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall("asset/mesh")}
    sections = {}
    for name in ("middle", "top"):
        parts = []
        for geom in root.findall("worldbody/body/geom"):
            if not geom.attrib["name"].startswith(f"board_{name}_"):
                continue
            vertices = np.fromstring(meshes[geom.attrib["mesh"]].attrib["vertex"], sep=" ").reshape(-1, 3)
            center = np.fromstring(geom.attrib["pos"], sep=" ")
            parts.append(MultiPoint((vertices + center)[:, :2]).convex_hull)
        # Serialization/recentering changes shared seams at ~1e-14 m. Snap those
        # seams before GEOS overlay; otherwise nearly coincident edges can make
        # its floating-precision union discard large regions spuriously.
        sections[name] = union_all(parts, grid_size=1e-12)
    return sections


def test_board_collision_sections_match_source(preparer):
    root = ET.parse(ASSETS / CATALOG["board"]["xml"]).getroot()
    source = preparer.read_stl(ASSETS / CATALOG["board"]["source"]["file"])
    sections = collision_sections(root)
    top_source = scale(preparer.surface_polygons(source, 2.5), xfact=0.01, yfact=0.01, origin=(0, 0))
    assert sections["top"].symmetric_difference(top_source).area < 2e-10
    holes = union_all([Polygon(h["profile_xy"]) for h in CATALOG["holes"].values()])
    expected_middle = Polygon([(-0.16, -0.125), (0.16, -0.125), (0.16, 0.125), (-0.16, 0.125)]).difference(holes)
    assert sections["middle"].symmetric_difference(expected_middle).area < 2e-10


def test_functional_board_fills_only_cosmetic_engravings(preparer):
    board = CATALOG["board"]
    assert board["collision_xmls"]["detailed"] == board["xml"]
    roots = {mode: ET.parse(ASSETS / path).getroot() for mode, path in board["collision_xmls"].items()}
    detailed = collision_sections(roots["detailed"])
    functional = collision_sections(roots["functional"])
    source = preparer.read_stl(ASSETS / board["source"]["file"])
    floors = scale(preparer.surface_polygons(source, 0.3), xfact=0.01, yfact=0.01, origin=(0, 0))
    footprint = Polygon([(-0.16, -0.125), (0.16, -0.125), (0.16, 0.125), (-0.16, 0.125)])
    expected = footprint.difference(floors)
    assert functional["top"].is_empty
    assert functional["middle"].symmetric_difference(expected).area < 2e-10
    assert functional["middle"].symmetric_difference(detailed["middle"]).area < 2e-10
    # The only extra solid is the engraving void, never a functional opening.
    cosmetic_fill = expected.difference(detailed["top"])
    assert cosmetic_fill.area > 1e-6
    assert detailed["top"].difference(expected).area < 2e-10
    assert functional["middle"].difference(detailed["top"]).symmetric_difference(cosmetic_fill).area < 2e-10
    assert functional["middle"].intersection(floors).area < 2e-10
    assert functional["middle"].bounds == pytest.approx(footprint.bounds, abs=1e-12)

    for mode, root in roots.items():
        metadata = board["collision_modes"][mode]
        geoms = root.findall("worldbody/body/geom")
        prisms = [geom for geom in geoms if geom.get("type") == "mesh" and geom.get("name") != "visual"]
        assert len(prisms) == sum(metadata["layer_prism_counts"])
        assert len(geoms) - 1 == metadata["collision_geom_count"]
        assert metadata["engraved_recesses_preserved"] == (mode == "detailed")
        assert metadata["layer_prism_counts"] == ([3004] if mode == "functional" else [3004, 7890])
    functional_root = roots["functional"]
    meshes = {mesh.get("name"): mesh for mesh in functional_root.findall("asset/mesh")}
    for geom in functional_root.findall("worldbody/body/geom"):
        if not geom.get("name").startswith("board_middle_"):
            continue
        vertices = np.fromstring(meshes[geom.get("mesh")].get("vertex"), sep=" ").reshape(-1, 3)
        vertices += np.fromstring(geom.get("pos"), sep=" ")
        assert vertices[:, 2].min() == pytest.approx(0.003, abs=1e-12)
        assert vertices[:, 2].max() == pytest.approx(0.025, abs=1e-12)
    for selector in ("asset/mesh[@name='board_visual_mesh']", "worldbody/body/geom[@name='visual']",
                     "worldbody/body/geom[@name='bottom']"):
        assert roots["functional"].find(selector).attrib == roots["detailed"].find(selector).attrib
    assert [site.attrib for site in functional_root.findall("worldbody/body/site")] == [
        site.attrib for site in roots["detailed"].findall("worldbody/body/site")]


def test_preparation_is_reproducible_and_check_is_read_only(preparer, monkeypatch):
    paths = list((ASSETS / "mujoco_assets/manipulation_net").glob("manipulation_*.xml"))
    paths.append(ASSETS / "mujoco_assets/manipulation_net/manipulation_catalog.json")
    paths.extend((ASSETS / "meshes/manipulation_net/collision").glob("*.obj"))
    paths.extend((ASSETS / "meshes/manipulation_net").glob("*.stl"))
    before = [(p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).digest()) for p in paths]
    checked = set()
    write_generated = preparer.write_generated

    def track_check(path, text, check=False):
        assert check
        checked.add(path)
        write_generated(path, text, check)

    monkeypatch.setattr(preparer, "write_generated", track_check)
    preparer.prepare(check=True)
    assert {ASSETS / path for path in CATALOG["board"]["collision_xmls"].values()} <= checked
    assert before == [(p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).digest()) for p in paths]


def test_stale_asset_check_does_not_overwrite(preparer, tmp_path):
    path = tmp_path / "derived.xml"
    path.write_text("stale")
    with pytest.raises(ValueError, match="stale generated asset"):
        preparer.write_generated(path, "current", check=True)
    assert path.read_text() == "stale"


def test_preparation_rejects_tapered_or_relayered_source(preparer):
    source = preparer.read_stl(ASSETS / CATALOG["variants"]["rectangle"]["0"]["source"]["file"])
    tapered = source.copy()
    tapered[:, :, 0] += np.where(tapered[:, :, 2] > 0, 0.1, 0)
    with pytest.raises(ValueError, match="nonvertical/tapered"):
        preparer.validate_extruded_source(tapered, [0, 5], "tapered")
    with pytest.raises(ValueError, match="unsupported height levels"):
        preparer.validate_extruded_source(source, [0, 4], "relayered")
