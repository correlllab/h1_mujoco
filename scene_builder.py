"""Compose the deployed Magpie model with project-local scene configurations.

Scene arguments are project-root-relative (bare names default to scene/).
Every asset reference is relative to the single CL_Assets root, never the YAML.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_ROOT = PROJECT_ROOT / "scene"
ASSET_ROOT = PROJECT_ROOT / "submodules" / "CL_Assets"
SHAPES = ("l", "rectangle", "triangle", "hex", "cylinder")


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    scene_xml: Path
    timestep: float
    table: dict
    board: dict
    catalog: dict | None


def resolve_scene_path(value: str | Path) -> Path:
    """Resolve one CLI candidate without ambiguous cwd-dependent searching."""
    text = str(value)
    if not text or text.endswith("/"):
        raise ValueError("--scene must specify a filename")
    path = Path(text).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    elif path.suffix != ".yaml":
        raise ValueError("--scene must use .yaml or omit the extension")
    if not path.is_absolute():
        root = PROJECT_ROOT if "/" in text or text.startswith(".") else SCENE_ROOT
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scene {text!r} resolved to missing file: {path}")
    return path


def resolve_asset_path(value: str, field: str = "asset") -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected a nonempty asset-root-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field}: use a CL_Assets-relative path without '..': {value}")
    resolved = (ASSET_ROOT / path).resolve()
    if not resolved.is_relative_to(ASSET_ROOT.resolve()):
        raise ValueError(f"{field}: asset escapes CL_Assets: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field}: asset not found: {resolved}")
    return resolved


def _mapping(value, field):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{field}: expected a mapping with string keys")
    return dict(value)


def _known(value, keys, field):
    unknown = set(value) - set(keys)
    if unknown:
        raise ValueError(f"{field}: unknown fields: {', '.join(sorted(unknown))}")


def _number(value, field, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a finite number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{field}: expected a finite {'positive ' if positive else ''}number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field}: must be at least {minimum}")
    return value


def _vector(value, length, field):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field}: expected {length} numbers")
    return [_number(v, f"{field}[{i}]") for i, v in enumerate(value)]


def _quaternion(value, field):
    q = _vector(value, 4, field)
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{field}: quaternion must have unit length (w, x, y, z)")
    return q


def _enabled(value, field):
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{field}.enabled: expected true or false")
    return enabled


def load_scene_config(value: str | Path) -> SceneConfig:
    path = resolve_scene_path(value)
    with path.open() as stream:
        cfg = _mapping(yaml.safe_load(stream), "scene configuration")
    _known(cfg, ("version", "scene", "simulation", "table", "board"), "configuration")
    if type(cfg.get("version")) is not int or cfg["version"] != 1:
        raise ValueError("version: expected scene schema version 1")
    scene = _mapping(cfg.get("scene", {}), "scene")
    _known(scene, ("xml",), "scene")
    scene_xml = resolve_asset_path(scene.get("xml", "mujoco_assets/manipulation_net/workcell.xml"), "scene.xml")
    simulation = _mapping(cfg.get("simulation", {}), "simulation")
    _known(simulation, ("timestep",), "simulation")
    timestep = _number(simulation.get("timestep", 0.005), "simulation.timestep", positive=True)

    table = _mapping(cfg.get("table", {"enabled": False}), "table")
    _known(table, ("enabled", "position", "quaternion", "size", "height", "top_thickness",
                   "leg_width", "leg_inset", "rgba", "friction"), "table")
    table["enabled"] = _enabled(table, "table")
    table["position"] = _vector(table.get("position", [0.65, 0, 0]), 3, "table.position")
    table["quaternion"] = _quaternion(table.get("quaternion", [1, 0, 0, 0]), "table.quaternion")
    table["size"] = _vector(table.get("size", [0.8, 0.7]), 2, "table.size")
    for key, default in (("height", 1.05), ("top_thickness", 0.04), ("leg_width", 0.05)):
        table[key] = _number(table.get(key, default), f"table.{key}", positive=True)
    table["leg_inset"] = _number(table.get("leg_inset", 0.03), "table.leg_inset", minimum=0)
    table["rgba"] = _vector(table.get("rgba", [0.55, 0.40, 0.28, 1]), 4, "table.rgba")
    table["friction"] = _vector(table.get("friction", [0.8, 0.005, 0.0001]), 3, "table.friction")
    if any(v < 0 or v > 1 for v in table["rgba"]) or any(v < 0 for v in table["friction"]):
        raise ValueError("table: rgba must be in [0, 1] and friction must be nonnegative")
    if table["top_thickness"] >= table["height"]:
        raise ValueError("table.top_thickness must be smaller than table.height")
    if min(table["size"]) <= table["leg_width"] + 2 * table["leg_inset"]:
        raise ValueError("table.size must leave room for the legs and their inset")

    board = _mapping(cfg.get("board", {"enabled": False}), "board")
    _known(board, ("enabled", "catalog", "collision", "position_on_table", "quaternion", "pegs"), "board")
    board["enabled"] = _enabled(board, "board")
    board["collision"] = board.get("collision", "functional")
    if board["collision"] not in ("functional", "detailed"):
        raise ValueError("board.collision: expected 'functional' or 'detailed'")
    board["position_on_table"] = _vector(board.get("position_on_table", [0, 0, 0]), 3, "board.position_on_table")
    board["quaternion"] = _quaternion(board.get("quaternion", [1, 0, 0, 0]), "board.quaternion")
    board["pegs"] = board.get("pegs", [])
    if not isinstance(board["pegs"], list):
        raise ValueError("board.pegs: expected a list")
    if board["enabled"] and not table["enabled"]:
        raise ValueError("board: a tabletop-relative board requires an enabled table")
    if not board["enabled"] and board["pegs"]:
        raise ValueError("board.pegs: cannot place pegs on a disabled board")

    catalog = None
    if board["enabled"]:
        catalog_path = resolve_asset_path(board.get("catalog", "mujoco_assets/manipulation_net/manipulation_catalog.json"), "board.catalog")
        with catalog_path.open() as stream:
            catalog = _mapping(json.load(stream), "board catalog")
        _validate_catalog(catalog)
        ids, occupied = set(), set()
        placements = []
        for i, raw in enumerate(board["pegs"]):
            field = f"board.pegs[{i}]"
            peg = _mapping(raw, field)
            _known(peg, ("id", "shape", "size", "hole", "insertion_depth"), field)
            name = peg.get("id")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"{field}.id: use a letter followed by letters, digits or underscores")
            if name in ids:
                raise ValueError(f"{field}.id: duplicate peg id {name}")
            ids.add(name)
            shape, size = peg.get("shape"), peg.get("size")
            if shape not in SHAPES or type(size) is not int or size not in range(4):
                raise ValueError(f"{field}: expected shape in {SHAPES} and integer size 0..3")
            hole_id = peg.get("hole")
            if not isinstance(hole_id, str) or hole_id not in catalog["holes"]:
                raise ValueError(f"{field}.hole: unknown hole {hole_id!r}")
            hole = catalog["holes"][hole_id]
            if hole_id in occupied:
                raise ValueError(f"{field}.hole: hole {hole_id} is already occupied")
            occupied.add(hole_id)
            if (hole["shape"], hole["size"]) != (shape, size):
                raise ValueError(f"{field}: peg {shape}/{size} is incompatible with hole {hole_id}")
            depth = _number(peg.get("insertion_depth", hole["seated_depth"]), f"{field}.insertion_depth", minimum=0)
            if depth > hole["max_insertion_depth"] + 1e-10:
                raise ValueError(f"{field}.insertion_depth exceeds hole depth {hole['max_insertion_depth']}")
            peg["insertion_depth"] = depth
            placements.append(peg)
        board["pegs"] = placements
    return SceneConfig(path, scene_xml, timestep, table, board, catalog)


def _validate_catalog(catalog):
    """Fail on incomplete physical definitions before starting any interfaces."""
    if type(catalog.get("version")) is not int or catalog["version"] != 1:
        raise ValueError("board catalog: expected version 1")
    board = _mapping(catalog.get("board"), "catalog.board")
    resolve_asset_path(board.get("xml"), "catalog.board.xml")
    collision_xmls = _mapping(board.get("collision_xmls"), "catalog.board.collision_xmls")
    for mode in ("functional", "detailed"):
        resolve_asset_path(collision_xmls.get(mode), f"catalog.board.collision_xmls.{mode}")
    footprint = _vector(board.get("footprint"), 2, "catalog.board.footprint")
    if not np.allclose(footprint, [0.32, 0.25], atol=1e-6, rtol=0):
        raise ValueError("catalog.board.footprint: expected calibrated 0.32 x 0.25 m board")
    holes = _mapping(catalog.get("holes"), "catalog.holes")
    variants = _mapping(catalog.get("variants"), "catalog.variants")
    expected = {f"{shape}_{size}" for shape in SHAPES for size in range(4)}
    if set(holes) != expected:
        raise ValueError("catalog.holes: expected exactly the 20 shape/size hole definitions")
    for shape in SHAPES:
        family = _mapping(variants.get(shape), f"catalog.variants.{shape}")
        for size in range(4):
            key = f"{shape}_{size}"
            hole = _mapping(holes.get(key), f"catalog.holes.{key}")
            if hole.get("shape") != shape or type(hole.get("size")) is not int or hole["size"] != size:
                raise ValueError(f"catalog.holes.{key}: inconsistent shape/size")
            for site in ("entry_site", "target_site"):
                if not isinstance(hole.get(site), str) or not hole[site]:
                    raise ValueError(f"catalog.holes.{key}.{site}: missing site name")
            maximum = _number(hole.get("max_insertion_depth"), f"catalog.holes.{key}.max_insertion_depth", positive=True)
            seated = _number(hole.get("seated_depth"), f"catalog.holes.{key}.seated_depth", minimum=0)
            if seated > maximum:
                raise ValueError(f"catalog.holes.{key}: seated depth exceeds maximum")
            variant = _mapping(family.get(str(size)), f"catalog.variants.{shape}.{size}")
            resolve_asset_path(variant.get("xml"), f"catalog.variants.{shape}.{size}.xml")


def _component(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.tag in ("include", "attach", "replicate", "model"):
            raise ValueError(f"{path}: components must be standalone; compose them with the scene builder")
        if element.tag in ("body", "site") and any(
            key in element.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")
        ):
            raise ValueError(f"{path}: component body/site orientations must use explicit quaternions")
        if element.tag == "compiler":
            element.set("meshdir", "")
            element.set("texturedir", "")
            element.set("strippath", "false")
        # Resolve before parsing, including texture faces and other file assets.
        # MuJoCo 3.6 otherwise resolves attached assets against the parent origin.
        for key, value in list(element.attrib.items()):
            if key.startswith("file") and value:
                element.set(key, str(resolve_asset_path(value, f"{path.name}: {element.tag}.{key}")))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    if spec.actuators or spec.sensors:
        raise ValueError(f"{path}: workcell components may not add actuators or sensors")
    return spec


def _compose(a, b):
    pos = np.empty(3)
    quat = np.empty(4)
    mujoco.mju_mulPose(pos, quat, a[0], a[1], b[0], b[1])
    return pos, quat


def _inverse(pose):
    pos = np.empty(3)
    quat = np.empty(4)
    mujoco.mju_negPose(pos, quat, pose[0], pose[1])
    return pos, quat


def _site_pose(spec, site_name, body_name, *, relative_to_body=False):
    site, body = spec.site(site_name), spec.body(body_name)
    if (site is None or body is None or site.parent != body or site.frame is not None
            or body.parent != spec.worldbody or body.frame is not None):
        raise ValueError(f"{site_name}: expected a site directly on component body {body_name}")
    if relative_to_body:
        return np.array(site.pos), np.array(site.quat)
    return _compose((body.pos, body.quat), (site.pos, site.quat))


def build_magpie_scene(config: SceneConfig, *, fixed=False) -> mujoco.MjModel:
    filename = "scene_h1_2_magpie_pelvis_fixed.xml" if fixed else "scene_h1_2_magpie.xml"
    parent = mujoco.MjSpec.from_file(str(resolve_asset_path(f"mujoco_assets/{filename}", "robot scene")))
    parent.compiler.meshdir = str(ASSET_ROOT / "meshes")
    for mesh in parent.meshes:
        if mesh.file:
            mesh.file = str(resolve_asset_path(f"meshes/{mesh.file}", f"robot mesh {mesh.name}"))
    parent.compiler.meshdir = ""
    workcell = _component(config.scene_xml)
    if workcell.joints:
        raise ValueError("scene.xml: the workcell must be static")

    table = config.table
    if table["enabled"]:
        body = workcell.worldbody.add_body(name="table", pos=table["position"], quat=table["quaternion"])
        width, length = table["size"]
        height, thickness, leg = table["height"], table["top_thickness"], table["leg_width"]
        style = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=table["rgba"], friction=table["friction"])
        body.add_geom(name="table_top", pos=[0, 0, height - thickness / 2], size=[width / 2, length / 2, thickness / 2], **style)
        for ix, x in enumerate((-1, 1)):
            for iy, y in enumerate((-1, 1)):
                body.add_geom(name=f"table_leg_{ix}_{iy}",
                              pos=[x * (width / 2 - table["leg_inset"] - leg / 2),
                                   y * (length / 2 - table["leg_inset"] - leg / 2),
                                   (height - thickness) / 2],
                              size=[leg / 2, leg / 2, (height - thickness) / 2], **style)
        body.add_site(name="tabletop", pos=[0, 0, height], size=[0.003] * 3, group=4)

    placements = []
    if config.board["enabled"]:
        board_config, catalog = config.board, config.catalog
        board = _component(resolve_asset_path(catalog["board"]["collision_xmls"][board_config["collision"]]))
        if board.joints:
            raise ValueError("board component must be static")
        # Board/hole poses are read before attachment, which transfers their ownership.
        hole_poses = {}
        for key, hole in catalog["holes"].items():
            hole_poses[key] = _site_pose(board, hole["entry_site"], "board")
            _site_pose(board, hole["target_site"], "board")
        table_pose = (np.asarray(table["position"]), np.asarray(table["quaternion"]))
        top_pose = _compose(table_pose, ([0, 0, table["height"]], [1, 0, 0, 0]))
        board_pose = _compose(top_pose, (board_config["position_on_table"], board_config["quaternion"]))
        mount = body.add_frame(name="board_mount",
                               pos=np.asarray(board_config["position_on_table"]) + [0, 0, table["height"]],
                               quat=board_config["quaternion"])
        workcell.attach(board, frame=mount, prefix="")
        for peg_config in board_config["pegs"]:
            shape, size = peg_config["shape"], str(peg_config["size"])
            peg = _component(resolve_asset_path(catalog["variants"][shape][size]["xml"]))
            if len(peg.joints) != 1 or peg.joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"peg {peg_config['id']}: expected exactly one free joint")
            if peg.joints[0].parent != peg.body("peg"):
                raise ValueError(f"peg {peg_config['id']}: the free joint must belong to body 'peg'")
            tip = _site_pose(peg, "tip", "peg", relative_to_body=True)
            hole_world = _compose(board_pose, hole_poses[peg_config["hole"]])
            tip_world = _compose(hole_world, ([0, 0, -peg_config["insertion_depth"]], [1, 0, 0, 0]))
            pos, quat = _compose(tip_world, _inverse(tip))
            peg_body = peg.body("peg")
            peg_body.pos, peg_body.quat = pos, quat
            placements.append((peg, f"task_peg_{peg_config['id']}_"))

    frame = parent.worldbody.add_frame(name="task_mount")
    parent.attach(workcell, frame=frame, prefix="task_")
    for peg, prefix in placements:
        parent.attach(peg, frame=frame, prefix=prefix)
    parent.option.timestep = config.timestep
    return parent.compile()
