"""Procedural scene assembly: merge the H1-2 robot into a Robocasa kitchen.

Used by h12_mujoco.py when invoked with `--scene kitchen`. Imports robosuite
and robocasa lazily so that running without `--scene kitchen` does not require
those packages to be installed.
"""
import os
import types
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.utils import env_utils as EnvUtils
from robosuite.models.base import MujocoXML
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T


class _NestedDefaultsMujocoXML(MujocoXML):
    """MujocoXML that handles nested `<default class="…">` blocks and
    preserves the block for childclass propagation.

    Upstream's __init__ does two things our MJCFs can't tolerate:
    (1) `_get_default_classes` only walks direct children of `<default>`, so
        nested classes (h1_2 → hip) raise KeyError when an element references
        the inner class; AND
    (2) it removes the entire `<default>` block after substitution. Our XML
        uses `childclass="h1_2"` propagation which mujoco resolves at compile
        time from the `<default>` block — without it, mujoco rejects the model
        with "unknown default childclass".

    This subclass flattens the nested class dict, skips the `<default>`
    subtree during the class-popping walk (so inner `<default class="X">`
    elements keep their class attrs), and leaves the block intact. Callers
    must copy `self.preserved_default` onto the merge target before save.
    """

    def __init__(self, fname):
        # Replicate MujocoXML.__init__ minus the self.root.remove(default) step.
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")

        default = self.create_default_element("default")
        self._replace_defaults_inline(default_dic=self._get_default_classes(default))
        self.preserved_default = default  # caller grafts onto merged tree

        self.resolve_asset_dependency()

    @staticmethod
    def _get_default_classes(default):
        flat: dict = {}

        def walk(node, inherited: dict) -> None:
            own = {child.tag: child for child in node if child.tag != "default"}
            merged = {**inherited, **own}
            cls = node.get("class")
            if cls is not None:
                flat[cls] = merged
            for child in node:
                if child.tag == "default":
                    walk(child, merged)

        for cls_node in default:
            if cls_node.tag == "default":
                walk(cls_node, {})
        return flat

    def _replace_defaults_inline(self, default_dic, root=None):
        if root is None:
            root = self.root
        # Don't recurse into the <default> tree — its <default class="X">
        # children must keep their class attrs so mujoco can match them at
        # compile time (used by childclass propagation).
        if root.tag == "default":
            return
        cls_name = root.attrib.pop("class", None)
        if cls_name is not None and cls_name in default_dic:
            tag_attrs = default_dic[cls_name].get(root.tag)
            if tag_attrs is not None:
                for k, v in tag_attrs.items():
                    if root.get(k, None) is None:
                        root.set(k, v)
        for child in root:
            self._replace_defaults_inline(default_dic=default_dic, root=child)

# Per-(layout_id, style_id) pelvis spawn pose (xyz, meters). Hand-tuned to
# place the robot in a clear aisle of the chosen kitchen layout. Add new
# entries when you bake additional combos.
SPAWN_POSES: dict[tuple[int, int], tuple[float, float, float]] = {
    # Layout 1 / Style 1: counters line the back wall (y ≈ -0.2 to -0.4); the
    # walkable aisle sits at y < -0.6. Center of the room is (2.75, -1.5).
    # Spawn ~0.65 m back from the counter face on the kitchen centerline.
    (1, 1): (2.75, -1.0, 1.05),
}


def build_kitchen_scene(
    layout_id: int = 1,
    style_id: int = 1,
    robot_xml: str = "/home/code/CL_Assets/mujoco_assets/h1_2_magpie_eflesh.xml",
    out_path: str = "/tmp/kitchen_scene.xml",
    fixture_seed: int = 0,
) -> str:
    """Construct a Robocasa kitchen with the H1-2 robot merged in.

    Returns the path to the assembled MJCF on disk, ready to hand to
    `mujoco.MjModel.from_xml_path()`.
    """
    arena = KitchenArena(layout_id=layout_id, style_id=style_id)
    arena.set_origin([0.0, 0.0, 0.0])

    # KitchenArena builds fixture *models* with default-origin positions for
    # decorative accessories (Toaster, KnifeBlock, PaperTowel, CoffeeMachine);
    # robocasa's full env flow normally runs a placement sampler that lands
    # them on countertops. Replicate that by feeding a minimal env shim —
    # just the attributes the sampler reaches for, plus Kitchen.get_fixture
    # bound onto it — to robocasa's get_single_fixture_sampler.
    fixture_cfgs = arena.get_fixture_cfgs()
    shim = types.SimpleNamespace(
        fixtures=arena.fixtures,
        objects={},
        fixture_cfgs=fixture_cfgs,
        rng=np.random.RandomState(fixture_seed),
    )
    shim.get_fixture = types.MethodType(Kitchen.get_fixture, shim)
    placements: dict = {}
    for cfg in fixture_cfgs:
        pos = getattr(cfg["model"], "pos", None)
        if pos is None or not cfg.get("placement"):
            continue
        if not (np.asarray(pos) == 0).all():
            continue
        sampler = EnvUtils.get_single_fixture_sampler(shim, cfg)
        placements.update(sampler.sample(placed_objects=placements))
    for obj_pos, obj_quat, obj in placements.values():
        obj.set_pos(obj_pos)
        obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))

    fixture_models = [cfg["model"] for cfg in fixture_cfgs]

    robot = _NestedDefaultsMujocoXML(robot_xml)

    # robosuite.MujocoXML.resolve_asset_dependency joins file refs with the
    # XML's directory but ignores <compiler meshdir>, so meshes end up at
    # /assets/pelvis.STL instead of /assets/meshes/pelvis.STL. Re-anchor every
    # mesh file under robot_dir/meshdir before merging into the arena.
    compiler = robot.root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if meshdir:
        abs_meshdir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(robot_xml)), meshdir)
        )
        # robosuite.resolve_asset_dependency has already prepended robot.folder
        # (the XML's directory) to every asset file ref, ignoring <compiler
        # meshdir>. Peel that prefix back off so the resulting relative path
        # (e.g. "h1_2/pelvis.STL", "magpie/mount.stl") preserves any subdir
        # structure, then rejoin against the real meshdir.
        for node in robot.asset.findall("./mesh[@file]"):
            cur = node.get("file")
            if not cur:
                continue
            rel = os.path.relpath(cur, robot.folder) if os.path.isabs(cur) else cur
            node.set("file", os.path.normpath(os.path.join(abs_meshdir, rel)))

    # robosuite leaves <include file="..."> elements untouched in the tree, so
    # task.save_model('/tmp/...') emits relative paths that mujoco then tries
    # to resolve against /tmp/. Anchor each include to the robot's source dir.
    robot_dir = os.path.dirname(os.path.abspath(robot_xml))
    for node in robot.root.findall(".//include[@file]"):
        inc = node.get("file")
        if inc and not os.path.isabs(inc):
            node.set("file", os.path.normpath(os.path.join(robot_dir, inc)))

    pelvis = robot.worldbody.find(".//body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError(
            f"body 'pelvis' not found in {robot_xml}; cannot set spawn pose"
        )
    try:
        spawn = SPAWN_POSES[(layout_id, style_id)]
    except KeyError:
        raise RuntimeError(
            f"no spawn pose recorded for (layout={layout_id}, style={style_id}); "
            f"add an entry to scene_builder.SPAWN_POSES after picking a clear "
            f"spot in the viewer."
        )
    pelvis.set("pos", f"{spawn[0]} {spawn[1]} {spawn[2]}")

    # ManipulationTask wraps the arena and appends each fixture's get_obj()
    # into the task's worldbody — that's the step KitchenArena alone skips,
    # without which the saved XML would be an empty room.
    task = ManipulationTask(
        mujoco_arena=arena,
        mujoco_robots=[],
        mujoco_objects=fixture_models,
    )
    task.merge(robot)

    # MujocoXML.merge() copies worldbody/asset/actuator/etc. but not <default>,
    # so the robot's nested-default classes (used for childclass="h1_2"
    # propagation) would be lost on save. Graft them onto the task tree.
    task_default = task.root.find("default")
    if task_default is None:
        task.root.append(robot.preserved_default)
    else:
        for child in list(robot.preserved_default):
            task_default.append(child)

    # The kitchen arena's compiler hard-codes inertiagrouprange="0 0", which
    # excludes the magpie gripper's group-3 collision geoms from inertia
    # computation and leaves the gripper bodies massless ("mass and inertia
    # of moving bodies must be larger than mjMINVAL"). Drop the attribute to
    # restore mujoco's default range (0..5); kitchen group-1 visuals are
    # contype=0/conaffinity=0 and get discardvisual'd anyway, so widening the
    # range doesn't affect kitchen physics.
    task_compiler = task.root.find("compiler")
    if task_compiler is not None and "inertiagrouprange" in task_compiler.attrib:
        del task_compiler.attrib["inertiagrouprange"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    task.save_model(out_path, pretty=True)
    return out_path
