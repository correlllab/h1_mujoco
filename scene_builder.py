"""Procedural scene assembly: merge the H1-2 robot into a Robocasa kitchen.

Used by h12_mujoco.py when invoked with `--scene kitchen`. Imports robosuite
and robocasa lazily so that running without `--scene kitchen` does not require
those packages to be installed.
"""
import os
import types

import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.utils import env_utils as EnvUtils
from robosuite.models.base import MujocoXML
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T

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
    robot_xml: str = "/home/code/assets/mujoco_assets/h1_2_handless.xml",
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

    robot = MujocoXML(robot_xml)

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
        for node in robot.asset.findall("./mesh[@file]"):
            node.set("file", os.path.join(abs_meshdir, os.path.basename(node.get("file"))))

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    task.save_model(out_path, pretty=True)
    return out_path
