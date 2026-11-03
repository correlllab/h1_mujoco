"""File-only relocation checks: no MuJoCo, model compilation, or DDS imports."""

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "submodules" / "CL_Assets"
COMPONENTS = ASSETS / "mujoco_assets" / "manipulation_net"
XML_FILES = (
    "workcell.xml",
    "manipulation_board.xml",
    "manipulation_board_functional.xml",
    "manipulation_peg_cylinder.xml",
    "manipulation_peg_hex.xml",
    "manipulation_peg_l.xml",
    "manipulation_peg_rectangle.xml",
    "manipulation_peg_triangle.xml",
)


def test_manipulation_assets_have_one_canonical_location():
    expected = {*XML_FILES, "manipulation_catalog.json"}
    assert {path.name for path in COMPONENTS.iterdir() if path.is_file()} == expected | {"README.md"}
    for name in expected:
        assert not (ASSETS / "mujoco_assets" / name).exists()


@pytest.mark.parametrize("filename", XML_FILES)
def test_component_resource_paths_remain_asset_root_relative(filename):
    root = ET.parse(COMPONENTS / filename).getroot()
    for element in root.iter():
        for key, value in element.attrib.items():
            if key.startswith("file"):
                path = Path(value)
                assert not path.is_absolute()
                assert ".." not in path.parts
                assert path.parts[:2] == ("meshes", "manipulation_net")
                assert (ASSETS / path).is_file()


@pytest.mark.parametrize("scene", ("magpie_table", "magpie_peg_in_hole",
                                   "magpie_peg_in_hole_left", "magpie_peg_in_hole_right"))
def test_scene_configuration_references_relocated_components(scene):
    config = yaml.safe_load((ROOT / "scene" / f"{scene}.yaml").read_text())
    assert config["scene"]["xml"] == "mujoco_assets/manipulation_net/workcell.xml"
    if config["board"]["enabled"]:
        assert config["board"]["catalog"] == "mujoco_assets/manipulation_net/manipulation_catalog.json"


def test_catalog_component_paths_resolve_inside_component_directory():
    catalog = json.loads((COMPONENTS / "manipulation_catalog.json").read_text())
    references = [catalog["board"]["xml"], *catalog["board"]["collision_xmls"].values()]
    references.extend(variant["xml"] for family in catalog["variants"].values()
                      for variant in family.values())
    for reference in references:
        path = ASSETS / reference
        assert path.parent == COMPONENTS
        assert path.is_file()
    assert catalog["board"]["source"]["file"].startswith("meshes/manipulation_net/")


@pytest.mark.parametrize("filename", ("scene_builder.py", "utility/prepare_manipulation_assets.py",
                                      "tests/test_scene_builder.py", "tests/test_manipulation_assets.py"))
def test_updated_python_paths_have_valid_syntax_and_no_flat_references(filename):
    source = (ROOT / filename).read_text()
    ast.parse(source, filename=filename)
    for name in (*XML_FILES, "manipulation_catalog.json"):
        assert f"mujoco_assets/{name}" not in source
