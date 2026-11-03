"""Prepare source-derived manipulation meshes and standalone MJCF components.

Run from any directory with the project's Python environment. Source STL files
are never modified. All mesh paths in generated XML are CL_Assets-root-relative.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "CL_Assets"
SOURCE = ASSET_ROOT / "meshes" / "manipulation_net"
SCALE = 0.01
SHAPES = {"cylinder": "cylinder", "hex": "hexagonal", "triangle": "triangular",
          "rectangle": "rectangular", "l": "l-shaped"}
PROFILE_TOLERANCE = 1e-6  # Source units: 10 nm after conversion to metres.
WALL_PLANARITY_TOLERANCE = 1e-4  # 1 um; accommodates source cosmetic engraving noise.


def read_stl(path: Path) -> np.ndarray:
    """Read binary STL vertices without trusting normals or units in its header."""
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path}: not a complete binary STL")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].astype(float)


def surface_polygons(triangles: np.ndarray, z: float):
    from shapely import Polygon, union_all

    faces = triangles[np.all(np.isclose(triangles[:, :, 2], z, atol=1e-7), axis=1)]
    # STL exporters duplicate vertices, sometimes with last-bit differences.
    return union_all([Polygon(np.round(face[:, :2], 6)) for face in faces]).buffer(0)


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return list(geometry.geoms)


def validate_extruded_source(triangles, expected_levels, name):
    """Reject tapered/chamfered replacements instead of silently extruding them."""
    levels = np.unique(triangles[:, :, 2])
    if len(levels) != len(expected_levels) or not np.allclose(levels, expected_levels, atol=1e-6, rtol=0):
        raise ValueError(f"{name}: unsupported height levels {levels}; inspect and update the collision model")
    horizontal = np.all(np.isclose(triangles[:, :, 2], triangles[:, :1, 2], atol=1e-7, rtol=0), axis=1)
    walls = triangles[~horizontal, :, :2]
    a, b = walls[:, 1] - walls[:, 0], walls[:, 2] - walls[:, 0]
    longest = np.max(np.linalg.norm(np.stack([a, b, a - b], axis=1), axis=2), axis=1)
    if np.any(longest == 0):
        raise ValueError(f"{name}: degenerate source wall")
    off_plane = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / longest
    maximum = float(off_plane.max())
    if maximum > WALL_PLANARITY_TOLERANCE:
        raise ValueError(f"{name}: nonvertical/tapered source walls ({maximum * SCALE:g} m); extrusion is not valid")
    return maximum * SCALE


def convex_parts(profile):
    """Constrained triangulation followed by exact adjacent-convex merging.

    Unconstrained Delaunay triangulation must not be used here: filtering its
    triangle centroids is insufficient to preserve narrow concave boundaries.
    """
    import mapbox_earcut
    from shapely import Polygon

    result = []
    for poly in polygons(profile):
        rings = [np.asarray(poly.exterior.coords)[:-1]]
        rings.extend(np.asarray(r.coords)[:-1] for r in poly.interiors)
        vertices = np.concatenate(rings)
        indices = mapbox_earcut.triangulate_float64(vertices, np.cumsum([len(r) for r in rings], dtype=np.uint32))
        triangles = indices.reshape(-1, 3)
        regions = {i: Polygon(vertices[t]) for i, t in enumerate(triangles)}
        neighbors = {i: set() for i in regions}
        edge_owner = {}
        for i, tri in enumerate(triangles):
            for a, b in zip(tri, np.roll(tri, -1)):
                edge = tuple(sorted((int(a), int(b))))
                if edge in edge_owner:
                    j = edge_owner[edge]
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                else:
                    edge_owner[edge] = i
        queue = deque((i, j) for i in neighbors for j in sorted(neighbors[i]) if i < j)
        while queue:
            i, j = queue.popleft()
            if i not in regions or j not in regions or j not in neighbors[i]:
                continue
            merged = regions[i].union(regions[j])
            if merged.geom_type != "Polygon":
                continue
            hull = merged.convex_hull
            if hull.area - merged.area > max(1e-12, merged.area * 1e-11):
                continue
            regions[i] = hull
            del regions[j]
            adjacent = (neighbors[i] | neighbors.pop(j)) - {i, j}
            neighbors[i] = adjacent
            for k in sorted(adjacent):
                neighbors[k].discard(j)
                neighbors[k].add(i)
                queue.append((i, k))
        parts = list(regions.values())
        if abs(sum(p.area for p in parts) - poly.area) > 1e-7:
            raise ValueError("Constrained collision decomposition changed polygon area")
        result.extend(parts)
    return result


def prism_vertices(poly, bottom: float, top: float):
    xy = np.asarray(poly.exterior.coords)[:-1] * SCALE
    return np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in (bottom, top)])


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def add_prism(asset, body, name, poly, bottom, top, *, file_path=None, check=False):
    vertices = prism_vertices(poly, bottom, top)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices -= center
    mesh_args = {"name": name}
    if file_path is None:
        mesh_args["vertex"] = numbers(vertices)
    else:
        from scipy.spatial import ConvexHull

        faces = ConvexHull(vertices).simplices
        # Generated derived OBJ, not an edit of the source STL.
        lines = ["# Source-derived convex prism in metres"]
        lines.extend("v " + numbers(v) for v in vertices)
        lines.extend("f " + " ".join(str(int(i) + 1) for i in face) for face in faces)
        write_generated(file_path, "\n".join(lines) + "\n", check)
        mesh_args["file"] = file_path.relative_to(ASSET_ROOT).as_posix()
    ET.SubElement(asset, "mesh", mesh_args)
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, pos=numbers(center),
                  mass="0", group="3", rgba="0.35 0.65 0.85 0.2", margin="0", gap="0",
                  friction="0.5 0.005 0.0001", solref="0.004 1", solimp="0.95 0.99 0.0001")


def write_generated(path, text, check=False):
    if check:
        if not path.is_file() or path.read_text() != text:
            raise ValueError(f"Missing or stale generated asset: {path}; rerun preparation without --check")
    else:
        path.write_text(text)


def write_xml(root, path, check=False):
    ET.indent(root, space="  ")
    write_generated(path, ET.tostring(root, encoding="unicode") + "\n", check)


def source_record(path):
    triangles = read_stl(path)
    # Rounding is below source float32 resolution at the board extents.
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 6), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"file": path.relative_to(ASSET_ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "triangle_count": len(triangles), "welded_vertex_count": len(vertices),
            "nonmanifold_edge_count_at_10nm_weld": int(np.sum(counts != 2)),
            "bounds_source_units": [triangles.min(axis=(0, 1)).tolist(), triangles.max(axis=(0, 1)).tolist()],
            "z_levels_source_units": np.unique(triangles[:, :, 2]).tolist()}


def profile_inertia(profile, length, density):
    """Exact planar polygon integrals for a uniform extruded peg (SI units)."""
    from shapely.geometry.polygon import orient

    xy = np.asarray(orient(profile, sign=1).exterior.coords) * SCALE
    a, b = xy[:-1], xy[1:]
    cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    area = cross.sum() / 2
    ex2 = np.sum((a[:, 0] ** 2 + a[:, 0] * b[:, 0] + b[:, 0] ** 2) * cross) / (12 * area)
    ey2 = np.sum((a[:, 1] ** 2 + a[:, 1] * b[:, 1] + b[:, 1] ** 2) * cross) / (12 * area)
    exy = np.sum((2 * a[:, 0] * a[:, 1] + a[:, 0] * b[:, 1] + b[:, 0] * a[:, 1]
                  + 2 * b[:, 0] * b[:, 1]) * cross) / (24 * area)
    mass = density * area * length
    return mass, [mass * (ey2 + length**2 / 12), mass * (ex2 + length**2 / 12),
                  mass * (ex2 + ey2), -mass * exy, 0, 0]


def aligned_pose(profile, hole, shape):
    from scipy.optimize import minimize
    from shapely import covers, distance, points
    from shapely.affinity import rotate, translate

    def longest_angle(poly):
        edges = np.diff(np.asarray(poly.simplify(1e-5).exterior.coords), axis=0)
        edge = max(edges, key=np.linalg.norm)
        return np.arctan2(edge[1], edge[0])

    angle = 0.0 if shape == "cylinder" else longest_angle(hole) - longest_angle(profile)
    # Symmetric families have equivalent orientations; use the smallest rotation.
    period = {"cylinder": 2 * np.pi, "hex": np.pi / 3, "triangle": 2 * np.pi / 3,
              "rectangle": np.pi, "l": 2 * np.pi}[shape]
    angle = (angle + period / 2) % period - period / 2
    rotated = rotate(profile, angle, origin=(0, 0), use_radians=True)
    center = np.array(hole.centroid.coords[0])
    offset = np.zeros(2)
    if shape == "l":
        local_hole = translate(hole, -center[0], -center[1])
        coords = np.asarray(rotated.exterior.coords)[:-1]

        def constraints(x):
            pts = points(coords + x[:2])
            signed = distance(pts, local_hole.boundary) * np.where(covers(local_hole, pts), 1, -1)
            return signed - x[2]

        solution = minimize(lambda x: -x[2], [0, 0, 0], method="SLSQP",
                            constraints={"type": "ineq", "fun": constraints},
                            bounds=[(-0.2, 0.2), (-0.2, 0.2), (0, 0.2)],
                            options={"ftol": 1e-12, "maxiter": 100})
        if not solution.success:
            raise ValueError(f"L peg alignment failed: {solution.message}")
        offset = solution.x[:2]
    position = center + offset
    placed = translate(rotated, *position)
    if not hole.covers(placed):
        raise ValueError(f"{shape}: source peg does not fit its source hole")
    clearance = placed.boundary.distance(hole.boundary) * SCALE
    return position * SCALE, [np.cos(angle / 2), 0, 0, np.sin(angle / 2)], clearance


def prepare(*, check=False) -> None:
    from shapely import box, union_all
    from shapely.affinity import translate

    output = ASSET_ROOT / "mujoco_assets" / "manipulation_net"
    collision_dir = SOURCE / "collision"
    if not check:
        output.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(exist_ok=True)
    board_path = SOURCE / "nist_atb_m1.stl"
    board_triangles = read_stl(board_path)
    board_wall_error = validate_extruded_source(board_triangles, [0, 0.3, 2.3, 2.5], board_path.name)
    bounds = [board_triangles.min(axis=(0, 1)), board_triangles.max(axis=(0, 1))]
    if not np.allclose(bounds, [[-16, -12.5, 0], [16, 12.5, 2.5]], atol=1e-6, rtol=0):
        raise ValueError("Board source bounds changed; recalibrate against the specified 0.32 x 0.25 m footprint")
    floors = sorted(polygons(surface_polygons(board_triangles, 0.3)), key=lambda p: p.centroid.x)
    if len(floors) != 20:
        raise ValueError("Expected exactly 20 source blind-hole floors")
    top_surface = surface_polygons(board_triangles, 2.5).simplify(PROFILE_TOLERANCE)
    outer = box(-16, -12.5, 16, 12.5)
    middle = outer.difference(union_all(floors)).simplify(PROFILE_TOLERANCE)
    root = ET.Element("mujoco", model="nist_atb_m1_board")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="board_visual_mesh", file="meshes/manipulation_net/nist_atb_m1.stl", scale="0.01 0.01 0.01")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="board")
    ET.SubElement(body, "geom", name="visual", type="mesh", mesh="board_visual_mesh", contype="0", conaffinity="0", mass="0", group="2", rgba="0.72 0.74 0.76 1")
    ET.SubElement(body, "geom", name="bottom", type="box", pos="0 0 0.0015", size="0.16 0.125 0.0015", mass="0", group="3", rgba="0.35 0.65 0.85 0.2", margin="0", gap="0",
                  friction="0.5 0.005 0.0001", solref="0.004 1", solimp="0.95 0.99 0.0001")
    # Copy only the shared visual/slab skeleton, never the detailed top meshes.
    functional_root = deepcopy(root)
    functional_root.set("model", "nist_atb_m1_board_functional")
    functional_asset = functional_root.find("asset")
    functional_body = functional_root.find("worldbody/body")
    layer_counts = []
    for label, section, bottom, top in (("middle", middle, 0.003, 0.023), ("top", top_surface, 0.023, 0.025)):
        parts = convex_parts(section)
        for i, part in enumerate(parts):
            add_prism(asset, body, f"board_{label}_{i:05d}", part, bottom, top)
            if label == "middle":
                add_prism(functional_asset, functional_body, f"board_middle_{i:05d}", part, bottom, 0.025)
        layer_counts.append(len(parts))
        print(f"Board {label}: {len(parts)} source-derived convex prisms", flush=True)
    catalog = {
        "version": 1,
        "units": "metres, kilograms, radians; quaternion order wxyz",
        "size_semantics": "IDs 0..3 are source hole rows top-to-bottom, tight-to-loose clearance; all four use the same source peg. They are not peg scale factors.",
        "board": {"xml": "mujoco_assets/manipulation_net/manipulation_board.xml", "body": "board", "footprint": [0.32, 0.25], "thickness": 0.025,
                  "scale_calibration": "Source XY extent 32 x 25 units matched to user-specified 0.32 x 0.25 m; uniform 0.01 m/unit, consistent with centimetres, not millimetres",
                  "mounting_plane_z": 0, "hole_type": "blind", "floor_z": 0.003, "entry_z": 0.025,
                  "engraving_floor_z": 0.023, "source": source_record(board_path),
                  "source_transform": {"uniform_scale": SCALE, "translation": [0, 0, 0], "quat": [1, 0, 0, 0]},
                  "collision": {"method": "source horizontal sections, constrained earcut triangulation and adjacent convex merging; extruded convex prisms plus solid bottom slab",
                                "layer_prism_counts": layer_counts, "boundary_simplification_tolerance_m": PROFILE_TOLERANCE * SCALE,
                                "source_vertex_weld_tolerance_m": 1e-8, "engraved_recesses_preserved": True,
                                "maximum_source_wall_nonverticality_m": board_wall_error,
                                "nonverticality_note": "Submicron source cosmetic engraving inconsistencies are projected vertically; functional profiles are source-derived",
                                "mesh_storage": "inline vertices in board XML, centered individually for numerical conditioning"}},
        "physical_assumptions": {"peg_density_kg_m3": 1200, "material": "assumed uniform polymer, not measured",
                                 "inertia": "exact uniform extrusion of source cross section; explicit full inertia at volume centroid",
                                 "board": "fixed body; no dynamic mass needed", "visuals": "massless, contype=conaffinity=0",
                                 "contact_margin_m": 0, "friction": [0.5, 0.005, 0.0001],
                                 "solref": [0.004, 1], "solimp": [0.95, 0.99, 0.0001],
                                 "solver_caveat": "MuJoCo clamps solref time constant to at least twice the simulation timestep; friction and stiffness are simulation assumptions, not measured force calibration",
                                 "precision": "geometry describes supplied float32 STL, not measured manufacturing tolerances"},
        "holes": {}, "variants": {},
    }
    board = catalog["board"]
    board["collision_xmls"] = {
        "functional": "mujoco_assets/manipulation_net/manipulation_board_functional.xml",
        "detailed": board["xml"],
    }
    detailed_collision = dict(board["collision"], collision_geom_count=sum(layer_counts) + 1,
                              layer_z_ranges_m=[[0.003, 0.023], [0.023, 0.025]])
    functional_collision = dict(
        board["collision"], layer_prism_counts=layer_counts[:1],
        collision_geom_count=layer_counts[0] + 1, layer_z_ranges_m=[[0.003, 0.025]],
        engraved_recesses_preserved=False,
        method="same source-derived middle convex partition extruded through full hole depth plus unchanged solid bottom slab",
        difference_from_detailed="Fills only cosmetic engraving recesses at z=0.023..0.025 m; exact functional hole profiles, blind floors, full height and outer footprint are unchanged; original detailed source visual retained",
    )
    board["collision_modes"] = {"functional": functional_collision, "detailed": detailed_collision}
    for family_index, (shape, source_name) in enumerate(SHAPES.items()):
        path = SOURCE / f"atb_m1_{source_name}_peg.stl"
        triangles = read_stl(path)
        wall_error = validate_extruded_source(triangles, [0, 5], path.name)
        source_profile = surface_polygons(triangles, 0)
        if source_profile.hausdorff_distance(surface_polygons(triangles, 5)) > WALL_PLANARITY_TOLERANCE:
            raise ValueError(f"{path.name}: top and bottom profiles differ; a constant extrusion is not valid")
        source_center = np.array(source_profile.centroid.coords[0])
        profile = translate(source_profile, -source_center[0], -source_center[1])
        length = float(triangles[:, :, 2].max()) * SCALE
        peg_root = ET.Element("mujoco", model=f"atb_m1_peg_{shape}")
        ET.SubElement(peg_root, "compiler", angle="radian")
        peg_asset = ET.SubElement(peg_root, "asset")
        ET.SubElement(peg_asset, "mesh", name="visual_mesh", file=path.relative_to(ASSET_ROOT).as_posix(), scale="0.01 0.01 0.01")
        peg_body = ET.SubElement(ET.SubElement(peg_root, "worldbody"), "body", name="peg")
        ET.SubElement(peg_body, "freejoint", name="freejoint")
        mass, inertia = profile_inertia(profile, length, 1200)
        ET.SubElement(peg_body, "inertial", pos=numbers([0, 0, length / 2]), mass=numbers([mass]), fullinertia=numbers(inertia))
        ET.SubElement(peg_body, "geom", name="visual", type="mesh", mesh="visual_mesh", pos=numbers(np.r_[-source_center * SCALE, 0]), mass="0", contype="0", conaffinity="0", group="2", rgba="0.85 0.48 0.16 1")
        parts = convex_parts(profile.simplify(PROFILE_TOLERANCE)) if shape == "l" else [profile.convex_hull]
        for i, part in enumerate(parts):
            add_prism(peg_asset, peg_body, f"collision_{i}", part, 0, length,
                      file_path=collision_dir / f"peg_{shape}_{i}.obj", check=check)
        ET.SubElement(peg_body, "site", name="tip", pos="0 0 0", size="0.001", rgba="0 0.8 0.2 0.7")
        xml_path = output / f"manipulation_peg_{shape}.xml"
        write_xml(peg_root, xml_path, check)
        variant = {"xml": xml_path.relative_to(ASSET_ROOT).as_posix(), "body": "peg", "freejoint": "freejoint", "tip_site": "tip",
                   "length": length, "mass": mass, "inertia_full": inertia, "inertial_pos": [0, 0, length / 2],
                   "cross_section_area": profile.area * SCALE**2, "collision_part_count": len(parts),
                   "maximum_source_wall_nonverticality_m": wall_error,
                   "bounds": [np.r_[np.array(profile.bounds)[:2] * SCALE, 0].tolist(), np.r_[np.array(profile.bounds)[2:] * SCALE, length].tolist()],
                   "source": source_record(path), "source_transform": {"uniform_scale": SCALE,
                       "translation": np.r_[-source_center * SCALE, 0].tolist(), "quat": [1, 0, 0, 0]},
                   "profile_xy": (np.asarray(profile.exterior.coords)[:-1] * SCALE).tolist()}
        catalog["variants"][shape] = {str(size): dict(variant) for size in range(4)}
        holes = sorted(floors[family_index * 4:family_index * 4 + 4], key=lambda p: -p.centroid.y)
        for size, hole in enumerate(holes):
            xy, quat, clearance = aligned_pose(profile, hole, shape)
            name = f"{shape}_{size}"
            entry = np.r_[xy, 0.025].tolist()
            target = np.r_[xy, 0.003].tolist()
            entry_site, target_site = f"hole_{name}_entry", f"hole_{name}_target"
            for site_name, position in ((entry_site, entry), (target_site, target)):
                site = ET.SubElement(body, "site", name=site_name, pos=numbers(position), quat=numbers(quat), size="0.0007", rgba="0.1 0.8 0.2 0.7")
                functional_body.append(deepcopy(site))
            catalog["holes"][name] = {"shape": shape, "size": size, "entry_site": entry_site, "target_site": target_site,
                                      "entry": entry, "target": target, "quat": quat,
                                      "max_insertion_depth": 0.022, "seated_depth": 0.022, "hole_type": "blind",
                                      "minimum_wall_clearance_at_site_pose": clearance,
                                      "profile_centroid": (np.array(hole.centroid.coords[0]) * SCALE).tolist(),
                                      "profile_xy": (np.asarray(hole.exterior.coords)[:-1] * SCALE).tolist()}
    write_xml(root, output / "manipulation_board.xml", check)
    write_xml(functional_root, output / "manipulation_board_functional.xml", check)
    write_generated(output / "manipulation_catalog.json", json.dumps(catalog, indent=2, sort_keys=True) + "\n", check)
    print(("Verified" if check else "Prepared") + " 2 board collision modes, 5 physical peg components, and 20 hole contracts")


def inspect_sources() -> None:
    report = {}
    for path in sorted(SOURCE.glob("*.stl")):
        triangles = read_stl(path)
        record = source_record(path)
        if path.name == "nist_atb_m1.stl":
            for z in (0.3, 2.3, 2.5):
                parts = polygons(surface_polygons(triangles, z))
                record[f"surfaces_{z}"] = [
                    {"bounds": list(p.bounds), "area": p.area,
                     "centroid": list(p.centroid.coords)[0], "vertices": len(p.exterior.coords) - 1,
                     "interiors": len(p.interiors)} for p in parts if p.area > 1e-8]
        else:
            p = surface_polygons(triangles, 0)
            record["profile"] = {"area": p.area, "centroid": list(p.centroid.coords)[0],
                                 "vertices": len(p.exterior.coords) - 1}
        report[path.name] = record
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="Report binary source bounds, topology, and planar surfaces")
    mode.add_argument("--check", action="store_true", help="Regenerate in memory and reject missing/stale outputs without writes")
    args = parser.parse_args()
    if args.inspect:
        inspect_sources()
    else:
        prepare(check=args.check)
