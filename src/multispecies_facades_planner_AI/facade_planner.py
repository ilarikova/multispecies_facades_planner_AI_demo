#global import
import math
import re
import numpy as np
from shapely.geometry import Point
import random
#local import
from . import facade_planner_functions as fpf
from .data_extraction import load_species_core_as_dict
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any
from typing import List, Dict, Any, Tuple

# NOTE: this module used to load an Excel sheet and a building JSON from
# hardcoded absolute paths at import time. Both were dead leftovers — every
# `building_dict` below is a function parameter — and they made the module
# unimportable anywhere but this one machine (one path even pointed into the
# old `multispecies_facades_planner` repo). Removed 2026-08-04 so the package
# can be deployed, e.g. in the Streamlit demo.
def to_meters_range(val, default=(200.0, 200.0)):
    a, b = fpf.parse_range_float(val, default=default)
    a = float(a)
    b = float(b)
    lo = min(a, b)
    hi = max(a, b)

    if hi > 10.0:
        return lo / 100.0, hi / 100.0

    return lo, hi


def dist_xyz(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def filter_by_other_colonies(cands, prev_xyz_list, min_dist_m):
    if not prev_xyz_list or min_dist_m <= 0:
        return cands

    out = []
    for pid, uv, xyz in cands:
        if all(dist_xyz(xyz, pxyz) >= min_dist_m for pxyz in prev_xyz_list):
            out.append((pid, uv, xyz))

    return out


def distance_at_least_m(pts, min_dist_m: float = 10.0) -> bool:
    """
    True iff every pair of points is >= min_dist_m apart.
    """
    if not pts:
        return False

    if len(pts) == 1:
        return True

    d2 = float(min_dist_m) * float(min_dist_m)

    for i in range(len(pts)):
        x1, y1, z1 = pts[i]

        for j in range(i + 1, len(pts)):
            x2, y2, z2 = pts[j]

            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            if (dx * dx + dy * dy + dz * dz) < d2:
                return False

    return True


def base_candidates_for_wall(
    building_dict: dict,
    wall_id: str,
    min_height_m: float,
    needs: dict | None = None,
):
    """
    Return all usable grid points on one wall above the species minimum height.
    No hottest-wall / height-band logic here.

    If needs["distance_to_roof"] is "strict", points are additionally
    restricted to within fpf.ROOF_STRICT_BAND_M of the nearest roof edge
    (follows sloped/triangular roof lines, see fpf.point_to_top_edge_distance).
    """
    wall = building_dict[wall_id]

    free_geom = fpf.derive_openings(wall)
    offset_geom = fpf.build_offset_area(wall)
    usable_geom = free_geom.difference(offset_geom)

    if usable_geom.is_empty:
        return None, None

    grid = wall.get("grid", {}) or {}
    if not grid:
        return None, None

    roof_strict = fpf.is_distance_to_roof_strict((needs or {}).get("distance_to_roof"))
    boundary_uv = wall.get("boundary_uv") or []
    v_up = fpf.v_axis_points_up(wall)

    zs_all = [
        pdata["point_on_wall"][2]
        for pdata in grid.values()
        if pdata.get("point_on_wall")
    ]
    zs_all = [z for z in zs_all if isinstance(z, (int, float))]

    if not zs_all:
        return None, None

    wall_ground_z = min(zs_all)

    base_candidates = []

    for pid, pdata in grid.items():
        uv = pdata.get("uv")
        xyz = pdata.get("point_on_wall")

        if not uv or not xyz:
            continue

        z = float(xyz[2])
        rel_h = z - float(wall_ground_z)

        if rel_h < min_height_m:
            continue

        p = Point(float(uv[0]), float(uv[1]))

        if not usable_geom.contains(p):
            continue

        if roof_strict and not fpf.is_within_roof_strict_band(
            boundary_uv, float(uv[0]), float(uv[1]), v_up=v_up
        ):
            continue

        base_candidates.append((
            pid,
            [float(uv[0]), float(uv[1])],
            [float(xyz[0]), float(xyz[1]), z],
        ))

    if not base_candidates:
        return None, None

    return base_candidates, wall_ground_z



def parse_next_nest_distance_m(
    val,
    default=(1.0, 1.5),
    close_default=(0.0, 0.5),
):
    """
    Parse distance_to_next_nest values into meters.

    Examples:
    '0.1 - 0.15'          -> (0.1, 0.15)
    '6 - 7'               -> (6.0, 7.0)
    '10-12'               -> (10.0, 12.0)
    '10 - 12 m'           -> (10.0, 12.0)
    'as close as possible'-> (0.0, 0.5)
    """
    if val is None:
        return default

    s = str(val).strip().lower()

    if not s or s in {"nan", "none"}:
        return default

    if "as close as possible" in s or "close" in s:
        return close_default

    s = (
        s.replace("–", "-")
        .replace("—", "-")
        .replace("to", "-")
        .replace(",", ".")
    )

    nums = re.findall(r"\d+(?:\.\d+)?", s)

    if len(nums) >= 2:
        a = float(nums[0])
        b = float(nums[1])
        return min(a, b), max(a, b)

    if len(nums) == 1:
        v = float(nums[0])
        return v, v

    return default

def build_one_colony_on_wall(
    building_dict: dict,
    wall_id: str,
    specie1_needs: dict,
    min_height_m: float,
    prev_xyz_all: list,
    colonies_size=None,
    attempts_per_wall: int = 100,
    seed: int | None = None,
    *,
    mode: str = "colony",
    solitary_min_dist_m: float = 10.0,
    solitary_min_nests: int = 1,
    solitary_max_nests: int = 10,
    base_candidates: list | None = None,
):
    """
    Returns one candidate layout dict or None.

    base_candidates: optional precomputed output of base_candidates_for_wall().
      It is a pure function of (building_dict, wall_id, min_height_m, needs) —
      all constant across the many attempts made on one wall — so the caller
      should compute it once per wall and pass it in. Recomputing it per attempt
      dominates generation time (a shapely contains() test over the whole grid).
      The list is never mutated here; it is copied before shuffling.

    mode="colony":
      - uses specie1_needs["colonie_size_local"] first
      - falls back to specie1_needs["colonie_size"]
      - normal species: accepts colony size within ±10% tolerance
      - bats: exact colony size required (no tolerance)

    mode="solitary":
      - targets one or several solitary nests depending on input
      - enforces minimum pairwise distance
    """
    rng = random.Random(seed) if seed is not None else random

    mode_norm = (mode or "colony").strip().lower()
    if mode_norm not in {"colony", "solitary"}:
        raise ValueError("mode must be 'colony' or 'solitary'.")

    taxa = (specie1_needs.get("taxa") or "").strip().lower()
    is_bat = taxa == "bat"

    max_allowed_colony_size = None

    if mode_norm == "colony":
        colonie_size_raw = (
            specie1_needs.get("colonie_size_local")
            or specie1_needs.get("colonie_size")
        )

        size_min, size_max = fpf.parse_min_max_count(
            colonie_size_raw,
            default=(5, 20),
        )

        size_min = int(size_min)
        size_max = (
            int(size_max)
            if size_max is not None and not np.isnan(size_max)
            else size_min
        )

        # bats: exact count required; other species: ±10% tolerance
        size_tol = 0.0 if is_bat else 0.10

        min_allowed_colony_size = max(
            1,
            int(math.floor(size_min * (1.0 - size_tol))),
        )

        max_allowed_colony_size = int(
            math.ceil(size_max * (1.0 + size_tol))
        )

        dist_raw = (
            specie1_needs.get("distance_to_next_nest")
            or specie1_needs.get("if_colonial_distance_to_next_nest")
            or specie1_needs.get("distance_between_nest_boxes")
        )

        dist_a, dist_b = parse_next_nest_distance_m(
            dist_raw,
            default=(1.0, 1.5),
            close_default=(0.0, 0.5),
        )

        dmin_m = min(dist_a, dist_b)
        dmax_m = max(dist_a, dist_b)

    else:
        size_min = int(solitary_min_nests)
        size_max = int(solitary_max_nests)

        dmin_m = float(solitary_min_dist_m)
        dmax_m = 1e9

        min_allowed_colony_size = size_min
        max_allowed_colony_size = size_max

    if base_candidates is None:
        base_candidates, _wall_ground_z = base_candidates_for_wall(
            building_dict=building_dict,
            wall_id=wall_id,
            min_height_m=min_height_m,
            needs=specie1_needs,
        )

    if not base_candidates:
        return None

    base_candidates = filter_by_other_colonies(
        base_candidates,
        prev_xyz_all,
        dmin_m,
    )

    if not base_candidates:
        return None

    best_selected = None
    best_target_size = None

    for _attempt in range(attempts_per_wall):
        if colonies_size is not None:
            target_size = int(colonies_size)
        else:
            target_size = rng.randint(
                int(min_allowed_colony_size),
                int(max_allowed_colony_size),
            )

        if mode_norm == "solitary":
            target_size = max(size_min, min(size_max, target_size))

        candidates = list(base_candidates)
        rng.shuffle(candidates)

        selected = fpf.pick_points_chained_band(
            candidates=candidates,
            target_size=target_size,
            dmin_m=dmin_m,
            dmax_m=dmax_m,
            max_tries_per_step=400,
            rng=rng,
        )

        if len(selected) < min_allowed_colony_size:
            continue

        if mode_norm == "colony" and len(selected) > max_allowed_colony_size:
            selected = selected[:max_allowed_colony_size]

        if mode_norm == "solitary":
            sel_xyz_tmp = [t[2] for t in selected]

            if not distance_at_least_m(
                sel_xyz_tmp,
                min_dist_m=solitary_min_dist_m,
            ):
                continue

        # bats: only keep attempts that hit the exact target
        if is_bat and mode_norm == "colony":
            if len(selected) != target_size:
                continue

        if best_selected is None or len(selected) > len(best_selected):
            best_selected = selected
            best_target_size = target_size

        if len(selected) >= target_size:
            best_selected = selected
            best_target_size = target_size
            break

    if not best_selected:
        return None

    sel_ids = [t[0] for t in best_selected]
    sel_uvs = [t[1] for t in best_selected]
    sel_xyz = [t[2] for t in best_selected]

    requested = (
        int(best_target_size)
        if best_target_size is not None
        else len(best_selected)
    )

    achieved = len(best_selected)

    if achieved < min_allowed_colony_size:
        return None

    if mode_norm == "colony" and achieved > max_allowed_colony_size:
        return None

    # bats: final strict check — exact count only
    if is_bat and mode_norm == "colony" and achieved != requested:
        return None

    out = {
        "selected_point_ids": sel_ids,
        "uv": sel_uvs,
        "xyz": sel_xyz,
        "requested_colony_size": int(requested),
        "achieved_colony_size": int(achieved),
        "colony_size": int(achieved),
        "min_allowed_colony_size": int(min_allowed_colony_size),
        "max_allowed_colony_size": int(max_allowed_colony_size),
        "dist_range_m": [float(dmin_m), float(dmax_m)],
        "seed": seed,
        "mode": mode_norm,
        "taxa": taxa,
        "is_bat_colony": bool(is_bat),
    }

    if achieved < requested:
        out["colony_note"] = "below requested but within allowed tolerance"

    return out


@dataclass
class Candidate:
    candidate_id: str
    wall_id: str
    species: str
    colony_index: int
    xyz: List
    uv: List | None
    wall_score: float

    wall_meta: Dict
    species_needs: Dict

    features: Dict[str, float] | None = None
    score: float | None = None
    score_breakdown: Dict[str, float] | None = None


def colony_signature(
    colony: Dict[str, Any],
    tol: float = 0.02,
) -> Tuple[Tuple[int, int, int], ...]:
    """
    De-duplicate colony layouts by quantized XYZ points.
    """
    pts = colony.get("xyz") or []
    q = [
        (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        for p in pts
    ]
    q.sort()
    return tuple(q)


def solitary_signature(
    layout: Dict[str, Any],
    tol: float = 0.02,
) -> Tuple[Tuple[int, int, int], ...]:
    """
    De-duplicate solitary layouts by quantized XYZ points.
    """
    pts = layout.get("xyz") or []
    q = [
        (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        for p in pts
    ]
    q.sort()
    return tuple(q)


def is_wall_unsuitable(wall: dict) -> bool:
    if not wall:
        return True

    floor_fn = (wall.get("floor_function") or "").strip().lower()

    if floor_fn in {"neighbor_building", "neigbor_building"}:
        return True

    return False



def generate_colony_candidates(
    building_dict: dict,
    specie_needs: dict,
    *,
    attempts_per_wall: int = 100,
    max_candidates_per_wall: int = 20,
    base_seed: int | None = None,
    signature_tol_m: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Generate colony candidates.

    One candidate = one local colony on one wall.
    """
    rng = random.Random(base_seed) if base_seed is not None else random.Random()

    if not specie_needs or len(specie_needs) != 1:
        raise ValueError("specie_needs must have exactly one species entry.")

    species_name = next(iter(specie_needs.keys()))
    needs = specie_needs[species_name] or {}

    # roof entries and other geometry-less records have no boundary_uv/grid;
    # including them here means one wasted build attempt each, per iteration.
    wall_rows: List[str] = [
        wall_id
        for wall_id, wall in building_dict.items()
        if isinstance(wall, dict)
        and not wall_id.startswith("_")
        and (wall.get("boundary_uv") or [])
        and not is_wall_unsuitable(wall)
    ]

    min_height_m = fpf.parse_min_height_m(needs.get("nest_height"), 0.0)

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for wall_id in wall_rows:
        wall = building_dict[wall_id]

        # computed once per wall, reused by every attempt on it
        wall_base_candidates, _ = base_candidates_for_wall(
            building_dict=building_dict,
            wall_id=wall_id,
            min_height_m=min_height_m,
            needs=needs,
        )

        if not wall_base_candidates:
            continue

        for _ in range(max_candidates_per_wall):
            attempt_seed = rng.randint(0, 1_000_000_000)

            colony = build_one_colony_on_wall(
                building_dict=building_dict,
                wall_id=wall_id,
                base_candidates=wall_base_candidates,
                specie1_needs=needs,
                min_height_m=min_height_m,
                prev_xyz_all=[],
                colonies_size=None,
                attempts_per_wall=attempts_per_wall,
                seed=attempt_seed,
                mode="colony",
            )

            if colony is None:
                continue

            sig = colony_signature(colony, tol=signature_tol_m)

            if sig in seen:
                continue

            seen.add(sig)

            candidates.append({
                "candidate_id": str(uuid.uuid4()),
                "wall_id": wall_id,
                "species": species_name,
                "colony_index": 0,
                "xyz": colony["xyz"],
                "uv": colony.get("uv"),
                "selected_point_ids": colony.get("selected_point_ids"),
                "wall_meta": wall,
                "species_needs": needs,
                "seed": attempt_seed,

                # candidate colony size
                "colony_size": colony["achieved_colony_size"],
                "requested_colony_size": colony["requested_colony_size"],
                "min_allowed_colony_size": colony["min_allowed_colony_size"],
            })

    return candidates


def generate_solitary_candidates(
    building_dict: dict,
    specie_needs: dict,
    *,
    attempts_per_wall: int = 100,
    max_candidates_per_wall: int = 20,
    base_seed: int | None = None,
    signature_tol_m: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Generate candidates for solitary species.

    One candidate = one individual nest box on one wall.
    Building-level count and spacing are handled later in the planner.
    """
    rng = random.Random(base_seed) if base_seed is not None else random.Random()

    if not specie_needs or len(specie_needs) != 1:
        raise ValueError("specie_needs must have exactly one species entry.")

    species_name = next(iter(specie_needs.keys()))
    needs = specie_needs[species_name] or {}

    # roof entries and other geometry-less records have no boundary_uv/grid;
    # including them here means one wasted build attempt each, per iteration.
    wall_rows: List[str] = [
        wall_id
        for wall_id, wall in building_dict.items()
        if isinstance(wall, dict)
        and not wall_id.startswith("_")
        and (wall.get("boundary_uv") or [])
        and not is_wall_unsuitable(wall)
    ]

    min_height_m = fpf.parse_min_height_m(needs.get("nest_height"), 0.0)

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for wall_id in wall_rows:
        wall = building_dict[wall_id]

        # computed once per wall, reused by every attempt on it
        wall_base_candidates, _ = base_candidates_for_wall(
            building_dict=building_dict,
            wall_id=wall_id,
            min_height_m=min_height_m,
            needs=needs,
        )

        if not wall_base_candidates:
            continue

        for _ in range(max_candidates_per_wall):
            attempt_seed = rng.randint(0, 1_000_000_000)

            layout = build_one_colony_on_wall(
                building_dict=building_dict,
                wall_id=wall_id,
                base_candidates=wall_base_candidates,
                specie1_needs=needs,
                min_height_m=min_height_m,
                prev_xyz_all=[],
                colonies_size=1,
                attempts_per_wall=attempts_per_wall,
                seed=attempt_seed,
                mode="solitary",
                solitary_min_dist_m=0.0,
                solitary_min_nests=1,
                solitary_max_nests=1,
            )

            if layout is None:
                continue

            pts = layout.get("xyz") or []

            if len(pts) != 1:
                continue

            sig = solitary_signature({"xyz": pts}, tol=signature_tol_m)

            if sig in seen:
                continue

            seen.add(sig)

            candidates.append({
                "candidate_id": str(uuid.uuid4()),
                "wall_id": wall_id,
                "species": species_name,
                "colony_index": 0,
                "xyz": pts,
                "uv": layout.get("uv"),
                "selected_point_ids": layout.get("selected_point_ids"),
                "wall_meta": wall,
                "species_needs": needs,
                "seed": attempt_seed,

                # candidate size
                "nests_placed": 1,

                # candidate type
                "mode": "solitary",
            })

    return candidates