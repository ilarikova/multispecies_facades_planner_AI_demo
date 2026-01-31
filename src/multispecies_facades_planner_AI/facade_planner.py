#global import
import math
import compas_ghpython
from shapely.geometry import Point
import random
#local import
from . import facade_planner_functions as fpf
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any
from typing import List, Dict, Any, Tuple


def to_meters_range(val, default=(200.0, 200.0)):
    a, b = fpf.parse_range_float(val, default=default)
    a = float(a)
    b = float(b)
    lo = min(a, b)
    hi = max(a, b)
    if hi > 10.0:
        return (lo / 100.0, hi / 100.0)
    return (lo, hi)

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
    n = len(pts)
    for i in range(n):
        x1, y1, z1 = pts[i]
        for j in range(i + 1, n):
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
    hottest_wall_id: str,
    low_band_m: float = 3.0,
    high_band_m: float = 10.0,
):
    wall = building_dict[wall_id]
    free_geom = fpf.derive_openings(wall)
    offset_geom = fpf.build_offset_area(wall)
    usable_geom = free_geom.difference(offset_geom)
    if usable_geom.is_empty:
        return None, None, None, None

    grid = wall.get("grid", {}) or {}
    if not grid:
        return None, None, None, None

    zs_all = [pdata["point_on_wall"][2] for pdata in grid.values() if pdata.get("point_on_wall")]
    zs_all = [z for z in zs_all if isinstance(z, (int, float))]
    if not zs_all:
        return None, None, None, None
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

        base_candidates.append((pid, [float(uv[0]), float(uv[1])], [float(xyz[0]), float(xyz[1]), z]))

    if not base_candidates:
        return None, None, None, None

    zmin_allowed = wall_ground_z + min_height_m
    zs_c = [c[2][2] for c in base_candidates]
    zmax_wall = max(zs_c)

    if wall_id == hottest_wall_id:
        z_cap = zmin_allowed + low_band_m
        base_candidates = [c for c in base_candidates if c[2][2] <= z_cap]
        height_band_label = "low+5m"
    else:
        z_floor = zmax_wall - high_band_m
        if z_floor < zmin_allowed:
            z_floor = zmin_allowed
        base_candidates = [c for c in base_candidates if c[2][2] >= z_floor]
        height_band_label = "top-6m"

    if not base_candidates:
        return None, None, None, None

    return base_candidates, wall_ground_z, None, height_band_label


def build_one_colony_on_wall(
    building_dict: dict,
    wall_id: str,
    specie1_needs: dict,
    min_height_m: float,
    hottest_wall_id: str,
    prev_xyz_all: list,
    colonies_size=None,
    attempts_per_wall: int = 20,
    low_band_m: float = 6.0,
    high_band_m: float = 10.0,
    seed: int | None = None,
    *,
    mode: str = "colony",                 # "colony" or "solitary"
    solitary_min_dist_m: float = 10.0,    # used only in mode="solitary"
    solitary_min_nests: int = 1,
    solitary_max_nests: int = 10,
):
    """
    Returns one solution (dict) or None.

    mode="colony":
      - uses specie1_needs["colonie_size"] (range) and specie1_needs["if_colonial_distance_to_next_nest"] (range)
      - accepts >= 80% of requested size

    mode="solitary":
      - ignores colony ranges in specie1_needs
      - targets 1..10 nests (or colonies_size if provided)
      - enforces all selected nests are within solitary_max_dist_m of the first nest
      - still accepts >= 80% of requested size (so a requested 10 could accept 8)
    """
    rng = random.Random(seed) if seed is not None else random

    # ---- other-species distance is still valid in both modes ----
    other_min_m, _ = to_meters_range(
        specie1_needs.get("distance_other_species"),
        default=(200.0, 200.0),
    )

    # ---- parse colony sizing / spacing depending on mode ----
    mode_norm = (mode or "colony").strip().lower()
    if mode_norm not in {"colony", "solitary"}:
        raise ValueError("mode must be 'colony' or 'solitary'.")

    if mode_norm == "colony":
        colonie_size_raw = specie1_needs.get("colonie_size")
        size_min, size_max = fpf.parse_range_int(colonie_size_raw, default=(5, 20))

        dist_cm_a, dist_cm_b = fpf.parse_range_float(
            specie1_needs.get("if_colonial_distance_to_next_nest"),
            default=(50.0, 100.0),
        )
        dmin_m = min(dist_cm_a, dist_cm_b) / 100.0
        dmax_m = max(dist_cm_a, dist_cm_b) / 100.0

        min_allowed_colony_size = size_min
        if isinstance(colonie_size_raw, str) and colonie_size_raw.strip().startswith(">"):
            min_allowed_colony_size = size_min  # keep your original intent; or clamp if you want

    else:
         # solitary: enforce MIN spacing between any two nests
        size_min = int(solitary_min_nests)
        size_max = int(solitary_max_nests)

        dmin_m = float(solitary_min_dist_m)
        dmax_m = 1e9  # effectively "no maximum"

        min_allowed_colony_size = size_min

    # ---- base candidates on wall ----
    base_candidates, _wall_ground_z, _ws_unused, height_band_label = base_candidates_for_wall(
        building_dict=building_dict,
        wall_id=wall_id,
        min_height_m=min_height_m,
        hottest_wall_id=hottest_wall_id,
        low_band_m=low_band_m,
        high_band_m=high_band_m,
    )
    if not base_candidates:
        return None

    base_candidates = filter_by_other_colonies(base_candidates, prev_xyz_all, other_min_m)
    if not base_candidates:
        return None

    best_selected = None
    best_target_size = None

    # ---- main exploration loop ----
    for _attempt in range(attempts_per_wall):
        # target size
        if colonies_size is not None:
            target_size = int(colonies_size)
        else:
            target_size = rng.randint(size_min, size_max)

        # safety clamp in solitary mode
        if mode_norm == "solitary":
            target_size = max(size_min, min(size_max, target_size))

        # diversify order
        candidates = list(base_candidates)
        rng.shuffle(candidates)

        selected = fpf.pick_points_chained_band(
            candidates=candidates,
            target_size=target_size,
            dmin_m=dmin_m,
            dmax_m=dmax_m,
            max_tries_per_step=400,
        )

        if len(selected) < min_allowed_colony_size:
            continue

        # additional solitary constraint: all within 10m of anchor
        if mode_norm == "solitary":
            sel_xyz_tmp = [t[2] for t in selected]
            if not distance_at_least_m(sel_xyz_tmp, min_dist_m=solitary_min_dist_m):
                continue

        # keep best seen so far
        if best_selected is None or len(selected) > len(best_selected):
            best_selected = selected
            best_target_size = target_size

        # stop early if we hit target
        if len(selected) >= target_size:
            best_selected = selected
            best_target_size = target_size
            break

    if not best_selected:
        return None

    sel_ids = [t[0] for t in best_selected]
    sel_uvs = [t[1] for t in best_selected]
    sel_xyz = [t[2] for t in best_selected]

    target_size = int(best_target_size) if best_target_size is not None else len(best_selected)
    achieved = len(best_selected)

    accept_80 = int(math.ceil(0.8 * float(target_size)))
    accepted = achieved >= target_size or achieved >= accept_80
    if not accepted:
        return None

    out = {
        "selected_point_ids": sel_ids,
        "uv": sel_uvs,
        "xyz": sel_xyz,
        "requested_colony_size": int(target_size),
        "achieved_colony_size": int(achieved),
        "min_allowed_colony_size": int(min_allowed_colony_size),
        "dist_range_m": [float(dmin_m), float(dmax_m)],
        "height_band": height_band_label,
        "seed": seed,
        "mode": mode_norm,
    }

    if achieved < target_size:
        out["colony_note"] = "only 80% of nests"

    return out


@dataclass
class Candidate:
    candidate_id: str
    wall_id: str
    species: str
    colony_index: int            # 0,1,2...
    xyz: List                  # list of points (for colony)
    uv: List | None
    wall_score: float

    # raw references (for feature builder)
    wall_meta: Dict
    species_needs: Dict

    # populated later
    features: Dict[str, float] | None = None
    score: float | None = None
    score_breakdown: Dict[str, float] | None = None

def colony_signature(colony: Dict[str, Any], tol: float = 0.02) -> Tuple[Tuple[int, int, int], ...]:
    """
    De-duplicate colony layouts by quantized XYZ points.
    tol: 0.02 = 2cm. (0.05 can collapse too much)
    """
    pts = colony.get("xyz") or []
    q = [(round(p[0] / tol), round(p[1] / tol), round(p[2] / tol)) for p in pts]
    q.sort()
    return tuple(q)

def solitary_signature(layout: Dict[str, Any], tol: float = 0.02) -> Tuple[Tuple[int, int, int], ...]:
    """
    De-duplicate solitary nest layouts by quantized XYZ points.
    tol: 0.02 = 2cm.
    """
    pts = layout.get("xyz") or []
    q = [(round(p[0] / tol), round(p[1] / tol), round(p[2] / tol)) for p in pts]
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
    specie_colonies: int,
    *,
    attempts_per_wall: int = 50,
    low_band_m: float = 3.0,
    high_band_m: float = 10.0,
    max_candidates_per_wall: int = 20,
    base_seed: int | None = None,
    signature_tol_m: float = 0.02,
) -> List[Dict[str, Any]]:

    rng = random.Random(base_seed) if base_seed is not None else random.Random()

    if not specie_needs or len(specie_needs) != 1:
        raise ValueError("specie_needs must have exactly one species entry.")

    species_name = next(iter(specie_needs.keys()))
    needs = specie_needs[species_name] or {}

    hottest_wall_id, _ = fpf.wall_hot_climate_median(building_dict)

    wall_rows: List[str] = []

    for wall_id, wall in building_dict.items():
        # skip meta blocks like "_meta"
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue

        # HARD filter first (guaranteed no-go)
        if is_wall_unsuitable(wall):
            continue

        wall_rows.append(wall_id)

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for wall_id in wall_rows:
        wall = building_dict[wall_id]

        # belt & suspenders: hard-check again before generating
        if is_wall_unsuitable(wall):
            continue

        for _ in range(max_candidates_per_wall):
            attempt_seed = rng.randint(0, 1_000_000_000)

            colony = build_one_colony_on_wall(
                building_dict=building_dict,
                wall_id=wall_id,
                specie1_needs=needs,
                min_height_m=fpf.parse_min_height_m(needs.get("nest_height"), 0.0),
                hottest_wall_id=hottest_wall_id,
                prev_xyz_all=[],
                colonies_size=None,
                attempts_per_wall=attempts_per_wall,
                low_band_m=low_band_m,
                high_band_m=high_band_m,
                seed=attempt_seed,
                mode = "colony"
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
                "xyz": colony["xyz"],
                "uv": colony.get("uv"),
                "wall_meta": wall,
                "species_needs": needs,
                "seed": attempt_seed,
            })

    return candidates

def generate_solitary_candidates(
    building_dict: dict,
    specie_needs: dict,
    *,
    attempts_per_wall: int = 50,
    low_band_m: float = 3.0,
    high_band_m: float = 10.0,
    max_candidates_per_wall: int = 20,
    base_seed: int | None = None,
    signature_tol_m: float = 0.02,
    min_nests_per_wall: int = 1,
    max_nests_per_wall: int = 10,
    min_pair_distance_m: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Generates candidate layouts for SOLITARY species.

    Rules:
    - 1..10 nests per wall
    - every nest must be at least `min_pair_distance_m` from all others (pairwise)
    - no colonies; colony_index is always 0
    """

    rng = random.Random(base_seed) if base_seed is not None else random.Random()

    if not specie_needs or len(specie_needs) != 1:
        raise ValueError("specie_needs must have exactly one species entry.")

    species_name = next(iter(specie_needs.keys()))
    needs = specie_needs[species_name] or {}

    hottest_wall_id, _ = fpf.wall_hot_climate_median(building_dict)

    # --- collect viable walls ---
    wall_rows: List[str] = []
    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue
        if is_wall_unsuitable(wall):
            continue
        wall_rows.append(wall_id)

    candidates: List[Dict[str, Any]] = []
    seen = set()

    min_n = max(1, int(min_nests_per_wall))
    max_n = max(min_n, int(max_nests_per_wall))

    for wall_id in wall_rows:
        wall = building_dict[wall_id]
        if is_wall_unsuitable(wall):
            continue

        for _ in range(max_candidates_per_wall):
            attempt_seed = rng.randint(0, 1_000_000_000)

            # choose how many nests this candidate places
            n_nests = rng.randint(min_n, max_n)

            layout = build_one_colony_on_wall(
                building_dict=building_dict,
                wall_id=wall_id,
                specie1_needs=needs,
                min_height_m=fpf.parse_min_height_m(needs.get("nest_height"), 0.0),
                hottest_wall_id=hottest_wall_id,
                prev_xyz_all=[],
                colonies_size=n_nests,
                attempts_per_wall=attempts_per_wall,
                low_band_m=low_band_m,
                high_band_m=high_band_m,
                seed=attempt_seed,
                mode="solitary",
                solitary_min_dist_m=min_pair_distance_m,
                solitary_min_nests=min_nests_per_wall,
                solitary_max_nests=max_nests_per_wall,
            )
            if layout is None:
                continue

            pts = layout.get("xyz") or []
            if len(pts) < 1:
                continue
            if len(pts) > max_n:
                pts = pts[:max_n]

            # de-duplicate layouts
            sig = solitary_signature({"xyz": pts}, tol=signature_tol_m)
            if sig in seen:
                continue
            seen.add(sig)

            candidates.append({
                "candidate_id": str(uuid.uuid4()),
                "wall_id": wall_id,
                "species": species_name,
                "colony_index": 0,          # for Candidate compatibility
                "xyz": pts,
                "uv": layout.get("uv"),
                "wall_meta": wall,
                "species_needs": needs,
                "seed": attempt_seed,
                "n_nests": len(pts),
            })

    return candidates

#test = generate_colony_candidates(building_dict, data_swift, specie_colonies=1, max_candidates_per_wall=10, base_seed=42)
#test = generate_colony_candidates(building_dict, data_swift, specie_colonies=1)
#print("blabla candidates:", len(test))

