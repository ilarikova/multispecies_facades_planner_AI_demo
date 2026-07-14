import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI import data_extraction as de
from multispecies_facades_planner_AI.data_training_model1_test import (
    _is_wall_hard_excluded,
    _build_wall_feature_row,
    _build_sector_feature_row,
    _build_point_feature_row,
    _wall_usable_geometry,
    _get_all_sectors,
    _get_sector_feasible_points,
    _place_colony_in_sector,
    _try_place_on_wall,
    _score_rows,
    _parse_colony_size,
    _parse_height_range,
    _parse_spacing,
    _parse_num_orientations,
    _resolve_target_walls,
)

# ─────────────────────────────────────────────────────────────────
# Two species jointly compete for wall/sector space on the same
# building. Each species keeps its own wall ranking, colony/solitary
# sizing and intra-species spacing (all reused from
# data_training_model1_test.py) — the only new ingredient is the
# interspecies nest-spacing constraint pulled from the dataset field
# "distance_to_next_nest_of_other_species" (colony variant) /
# "distance_next_nest_of_other_species" (solitary variant, see
# data_extraction.COLUMN_ALIASES).
# ─────────────────────────────────────────────────────────────────

SECTOR_FAR_ENOUGH_FRACTION = 0.20


def _dist_uv(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ─────────────────────────────────────────────────────────────────
# PARSE — interspecies distance
# ─────────────────────────────────────────────────────────────────

def _parse_interspecies_distance_m(needs: dict) -> float:
    # Returns the minimum required distance to another species' nest, in metres.
    # Uses the upper bound of any parsed range (consistent with "the bigger
    # of the two distances always applies").
    raw = needs.get("distance_to_next_nest_of_other_species")
    if raw is None:
        return 0.0

    s = str(raw).strip().lower()
    if not s or s in {"nan", "none", "no"}:
        return 0.0
    if "close" in s:
        return 0.5

    try:
        lo, hi = fpf.parse_range_float(raw, default=(0.0, 0.0))
    except (ValueError, TypeError):
        return 0.0

    hi = float(hi) if hi is not None else float(lo)
    if hi > 10.0:
        hi = hi / 100.0
    return max(0.0, hi)


def _combined_interspecies_min_distance(needs1: dict, needs2: dict) -> float:
    # Between the two species' required distances, the bigger one always applies.
    return max(_parse_interspecies_distance_m(needs1), _parse_interspecies_distance_m(needs2))


# ─────────────────────────────────────────────────────────────────
# SHARED SETUP  (mirrors the preamble of data_training_model1_test.plan)
# ─────────────────────────────────────────────────────────────────

def _precompute_building_wall_features(building_dict: dict) -> None:
    fpf.precompute_wall_climate_features(building_dict)
    fpf.precompute_wall_orientations(building_dict)

    building_function = str(building_dict.get("building_function") or "").strip().lower() or None
    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue
        wall["wall_free_area_m2"] = fpf.free_wall_area(wall)
        wall["building_function"] = building_function
        try:
            nbs = fpf.neighbor_floor_functions(building_dict, wall_id)
        except Exception:
            nbs = []
        wall["neighbor_umin"] = "none"
        wall["neighbor_umax"] = "none"
        for nb in (nbs or []):
            side = (nb.get("matched_side") or "").strip().lower()
            ff = str(nb.get("floor_function") or "none").strip().lower()
            if side == "umin":
                wall["neighbor_umin"] = ff
            elif side == "umax":
                wall["neighbor_umax"] = ff


def _building_zero_z(building_dict: dict) -> float:
    all_zs = []
    for wall in building_dict.values():
        if not isinstance(wall, dict):
            continue
        for pdata in (wall.get("grid") or {}).values():
            xyz = pdata.get("point_on_wall")
            if xyz and isinstance(xyz[2], (int, float)):
                all_zs.append(float(xyz[2]))
    return float(min(all_zs)) if all_zs else 0.0


def _rank_viable_walls_for_species(
    building_dict: dict,
    needs: dict,
    model,
    model_type: str,
    xgb_encoders,
    height_min: float,
    height_max: float,
    colony_min: int,
    building_zero_z: float,
) -> List[str]:
    # Same two-step logic as STEP 1/2 of plan(): hard-filter walls, score and
    # rank them for this species, then keep only walls with >= colony_min
    # feasible grid points in at least one sector.
    eligible_walls = [
        wid for wid, wall in building_dict.items()
        if not _is_wall_hard_excluded(wid, wall)
    ]
    if not eligible_walls:
        return []

    wall_rows = [
        _build_wall_feature_row(wid, building_dict[wid], needs)
        for wid in eligible_walls
    ]
    df_walls = pd.DataFrame(wall_rows)
    scores = _score_rows(df_walls, model, model_type=model_type, xgb_encoders=xgb_encoders)
    df_walls["_score"] = scores
    df_walls = df_walls.sort_values("_score", ascending=False).reset_index(drop=True)
    ranked_wall_ids = df_walls["wall_id"].tolist()

    viable = []
    for wid in ranked_wall_ids:
        wall = building_dict[wid]
        usable_geom, _ = _wall_usable_geometry(wall)
        if usable_geom is None:
            continue
        all_sectors = _get_all_sectors(building_dict, wid)
        has_any = any(
            len(_get_sector_feasible_points(
                building_dict=building_dict, wall_id=wid,
                sector_row=sr, sector_col=sc,
                min_height_m=height_min, max_height_m=height_max,
                usable_geom=usable_geom, building_zero_z=building_zero_z,
            )) >= colony_min
            for sr, sc in all_sectors
        )
        if has_any:
            viable.append(wid)
    return viable


# ─────────────────────────────────────────────────────────────────
# SECTOR-PAIR SELECTION FOR A SHARED WALL
# ─────────────────────────────────────────────────────────────────

def _filter_points_by_other_species(points: list, other_uv_list: list, dmin_m: float) -> list:
    if dmin_m <= 0 or not other_uv_list:
        return points
    return [
        (pid, uv, xyz) for pid, uv, xyz in points
        if all(_dist_uv(uv, ouv) >= dmin_m for ouv in other_uv_list)
    ]


def _sector_far_enough(
    pts_a: list,
    pts_b: list,
    dmin_m: float,
    frac_threshold: float = SECTOR_FAR_ENOUGH_FRACTION,
) -> bool:
    # A sector pair is viable if at least frac_threshold of each sector's own
    # points are >= dmin_m from every point of the other sector.
    if dmin_m <= 0:
        return True
    if not pts_a or not pts_b:
        return False

    uvs_a = [p[1] for p in pts_a]
    uvs_b = [p[1] for p in pts_b]

    far_a = sum(1 for ua in uvs_a if all(_dist_uv(ua, ub) >= dmin_m for ub in uvs_b))
    far_b = sum(1 for ub in uvs_b if all(_dist_uv(ub, ua) >= dmin_m for ua in uvs_a))

    return (far_a / len(uvs_a)) >= frac_threshold and (far_b / len(uvs_b)) >= frac_threshold


def _feasible_scored_sectors(
    building_dict: dict,
    wall_id: str,
    wall: dict,
    usable_geom,
    all_sectors: list,
    needs: dict,
    model,
    model_type: str,
    xgb_encoders,
    height_min: float,
    height_max: float,
    colony_min: int,
    building_zero_z: float,
) -> List[Tuple[str, str, list, float]]:
    sectors = []
    for sr, sc in all_sectors:
        pts = _get_sector_feasible_points(
            building_dict=building_dict, wall_id=wall_id,
            sector_row=sr, sector_col=sc,
            min_height_m=height_min, max_height_m=height_max,
            usable_geom=usable_geom, building_zero_z=building_zero_z,
        )
        if len(pts) >= colony_min:
            sectors.append((sr, sc, pts))
    if not sectors:
        return []

    rows = [
        _build_sector_feature_row(wall_id=wall_id, wall=wall, sector_row=sr, sector_col=sc,
                                   feasible_pts=pts, needs=needs)
        for sr, sc, pts in sectors
    ]
    df = pd.DataFrame(rows)
    scores = _score_rows(df, model, model_type=model_type, xgb_encoders=xgb_encoders)
    return [(sr, sc, pts, float(s)) for (sr, sc, pts), s in zip(sectors, scores)]


def _select_sector_pair_on_shared_wall(
    building_dict: dict,
    wall_id: str,
    needs1: dict,
    needs2: dict,
    model,
    model_type: str,
    xgb_encoders,
    height_min1: float, height_max1: float,
    height_min2: float, height_max2: float,
    colony_min1: int, colony_min2: int,
    dmin_inter: float,
    building_zero_z: float,
):
    wall = building_dict[wall_id]
    usable_geom, _ = _wall_usable_geometry(wall)
    if usable_geom is None:
        return None

    all_sectors = _get_all_sectors(building_dict, wall_id)
    if not all_sectors:
        return None

    sectors_1 = _feasible_scored_sectors(
        building_dict, wall_id, wall, usable_geom, all_sectors, needs1,
        model, model_type, xgb_encoders, height_min1, height_max1, colony_min1, building_zero_z,
    )
    sectors_2 = _feasible_scored_sectors(
        building_dict, wall_id, wall, usable_geom, all_sectors, needs2,
        model, model_type, xgb_encoders, height_min2, height_max2, colony_min2, building_zero_z,
    )
    if not sectors_1 or not sectors_2:
        return None

    best_pair = None
    best_combined_score = -1e18
    for sr1, sc1, pts1, score1 in sectors_1:
        for sr2, sc2, pts2, score2 in sectors_2:
            if sr1 == sr2 and sc1 == sc2:
                continue
            if not _sector_far_enough(pts1, pts2, dmin_inter):
                continue
            combined = score1 + score2
            if combined > best_combined_score:
                best_combined_score = combined
                best_pair = ((sr1, sc1, pts1, score1), (sr2, sc2, pts2, score2))

    return best_pair


def _score_placement_result(
    building_dict, wall_id, wall, sr, sc, sec_score, placement, needs, species_name,
    model, model_type, xgb_encoders, building_zero_z,
) -> dict:
    placed_pts = list(zip(placement["selected_point_ids"], placement["uv"], placement["xyz"]))
    rows = [
        _build_point_feature_row(pid=pid, uv=uv, xyz=xyz, wall_id=wall_id, wall=wall,
                                  sector_row=sr, sector_col=sc, wall_ground_z=building_zero_z, needs=needs)
        for pid, uv, xyz in placed_pts
    ]
    df_pts = pd.DataFrame(rows)
    pt_scores = _score_rows(df_pts, model, model_type=model_type, xgb_encoders=xgb_encoders)
    placement_score = float(np.mean(pt_scores))

    wall_score = float(_score_rows(
        pd.DataFrame([_build_wall_feature_row(wall_id, wall, needs)]),
        model, model_type=model_type, xgb_encoders=xgb_encoders,
    )[0])

    return {
        "wall_id": wall_id,
        "wall_score": round(wall_score, 4),
        "section_row": sr,
        "section_col": sc,
        "sector_score": round(sec_score, 4),
        "placement_score": round(placement_score, 4),
        "xyz": placement["xyz"],
        "uv": placement["uv"],
        "colony_size": placement["colony_size"],
        "species": species_name,
        "selected_point_ids": placement["selected_point_ids"],
        "shared_wall_with": None,
    }


def _place_pair_on_shared_wall(
    building_dict: dict,
    wall_id: str,
    species1_name: str, needs1: dict,
    species2_name: str, needs2: dict,
    model, model_type: str, xgb_encoders,
    height_min1: float, height_max1: float,
    height_min2: float, height_max2: float,
    colony_min1: int, colony_max1: int, dmin1: float, dmax1: float,
    colony_min2: int, colony_max2: int, dmin2: float, dmax2: float,
    dmin_inter: float,
    building_zero_z: float,
) -> Optional[Tuple[dict, dict]]:
    pair = _select_sector_pair_on_shared_wall(
        building_dict, wall_id, needs1, needs2, model, model_type, xgb_encoders,
        height_min1, height_max1, height_min2, height_max2,
        colony_min1, colony_min2, dmin_inter, building_zero_z,
    )
    if pair is None:
        return None

    (sr1, sc1, pts1, score1), (sr2, sc2, pts2, score2) = pair
    wall = building_dict[wall_id]

    placement1 = _place_colony_in_sector(
        feasible_pts=pts1, wall_id=wall_id, wall=wall,
        sector_row=sr1, sector_col=sc1, wall_ground_z=building_zero_z,
        needs=needs1, model=model,
        colony_size_min=colony_min1, colony_size_max=colony_max1,
        dmin_m=dmin1, dmax_m=dmax1,
        model_type=model_type, xgb_encoders=xgb_encoders,
    )
    if placement1 is None:
        return None

    # hard filter: species 2's candidates must also respect the interspecies
    # distance from species 1's *actual* chosen points (the sector-level gate
    # above is only a statistical approximation).
    pts2_filtered = _filter_points_by_other_species(pts2, placement1["uv"], dmin_inter)
    if len(pts2_filtered) < colony_min2:
        pts2_filtered = pts2

    placement2 = _place_colony_in_sector(
        feasible_pts=pts2_filtered, wall_id=wall_id, wall=wall,
        sector_row=sr2, sector_col=sc2, wall_ground_z=building_zero_z,
        needs=needs2, model=model,
        colony_size_min=colony_min2, colony_size_max=colony_max2,
        dmin_m=dmin2, dmax_m=dmax2,
        model_type=model_type, xgb_encoders=xgb_encoders,
    )
    if placement2 is None:
        return None

    result1 = _score_placement_result(
        building_dict, wall_id, wall, sr1, sc1, score1, placement1, needs1, species1_name,
        model, model_type, xgb_encoders, building_zero_z,
    )
    result2 = _score_placement_result(
        building_dict, wall_id, wall, sr2, sc2, score2, placement2, needs2, species2_name,
        model, model_type, xgb_encoders, building_zero_z,
    )
    result1["shared_wall_with"] = species2_name
    result2["shared_wall_with"] = species1_name
    return result1, result2


def _place_one_species_avoiding_other(
    building_dict: dict,
    wall_id: str,
    needs: dict,
    model, model_type: str, xgb_encoders,
    height_min: float, height_max: float,
    colony_min: int, colony_max: int, dmin_m: float, dmax_m: float,
    avoid_uv_list: list, dmin_inter: float,
    building_zero_z: float,
    species_name: str,
) -> Optional[dict]:
    # Used when a wall is already occupied by the other species (from an
    # earlier round of independent assignment) — places this species on the
    # same wall, keeping every candidate sector's points far enough from the
    # other species' existing points.
    wall = building_dict[wall_id]
    usable_geom, _ = _wall_usable_geometry(wall)
    if usable_geom is None:
        return None
    all_sectors = _get_all_sectors(building_dict, wall_id)
    if not all_sectors:
        return None

    feasible_sectors = []
    for sr, sc in all_sectors:
        pts = _get_sector_feasible_points(
            building_dict=building_dict, wall_id=wall_id, sector_row=sr, sector_col=sc,
            min_height_m=height_min, max_height_m=height_max,
            usable_geom=usable_geom, building_zero_z=building_zero_z,
        )
        if not pts:
            continue
        far_pts = _filter_points_by_other_species(pts, avoid_uv_list, dmin_inter)
        if avoid_uv_list and dmin_inter > 0 and (len(far_pts) / len(pts)) < SECTOR_FAR_ENOUGH_FRACTION:
            continue
        if len(far_pts) >= colony_min:
            feasible_sectors.append((sr, sc, far_pts))

    if not feasible_sectors:
        return None

    sector_rows = [
        _build_sector_feature_row(wall_id=wall_id, wall=wall, sector_row=sr, sector_col=sc,
                                   feasible_pts=pts, needs=needs)
        for sr, sc, pts in feasible_sectors
    ]
    df_sectors = pd.DataFrame(sector_rows)
    scores = _score_rows(df_sectors, model, model_type=model_type, xgb_encoders=xgb_encoders)
    order = list(np.argsort(scores)[::-1])

    for idx in order:
        sr, sc, pts = feasible_sectors[idx]
        placement = _place_colony_in_sector(
            feasible_pts=pts, wall_id=wall_id, wall=wall, sector_row=sr, sector_col=sc,
            wall_ground_z=building_zero_z, needs=needs, model=model,
            colony_size_min=colony_min, colony_size_max=colony_max,
            dmin_m=dmin_m, dmax_m=dmax_m, model_type=model_type, xgb_encoders=xgb_encoders,
        )
        if placement is None:
            continue
        return _score_placement_result(
            building_dict, wall_id, wall, sr, sc, float(scores[idx]), placement, needs, species_name,
            model, model_type, xgb_encoders, building_zero_z,
        )

    return None


def _merge_solitary_boxes(result: dict) -> dict:
    # Mirrors the merge step in data_training_model1_test.plan for multi-box
    # solitary species placed via _try_place_on_wall.
    if result.get("solitary_boxes") and len(result["solitary_boxes"]) > 1:
        all_boxes = result["solitary_boxes"]
        result["xyz"] = [b["xyz"][0] for b in all_boxes]
        result["uv"] = [b["uv"][0] for b in all_boxes]
        result["selected_point_ids"] = [b["selected_point_ids"][0] for b in all_boxes]
        result["colony_size"] = len(all_boxes)
    result.setdefault("shared_wall_with", None)
    return result


# ─────────────────────────────────────────────────────────────────
# MAIN TWO-SPECIES PLANNER
# ─────────────────────────────────────────────────────────────────

def plan_species_combination(
    model,
    building_dict: dict,
    species1_name: str,
    needs1: dict,
    species2_name: str,
    needs2: dict,
    model_type: str = "lgbm",
    xgb_encoders: dict | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Plans nest placements for two species on the same building at once.

    Each species keeps its own wall ranking, colony/solitary sizing and
    intra-species spacing (all reused from data_training_model1_test.py).
    Whenever both species want the same wall, the wall is split into two
    sectors — one per species — chosen so they are as far apart as the
    interspecies distance requires (the bigger of the two species' values
    always applies), while both sectors stay as high-scoring as possible.

    Returns
    ───────
    {species1_name: [placement, ...], species2_name: [placement, ...]}
    Each placement dict matches the shape produced by plan()'s options
    (wall_id, wall_score, section_row, section_col, sector_score,
    placement_score, xyz, uv, colony_size, species, selected_point_ids),
    plus "shared_wall_with" (the other species' name, or None).
    """
    _precompute_building_wall_features(building_dict)

    colony_min1, colony_max1 = _parse_colony_size(needs1)
    colony_min2, colony_max2 = _parse_colony_size(needs2)
    height_min1, height_max1 = _parse_height_range(needs1)
    height_min2, height_max2 = _parse_height_range(needs2)
    dmin1, dmax1 = _parse_spacing(needs1)
    dmin2, dmax2 = _parse_spacing(needs2)
    dmin_inter = _combined_interspecies_min_distance(needs1, needs2)

    building_zero_z = _building_zero_z(building_dict)

    walls1 = _rank_viable_walls_for_species(
        building_dict, needs1, model, model_type, xgb_encoders,
        height_min1, height_max1, colony_min1, building_zero_z,
    )
    walls2 = _rank_viable_walls_for_species(
        building_dict, needs2, model, model_type, xgb_encoders,
        height_min2, height_max2, colony_min2, building_zero_z,
    )
    if not walls1 or not walls2:
        raise ValueError("Planning failed: no viable walls found for one or both species.")

    target_walls1 = _resolve_target_walls(_parse_num_orientations(needs1), len(walls1))
    target_walls2 = _resolve_target_walls(_parse_num_orientations(needs2), len(walls2))

    results1: List[dict] = []
    results2: List[dict] = []
    used_walls1: set = set()
    used_walls2: set = set()
    idx1 = 0
    idx2 = 0

    def _next_wall(walls, used, start_idx):
        i = start_idx
        while i < len(walls):
            if walls[i] not in used:
                return i
            i += 1
        return None

    while len(results1) < target_walls1 or len(results2) < target_walls2:
        need1 = len(results1) < target_walls1
        need2 = len(results2) < target_walls2

        i1 = _next_wall(walls1, used_walls1, idx1) if need1 else None
        i2 = _next_wall(walls2, used_walls2, idx2) if need2 else None

        if i1 is None and i2 is None:
            break

        wall1 = walls1[i1] if i1 is not None else None
        wall2 = walls2[i2] if i2 is not None else None

        # both species independently want the SAME wall this round -> split it
        if wall1 is not None and wall1 == wall2:
            pair = _place_pair_on_shared_wall(
                building_dict, wall1,
                species1_name, needs1, species2_name, needs2,
                model, model_type, xgb_encoders,
                height_min1, height_max1, height_min2, height_max2,
                colony_min1, colony_max1, dmin1, dmax1,
                colony_min2, colony_max2, dmin2, dmax2,
                dmin_inter, building_zero_z,
            )
            idx1 = i1 + 1
            idx2 = i2 + 1
            if pair is not None:
                result1, result2 = pair
                results1.append(result1)
                results2.append(result2)
                used_walls1.add(wall1)
                used_walls2.add(wall1)
            continue

        progressed = False

        if wall1 is not None:
            if wall1 in used_walls2:
                other = next(r for r in results2 if r["wall_id"] == wall1)
                placement1 = _place_one_species_avoiding_other(
                    building_dict, wall1, needs1, model, model_type, xgb_encoders,
                    height_min1, height_max1, colony_min1, colony_max1, dmin1, dmax1,
                    avoid_uv_list=other["uv"], dmin_inter=dmin_inter,
                    building_zero_z=building_zero_z, species_name=species1_name,
                )
                if placement1 is not None:
                    placement1["shared_wall_with"] = species2_name
                    other["shared_wall_with"] = species1_name
            else:
                placement1 = _try_place_on_wall(
                    building_dict=building_dict, wall_id=wall1, needs=needs1, model=model,
                    height_min=height_min1, height_max=height_max1, building_zero_z=building_zero_z,
                    colony_min=colony_min1, colony_max=colony_max1, dmin_m=dmin1, dmax_m=dmax1,
                    model_type=model_type, xgb_encoders=xgb_encoders, species_name=species1_name,
                    sector_rank=0,
                )
                if placement1 is not None:
                    placement1 = _merge_solitary_boxes(placement1)
            idx1 = i1 + 1
            progressed = True
            if placement1 is not None:
                results1.append(placement1)
                used_walls1.add(wall1)

        if wall2 is not None:
            if wall2 in used_walls1:
                other = next(r for r in results1 if r["wall_id"] == wall2)
                placement2 = _place_one_species_avoiding_other(
                    building_dict, wall2, needs2, model, model_type, xgb_encoders,
                    height_min2, height_max2, colony_min2, colony_max2, dmin2, dmax2,
                    avoid_uv_list=other["uv"], dmin_inter=dmin_inter,
                    building_zero_z=building_zero_z, species_name=species2_name,
                )
                if placement2 is not None:
                    placement2["shared_wall_with"] = species1_name
                    other["shared_wall_with"] = species2_name
            else:
                placement2 = _try_place_on_wall(
                    building_dict=building_dict, wall_id=wall2, needs=needs2, model=model,
                    height_min=height_min2, height_max=height_max2, building_zero_z=building_zero_z,
                    colony_min=colony_min2, colony_max=colony_max2, dmin_m=dmin2, dmax_m=dmax2,
                    model_type=model_type, xgb_encoders=xgb_encoders, species_name=species2_name,
                    sector_rank=0,
                )
                if placement2 is not None:
                    placement2 = _merge_solitary_boxes(placement2)
            idx2 = i2 + 1
            progressed = True
            if placement2 is not None:
                results2.append(placement2)
                used_walls2.add(wall2)

        if not progressed:
            break

    return {
        species1_name: results1,
        species2_name: results2,
    }


# ─────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multispecies_facades_planner_AI import facade_planner_visAI as fpvAI

    excel_path = r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\Datasets\bird_species.xlsx"

    DATA_ROOT = Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\tests\data_training")
    MODEL_DIR = DATA_ROOT / "models"

    BUILDING_JSON = Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building4868.json")

    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SPECIES_1 = "black_redstart"
    SPECIES_2 = "house_martin"

    MODEL_TYPE = "xgb"
    MODEL_VARIANT = "reduced"  # "full" or "reduced"

    model_dir_resolved = MODEL_DIR / "reduced" if MODEL_VARIANT == "reduced" else MODEL_DIR
    suffix = "_reduced" if MODEL_VARIANT == "reduced" else ""

    model        = joblib.load(model_dir_resolved / f"nestworks_xgb_ranker{suffix}.pkl")
    xgb_encoders = joblib.load(model_dir_resolved / f"nestworks_xgb_encoders{suffix}.pkl")

    building_dict = fpf.load_building_dict(str(BUILDING_JSON))

    needs1 = de.load_species_training_as_dict(excel_path, SPECIES_1)[SPECIES_1]
    needs2 = de.load_species_training_as_dict(excel_path, SPECIES_2)[SPECIES_2]

    combination_result = plan_species_combination(
        model=model,
        building_dict=building_dict,
        species1_name=SPECIES_1,
        needs1=needs1,
        species2_name=SPECIES_2,
        needs2=needs2,
        model_type=MODEL_TYPE,
        xgb_encoders=xgb_encoders,
    )

    print(f"{SPECIES_1}: {len(combination_result[SPECIES_1])} placement(s)")
    print(f"{SPECIES_2}: {len(combination_result[SPECIES_2])} placement(s)")

    wall_label_pos = fpvAI.compute_wall_label_positions(building_dict)

    pdf_path = RESULTS_DIR / f"{SPECIES_1}_vs_{SPECIES_2}_{BUILDING_JSON.stem}_{MODEL_TYPE}{suffix}.pdf"

    fpvAI.export_species_combination_pdf(
        building_dict=building_dict,
        combination_result=combination_result,
        species1_name=SPECIES_1,
        species2_name=SPECIES_2,
        wall_label_pos=wall_label_pos,
        pdf_path=str(pdf_path),
    )

    print(f"PDF saved: {pdf_path}")
