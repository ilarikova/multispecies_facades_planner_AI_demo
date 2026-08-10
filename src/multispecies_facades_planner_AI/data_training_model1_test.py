import math
import uuid
import random
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point

from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI import facade_planner_species as fps
from multispecies_facades_planner_AI import data_extraction as de

# ─────────────────────────────────────────────────────────────────
# CONSTANTS  — mirror training pipeline exactly
# ─────────────────────────────────────────────────────────────────
excel_path = r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\Datasets\bird_species.xlsx"



ALL_ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Upper bound on nest height at planning time.
#
# The training generator (facade_planner.base_candidates_for_wall) gates on
# min_height_m ONLY — it never dropped points above the species' stated maximum,
# so the exported candidates and the expert labels both span the full wall.
# Enforcing the cap here made the planner stricter than the data it learned from:
# for black_redstart (nest_height "2-6") it removed 53% of the training rows and
# 55% of the expert's own BEST picks; for common_pipistrelle ("...-9") it removed
# 67% of them. The expert's best-pick rate was effectively flat across the cap
# (5.2% below vs 5.6% above), i.e. they did not treat higher placements as worse.
#
# Set to False (2026-08-04) so inference matches the generator exactly. Flip it
# back to True to reinstate the cap — nothing else needs to change.
ENFORCE_MAX_HEIGHT = False

# Minimum usable facade area for a wall to be considered at all, applied in
# _is_wall_hard_excluded alongside the floor_function exclusion.
#
# Raised from 2.0 to 2.5 on 2026-08-04. The 2.0 threshold let 5128-wall-00
# through at 2.23 m2 — a 1.56 m tall parapet-sized element that is not a real
# nesting facade.
MIN_WALL_FREE_AREA_M2 = 2.5
 
# MUST stay identical to data_training_model1.ALL_FEATURES — a feature listed
# there but not built here silently arrives as NaN at inference.
ALL_FEATURES = [
    "wall_free_area_m2", "wall_height_m", "door_count", "wall_shape",
    "orientation", "floor_function", "neighbor_umin", "neighbor_umax",
    "building_function", "wall_climate_median",
    "section_row", "section_col",
    "mean_height_m", "min_height_m", "max_height_m", "height_std_m",
    "colony_nests_placed",
    "dist_to_top_edge_median", "dist_to_side_edge_median",
    "side_edge_label", "orientation_match", "sector_climate_median",
    "distance_to_window", "distance_to_window_median",
    "colonial", "colony_size_local_min", "colony_size_local_max",
    "nest_distance_min_m", "nest_distance_max_m",
    "noise_level", "human_tolerance_level", "dirt", "taxa",
    "is_bird", "is_bat",
    "is_day_active", "is_evening_active", "is_dusk_active", "is_night_active",
    "nest_use_start_month", "nest_use_end_month", "nest_use_duration",
    "preferred_height_min_m", "preferred_height_max_m",
    "prefers_edges_proximity", "prefers_roof_proximity",
    "far_from_windows_important",
    "nest_temp_min_c", "nest_temp_max_c",
]
 
CATEGORICAL_COLS = [
    "orientation", "floor_function", "neighbor_umin", "neighbor_umax",
    "building_function", "section_row", "section_col",
    "side_edge_label", "wall_shape", "taxa",
]
 
 
# ─────────────────────────────────────────────────────────────────
# SHARED FEATURE BUILDERS — must mirror the training generator exactly
# ─────────────────────────────────────────────────────────────────

_TRAITS_CACHE: Dict[int, Tuple[dict, dict]] = {}


def _species_fields(needs: dict) -> dict:
    # All species-trait features, encoded exactly as the training generator
    # encodes them (fps.encode_species_traits, verified against export3107).
    #
    # `needs` is the RAW species-table row: its keys are sheet column names
    # ("species_noise", "time_activity", "temperature_optimum_in_nest_box"),
    # not feature names. This module used to read needs.get("noise_level"),
    # needs.get("is_day_active"), needs.get("nest_temp_min_c") and so on
    # directly — every one of those returned None, leaving 17 of the model's
    # features dead at inference while training saw real values.
    #
    # Cached per `needs` object: the row builders are called once per candidate
    # point, and the encoding is pure string/regex parsing of the same dict.
    # The cache holds a reference to `needs` alongside its traits, so the dict
    # cannot be collected and have its id reused by another species' dict —
    # which would otherwise serve one species the traits of another.
    key = id(needs)
    cached = _TRAITS_CACHE.get(key)
    if cached is not None and cached[0] is needs:
        return cached[1]

    traits = fps.encode_species_traits(needs)
    _TRAITS_CACHE[key] = (needs, traits)
    return traits


def _orientation_fields(wall: dict, needs: dict) -> dict:
    # Wall orientation and its match against the species preference.
    #
    # needs[...] holds the RAW species-table text ("north, east"). The training
    # CSV stores the normalised set ("E,N") and derives orientation_match with
    # fps.orientation_match. Splitting the raw text on commas instead — as this
    # module did before — gives ["NORTH", "EAST"], which can never equal a
    # compass label, so orientation_match was pinned to 0 on exactly the walls
    # the species prefers, and all 16 orientation one-hot flags built by
    # _engineer_features were pinned to 0.
    ori = (wall.get("orientation") or "").strip().upper() or None

    return {
        "orientation":       ori,
        "orientation_match": fps.orientation_match(ori, needs.get("preferred_orientation")),
    }


def _wall_context_fields(wall: dict) -> dict:
    # Wall-level features. wall_free_area_m2 / building_function /
    # neighbor_umin / neighbor_umax are stamped onto each wall by plan();
    # door_count and wall_height_m are not stored on the wall at all, so they
    # are derived the same way the generator derives them.
    return {
        "wall_free_area_m2":   wall.get("wall_free_area_m2"),
        "wall_height_m":       fpf.wall_height_m(wall),
        "wall_shape":          wall.get("wall_shape"),
        "door_count":          len(wall.get("doors") or {}),
        "floor_function":      wall.get("floor_function"),
        "neighbor_umin":       wall.get("neighbor_umin"),
        "neighbor_umax":       wall.get("neighbor_umax"),
        "building_function":   wall.get("building_function"),
        "wall_climate_median": wall.get("wall_climate_median"),
    }


def _candidate_fields(wall_id: str, wall: dict, uvs: list, xyzs: list,
                      needs: dict | None = None) -> dict:
    # Edge and sector-climate features via the same fpf helpers the training
    # generator uses, so inference and training agree on their definitions.
    #
    # Those helpers only ever look up building_dict[wall_id], so a one-wall dict
    # is enough. They are also v_up-aware (see
    # docs/roof_band_axis_bug_and_repair.md). What stood here before was not:
    # dist_to_top_edge_median was approximated as "wall's highest grid Z minus
    # this point's Z" instead of the perpendicular distance to the roofline, and
    # dist_to_side_edge_median / side_edge_label / sector_climate_median were
    # read from wall-level keys that do not exist, so they were always None.
    EMPTY = {
        "dist_to_top_edge_median":   None,
        "dist_to_side_edge_median":  None,
        "side_edge_label":           None,
        "distance_to_window":        None,
        "distance_to_window_median": None,
        "sector_climate_median":     None,
        "mean_height_m":             None,
        "min_height_m":              None,
        "max_height_m":              None,
        "height_std_m":              None,
        "colony_nests_placed":       np.nan,
    }
    if not uvs or not xyzs:
        return dict(EMPTY)

    bd = {wall_id: wall}
    candidate = {
        "wall_id": wall_id,
        "uv":  [[float(p[0]), float(p[1])] for p in uvs],
        "xyz": [list(p) for p in xyzs],
    }
    ef = fpf.candidate_edge_features(bd, candidate)
    hs = fpf.candidate_local_height_stats(bd, candidate)

    # Colony size is NaN for solitary species in training, so mirror that
    # rather than writing 1 — the model saw NaN for every solitary row.
    colonial = _species_fields(needs).get("colonial") if needs is not None else None
    colony_nests_placed = len(candidate["xyz"]) if colonial == 1 else np.nan

    return {
        "dist_to_top_edge_median":   ef["dist_to_top_edge_median"],
        "dist_to_side_edge_median":  ef["dist_to_side_edge_median"],
        "side_edge_label":           ef["side_edge_label"],
        "distance_to_window":        ef["distance_to_window"],
        "distance_to_window_median": ef["distance_to_window_median"],
        "sector_climate_median":     fpf.candidate_sector_climate_median_3x3(bd, candidate),
        "mean_height_m":             hs["mean_height_m"],
        "min_height_m":              hs["min_height_m"],
        "max_height_m":              hs["max_height_m"],
        "height_std_m":              hs["height_std_m"],
        "colony_nests_placed":       colony_nests_placed,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 1 — HARD CONSTRAINT WALL FILTER
# ─────────────────────────────────────────────────────────────────
 
def _is_wall_hard_excluded(wall_id: str, wall: dict) -> bool:
    # Returns True if the wall must be excluded before any AI scoring.
    # Only floor_function is checked here — it is a building-level structural
    # exclusion independent of species. Orientation and area constraints are
    # applied later at sector and placement level where species params are available.
    if not isinstance(wall, dict):
        return True
 
    if wall_id.startswith("_"):
        return True
 
    floor_fn = (wall.get("floor_function") or "").strip().lower()
    if floor_fn in {"neighbor_building", "neigbor_building"}:
        return True
 
    # exclude walls with negligible free area — too small to place anything.
    # Computed on the spot when plan() has not stamped it onto the wall yet,
    # so the check can never silently no-op.
    free_area = wall.get("wall_free_area_m2")
    if free_area is None:
        try:
            free_area = fpf.free_wall_area(wall)
        except Exception:
            free_area = None

    if free_area is not None:
        try:
            if float(free_area) < MIN_WALL_FREE_AREA_M2:
                return True
        except (ValueError, TypeError):
            pass

    return False
 
 
# ─────────────────────────────────────────────────────────────────
# STEP 2 — WALL-LEVEL FEATURE ROWS + AI SCORING
# ─────────────────────────────────────────────────────────────────
 
def _build_wall_feature_row(wall_id: str, wall: dict, needs: dict) -> dict:
    # Builds a single feature row representing this wall (no placement yet).
    # Uses wall-level fields only; sector and candidate fields left as NaN.
    row = {
        "wall_id": wall_id,
        **_wall_context_fields(wall),
        **_orientation_fields(wall, needs),
        **_candidate_fields(wall_id, wall, [], [], needs),
        "section_row":          None,
        "section_col":          None,
        **_species_fields(needs),
    }
    return row


# ─────────────────────────────────────────────────────────────────
# STEP 3 — SECTOR FEASIBILITY CHECK
# ─────────────────────────────────────────────────────────────────
 
def _get_sector_feasible_points(
    building_dict: dict,
    wall_id: str,
    sector_row: str,
    sector_col: str,
    min_height_m: float,
    max_height_m: float,
    usable_geom,
    needs: dict | None = None,
) -> list:
    # Returns grid points in this sector that pass the same hard constraints the
    # training generator applied in facade_planner.base_candidates_for_wall:
    #
    #   1. inside the usable area (openings minus the window/door offset band)
    #   2. rel_h >= min_height_m, measured from THIS WALL's lowest grid Z
    #   3. if needs["distance_to_roof"] == "strict", within fpf.ROOF_STRICT_BAND_M
    #      (1.5 m) of the nearest roof edge — v_up-aware, so it follows sloped and
    #      gable roof lines rather than the bottom of a flipped wall
    #
    # Swift and house martin are the "strict" species: swift is `> 6` m and
    # strict, house martin `> 2` m and strict. black_redstart and house_sparrow
    # are "close", so the 1.5 m band does not apply to them.
    #
    # Height reference: the generator measures rel_h from the wall's own lowest
    # grid Z, and candidate_local_height_stats (which produces mean/min/max_height_m)
    # does the same. This used to take building_zero_z — the lowest Z across the
    # WHOLE building — which admitted points the generator would never have
    # emitted as candidates on any wall whose base sits above the building low
    # point (e.g. 5128-wall-00, where it let 24 swift points through against the
    # generator's 0).
    #
    # sector_row/col are computed from UV via fpf.sector_3x3_labels — NOT read
    # from stored grid fields (they are not stored there).
    wall = building_dict[wall_id]
    grid = wall.get("grid") or {}
    bbox = fpf.wall_uv_bbox_from_building(building_dict, wall_id)
    if bbox is None:
        return []

    roof_strict = fpf.is_distance_to_roof_strict((needs or {}).get("distance_to_roof"))
    boundary_uv = wall.get("boundary_uv") or []
    v_up = fpf.v_axis_points_up(wall)

    zs_all = [
        float(pdata["point_on_wall"][2])
        for pdata in grid.values()
        if pdata.get("point_on_wall")
        and isinstance(pdata["point_on_wall"][2], (int, float))
    ]
    if not zs_all:
        return []
    wall_ground_z = min(zs_all)

    pts = []

    for pid, pdata in grid.items():
        uv  = pdata.get("uv")
        xyz = pdata.get("point_on_wall")
        if not uv or not xyz:
            continue

        row_label, col_label = fpf.sector_3x3_labels(
            float(uv[0]), float(uv[1]), bbox, v_up=v_up
        )
        if row_label != sector_row or col_label != sector_col:
            continue

        z = float(xyz[2])
        rel_h = z - wall_ground_z

        if rel_h < min_height_m:
            continue
        if ENFORCE_MAX_HEIGHT and max_height_m > 0 and rel_h > max_height_m:
            continue

        p = Point(float(uv[0]), float(uv[1]))
        if not usable_geom.contains(p):
            continue

        if roof_strict and not fpf.is_within_roof_strict_band(
            boundary_uv, float(uv[0]), float(uv[1]), v_up=v_up
        ):
            continue

        pts.append((pid, [float(uv[0]), float(uv[1])], [float(xyz[0]), float(xyz[1]), z]))

    return pts
 
 
def _wall_usable_geometry(wall: dict, needs: dict | None = None):
    # Computes the usable area polygon for this wall (openings removed, offset applied).
    # Returns usable_geom and wall_ground_z, or (None, None) if wall is unusable.
    #
    # `needs` selects the window-offset rules: colony species get the wider
    # between-windows band and the top-floor window-width strip, matching
    # facade_planner.base_candidates_for_wall. Omitting it keeps the flat 0.35 m.
    colonial = _species_fields(needs).get("colonial") == 1 if needs else False

    free_geom   = fpf.derive_openings(wall)
    offset_geom = fpf.build_offset_area(wall, colonial=colonial)
    usable_geom = free_geom.difference(offset_geom)
 
    if usable_geom.is_empty:
        return None, None
 
    grid = wall.get("grid") or {}
    zs = [
        float(pdata["point_on_wall"][2])
        for pdata in grid.values()
        if pdata.get("point_on_wall") and isinstance(pdata["point_on_wall"][2], (int, float))
    ]
    if not zs:
        return None, None
 
    return usable_geom, min(zs)
 
 
def _get_all_sectors(building_dict: dict, wall_id: str) -> list:
    # Returns the unique (section_row, section_col) pairs for this wall,
    # computed from grid point UV coordinates via fpf.sector_3x3_labels.
    wall = building_dict[wall_id]
    grid = wall.get("grid") or {}
    bbox = fpf.wall_uv_bbox_from_building(building_dict, wall_id)
    if bbox is None:
        return []
 
    v_up = fpf.v_axis_points_up(wall)

    seen = set()
    for pdata in grid.values():
        uv = pdata.get("uv")
        if not uv or len(uv) < 2:
            continue
        row_label, col_label = fpf.sector_3x3_labels(
            float(uv[0]), float(uv[1]), bbox, v_up=v_up
        )
        seen.add((row_label, col_label))
    return list(seen)
 
 
# ─────────────────────────────────────────────────────────────────
# STEP 3 — SECTOR-LEVEL FEATURE ROWS + AI SCORING
# ─────────────────────────────────────────────────────────────────
 
def _build_sector_feature_row(
    wall_id: str,
    wall: dict,
    sector_row: str,
    sector_col: str,
    feasible_pts: list,
    needs: dict,
) -> dict:
    # Builds a feature row for one sector using the feasible points in that sector.
    row = {
        "wall_id":    wall_id,
        "sector_row": sector_row,
        "sector_col": sector_col,
        **_wall_context_fields(wall),
        **_orientation_fields(wall, needs),
        **_candidate_fields(
            wall_id, wall,
            [pt[1] for pt in feasible_pts],
            [pt[2] for pt in feasible_pts],
            needs,
        ),
        "section_row":          sector_row,
        "section_col":          sector_col,
        **_species_fields(needs),
    }
    return row


# ─────────────────────────────────────────────────────────────────
# STEP 4 — PLACEMENT WITHIN SECTOR (AI-guided, hard constraints enforced)
# ─────────────────────────────────────────────────────────────────
 
def _build_point_feature_row(
    pid: str,
    uv: list,
    xyz: list,
    wall_id: str,
    wall: dict,
    sector_row: str,
    sector_col: str,
    wall_ground_z: float,
    needs: dict,
) -> dict:
    # Builds a feature row for one individual grid point (placement candidate).
    # wall_ground_z is kept in the signature for call-site compatibility; the
    # height stats now come from fpf.candidate_local_height_stats, which uses
    # the wall's own lowest grid Z — the same reference the generator used.
    row = {
        "point_id":   pid,
        "wall_id":    wall_id,
        **_wall_context_fields(wall),
        **_orientation_fields(wall, needs),
        **_candidate_fields(wall_id, wall, [uv], [xyz], needs),
        "section_row":          sector_row,
        "section_col":          sector_col,
        **_species_fields(needs),
    }
    return row


def _pick_points_cluster(
    candidates: list,
    target_size: int,
    dmin_m: float,
    dmax_m: float,
) -> list:
    # Cluster growth placement — replaces pick_points_chained_band for the planner.
    # The old chain function required each new point to be within dmax of the LAST
    # placed point only, which caused short colonies with tight dmax constraints.
    # This function requires each new point to be:
    #   >= dmin_m from ALL already placed points  (spacing floor)
    #   <= dmax_m from AT LEAST ONE placed point  (cluster reachability)
    # candidates: list of (pid, uv, xyz) — first element is the anchor.
    # Returns list of selected (pid, uv, xyz).
    if not candidates:
        return []
 
    def _dist2(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
 
    selected = [candidates[0]]
    placed_uvs = [candidates[0][1]]
 
    for cand in candidates[1:]:
        if len(selected) >= target_size:
            break
        c_uv = cand[1]
        # reject if too close to any placed point
        if any(_dist2(c_uv, p_uv) < dmin_m for p_uv in placed_uvs):
            continue
        # reject if not reachable from any placed point
        if dmax_m < 1e8:
            if not any(_dist2(c_uv, p_uv) <= dmax_m for p_uv in placed_uvs):
                continue
        selected.append(cand)
        placed_uvs.append(c_uv)
 
    return selected
 
 
def _place_colony_in_sector(
    feasible_pts: list,
    wall_id: str,
    wall: dict,
    sector_row: str,
    sector_col: str,
    wall_ground_z: float,
    needs: dict,
    model,
    colony_size_min: int,
    colony_size_max: int,
    dmin_m: float,
    dmax_m: float,
    model_type: str = "lgbm",
    xgb_encoders: dict | None = None,
    n_attempts: int = 200,
    seed: int | None = 42,
    exclude_point_ids: set | None = None,
) -> Optional[Dict[str, Any]]:
    # Hybrid placement — two passes:
    # Pass 1: weighted-random anchor (AI scores as sampling weights), 200 attempts.
    # Pass 2 (fallback): connectivity-filtered anchors — only anchors with >=
    #   colony_size_min valid neighbours in [dmin_m, dmax_m] are eligible.
    #   Triggered when Pass 1 fails OR produces only duplicates of exclude_point_ids.
    # exclude_point_ids: set of point ids already used in a previous option —
    #   used to detect duplicates and trigger Pass 2.
 
    if not feasible_pts:
        return None
 
    def _dist2(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
 
    # score all feasible points
    rows = [
        _build_point_feature_row(
            pid=pid, uv=uv, xyz=xyz,
            wall_id=wall_id, wall=wall,
            sector_row=sector_row, sector_col=sector_col,
            wall_ground_z=wall_ground_z,
            needs=needs,
        )
        for pid, uv, xyz in feasible_pts
    ]
    df_pts = pd.DataFrame(rows)
    scores = _score_rows(df_pts, model, model_type=model_type, xgb_encoders=xgb_encoders)
 
    s = np.array(scores, dtype=float)
    s = s - s.max()
    weights = np.exp(s)
    weights = weights / weights.sum()
 
    pt_lookup = {pid: (uv, xyz) for pid, uv, xyz in feasible_pts}
    pid_list  = [pid for pid, _, _ in feasible_pts]
 
    rng = random.Random(seed)
 
    def _run_attempts(eligible_indices, n):
        # run n placement attempts using anchors from eligible_indices
        best_sel   = None
        best_score = -1e9
        elig_weights = np.array([weights[i] for i in eligible_indices])
        elig_weights = elig_weights / elig_weights.sum()
 
        for _ in range(n):
            anchor_local = rng.choices(range(len(eligible_indices)),
                                       weights=elig_weights.tolist(), k=1)[0]
            anchor_idx   = eligible_indices[anchor_local]
            anchor_pid   = pid_list[anchor_idx]
 
            remaining  = [pt for pt in feasible_pts if pt[0] != anchor_pid]
            rng.shuffle(remaining)
            candidates = [feasible_pts[anchor_idx]] + remaining
 
            selected = _pick_points_cluster(
                candidates=candidates,
                target_size=colony_size_max,
                dmin_m=dmin_m,
                dmax_m=dmax_m,
            )
 
            if len(selected) < colony_size_min:
                continue
            if len(selected) > colony_size_max:
                selected = selected[:colony_size_max]
 
            sel_indices = [pid_list.index(pt[0]) for pt in selected if pt[0] in pid_list]
            mean_score  = float(np.mean(scores[sel_indices])) if sel_indices else -1e9
 
            if best_sel is None:
                best_sel   = selected
                best_score = mean_score
            elif len(selected) > len(best_sel):
                best_sel   = selected
                best_score = mean_score
            elif len(selected) == len(best_sel) and mean_score > best_score:
                best_sel   = selected
                best_score = mean_score
 
        return best_sel
 
    def _is_duplicate(selected):
        if not exclude_point_ids or selected is None:
            return False
        sel_ids = set(pt[0] for pt in selected)
        return sel_ids == exclude_point_ids
 
    # ── Pass 1: all anchors, weighted random
    all_indices = list(range(len(pid_list)))
    best_selected = _run_attempts(all_indices, n_attempts)
 
    # ── Pass 2: pure greedy score-ordered scan — no anchor, no randomness
    # triggered if Pass 1 failed or produced a duplicate
    if best_selected is None or _is_duplicate(best_selected):
        # pure geometric scan — no score sorting, just find ANY valid placement
        greedy_selected = []
        greedy_uvs = []
 
        for pt in feasible_pts:
            if len(greedy_selected) >= colony_size_max:
                break
            c_uv = pt[1]
            # must be >= dmin from ALL placed points
            if any(_dist2(c_uv, p_uv) < dmin_m for p_uv in greedy_uvs):
                continue
            # must be <= dmax from AT LEAST ONE placed point (skip for first point)
            if dmax_m < 1e8 and greedy_uvs:
                if not any(_dist2(c_uv, p_uv) <= dmax_m for p_uv in greedy_uvs):
                    continue
            greedy_selected.append(pt)
            greedy_uvs.append(c_uv)
 
        # accept if greedy reaches colony_size_min, or colony_size_min - 1 as last resort
        # tolerance only applies when colony_size_min > 2 — for small colonies (2-3) no relaxation
        greedy_threshold = (max(1, colony_size_min - 1) if colony_size_min > 2 else colony_size_min)
        if len(greedy_selected) >= greedy_threshold:
            pass2_result = greedy_selected
            if not _is_duplicate(pass2_result):
                best_selected = pass2_result
            elif best_selected is None:
                best_selected = pass2_result
 
    if best_selected is None:
        return None
 
    sel_ids = [pt[0] for pt in best_selected]
    sel_uvs = [pt[1] for pt in best_selected]
    sel_xyz = [pt[2] for pt in best_selected]
 
    return {
        "selected_point_ids": sel_ids,
        "xyz":                sel_xyz,
        "uv":                 sel_uvs,
        "colony_size":        len(sel_ids),
        "colony_size_min":    colony_size_min,
        "colony_size_max":    colony_size_max,
    }
 
 
# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING + SCORING  (shared by all three passes)
# ─────────────────────────────────────────────────────────────────
 
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    for col in ["preferred_orientation", "avoided_orientation"]:
        if col in df.columns:
            for ori in ALL_ORIENTATIONS:
                df[f"{col}_{ori}"] = (
                    df[col].fillna("")
                    .apply(lambda x, o=ori: 1 if o in str(x).split(",") else 0)
                )
            df.drop(columns=[col], inplace=True)
 
    numeric_cols = [
        "wall_free_area_m2", "wall_height_m",
        "wall_climate_median", "sector_climate_median",
        "mean_height_m", "min_height_m", "max_height_m", "height_std_m",
        "colony_nests_placed",
        "dist_to_top_edge_median", "dist_to_side_edge_median",
        "distance_to_window", "distance_to_window_median",
        "nest_temp_min_c", "nest_temp_max_c",
        "preferred_height_min_m", "preferred_height_max_m",
        "colony_size_local_min", "colony_size_local_max",
        "nest_distance_min_m", "nest_distance_max_m",
        "orientation_match",
        "door_count", "colonial",
        "noise_level", "human_tolerance_level", "dirt",
        "is_bird", "is_bat",
        "is_day_active", "is_evening_active", "is_dusk_active", "is_night_active",
        "nest_use_start_month", "nest_use_end_month", "nest_use_duration",
        "prefers_edges_proximity", "prefers_roof_proximity",
        "far_from_windows_important",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
 
    df["group_id"] = "inference"
    return df
 
 
def _label_encode_categoricals(X: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    # Applies saved LabelEncoders to categorical columns for XGBoost inference.
    # Unseen labels are mapped to the first known class.
    X = X.copy()
    for col in CATEGORICAL_COLS:
        if col not in X.columns or col not in encoders:
            continue
        enc = encoders[col]
        X[col] = X[col].astype(str).fillna("unknown")
        known = set(enc.classes_)
        X[col] = X[col].apply(lambda v: v if v in known else enc.classes_[0])
        X[col] = enc.transform(X[col])
    return X
 
 
def _build_feature_matrix(
    df: pd.DataFrame,
    model,
    model_type: str = "lgbm",
    xgb_encoders: dict | None = None,
) -> pd.DataFrame:
    orientation_flags = (
        [f"preferred_orientation_{o}" for o in ALL_ORIENTATIONS] +
        [f"avoided_orientation_{o}"   for o in ALL_ORIENTATIONS]
    )
    feature_cols = [c for c in ALL_FEATURES + orientation_flags if c in df.columns]
    X = df[feature_cols].copy()
 
    if model_type == "lgbm":
        trained_cols = model.feature_name_
        for c in trained_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[trained_cols]
        for col in CATEGORICAL_COLS:
            if col in X.columns and hasattr(X[col], "cat"):
                if "unknown" not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories("unknown")
                X[col] = X[col].fillna("unknown")
 
    else:
        # XGBoost: label-encode categoricals, fill NaN, cast to float
        trained_cols = model.feature_names_in_
        for c in trained_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[list(trained_cols)]
        if xgb_encoders:
            X = _label_encode_categoricals(X, xgb_encoders)
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.astype(float).fillna(-999.0)
 
    return X
 
 
def _score_rows(
    df: pd.DataFrame,
    model,
    model_type: str = "lgbm",
    xgb_encoders: dict | None = None,
) -> np.ndarray:
    df_eng = _engineer_features(df)
    X = _build_feature_matrix(df_eng, model, model_type=model_type, xgb_encoders=xgb_encoders)
    return model.predict(X)
 
 
# ─────────────────────────────────────────────────────────────────
# PARSE HELPERS
# ─────────────────────────────────────────────────────────────────
 
def _parse_colony_size(needs: dict) -> Tuple[int, int]:
    # Uses fpf.parse_min_max_count — same parser as the random generator.
    # Key is "colonie_size_local" (with typo) matching the species Excel sheet.
    # For solitary species colonie_size_local may be "no", "none", or non-numeric
    # — falls back to (1, 1) in that case.
    raw = (
        needs.get("colonie_size_local")
        or needs.get("colonie_size")
        or needs.get("colony_size_local")
        or needs.get("colony_size")
    )
    # guard against non-numeric values like "no", "none", "yes"
    if raw is not None:
        s = str(raw).strip().lower().replace('"', '').replace(' ', '')
        if not any(c.isdigit() for c in s):
            raw = None
    try:
        lo, hi = fpf.parse_min_max_count(raw, default=(1, 1))
    except (ValueError, TypeError):
        return 1, 1
    lo = max(1, int(lo)) if lo is not None and not (isinstance(lo, float) and math.isnan(lo)) else 1
    hi = max(1, int(hi)) if hi is not None and not (isinstance(hi, float) and math.isnan(hi)) else lo
    return lo, hi
 
 
def _parse_height_range(needs: dict) -> Tuple[float, float]:
    # Uses fpf.parse_min_height_m and fpf.parse_max_height_m — same as random generator.
    raw = needs.get("nest_height")
    lo = fpf.parse_min_height_m(raw, default=0.0)
    hi = fpf.parse_max_height_m(raw)
    if hi is None:
        hi = 999.0
    return float(lo), float(hi)
 
 
def _parse_spacing(needs: dict) -> Tuple[float, float]:
    # Uses fpf.parse_range_float for the spacing range — same as random generator.
    # Applies 10% tolerance on both ends so e.g. 1.0–1.5m becomes 0.9–1.65m.
    raw = (
        needs.get("distance_to_next_nest")
        or needs.get("if_colonial_distance_to_next_nest")
        or needs.get("distance_between_nest_boxes")
    )
    if raw is None:
        return 1.0, 1e9
 
    s = str(raw).strip().lower()
    if "close" in s:
        return 0.0, 0.5
 
    # guard against non-numeric values like "no", "none"
    if not any(c.isdigit() for c in s):
        return 0.0, 1e9
 
    try:
        lo, hi = fpf.parse_range_float(raw, default=(1.0, 1.0))
    except (ValueError, TypeError):
        return 0.0, 1e9

    tol = 0.10
    lo = max(0.0, lo * (1.0 - tol))
    hi = hi * (1.0 + tol)
    return float(lo), float(hi)
 
 
# ─────────────────────────────────────────────────────────────────
# MAIN PLANNER
# ─────────────────────────────────────────────────────────────────
 
 
def _parse_num_orientations(needs: dict) -> str:
    # Returns "1", "2", "2-3", or "3-4" based on number_of_orientations field.
    # Defaults to "1" if missing or unparseable.
    raw = needs.get("number_of_orientations")
    if raw is None:
        return "1"
    s = str(raw).strip().lower().replace('"', '').replace(' ', '')
    if s in {"2-3", "2–3", "2—3"}:
        return "2-3"
    if s in {"3-4", "3–4", "3—4"}:
        return "3-4"
    if s == "2":
        return "2"
    return "1"
 
 
def _resolve_target_walls(num_ori_str: str, n_eligible: int) -> int:
    # Resolves how many walls to use given species orientation preference
    # and number of eligible walls available.
    if num_ori_str == "1":
        return 1
    if num_ori_str == "2":
        return min(2, n_eligible)
    if num_ori_str == "2-3":
        if n_eligible <= 3:
            return min(2, n_eligible)
        return min(3, n_eligible)
    if num_ori_str == "3-4":
        # 2 walls available -> 2 placements, 3 -> 3, 4+ -> 4 (capped at 4)
        return min(4, n_eligible)
    return 1
 
 
def _parse_solitary_boxes(needs: dict) -> Tuple[int, int]:
    # Parses number_of_individual_nest_boxes_on_building.
    # Returns (min_boxes, max_boxes). Returns (1, 1) if "no", None, or non-numeric.
    raw = needs.get("number_of_individual_nest_boxes_on_building")
    if raw is None:
        return 1, 1
    s = str(raw).strip().lower().replace('"', '').replace(' ', '')
    if s in {"no", "none", "nan", ""}:
        return 1, 1
    if not any(c.isdigit() for c in s):
        return 1, 1
    try:
        lo, hi = fpf.parse_min_max_count(raw, default=(1, 1))
        lo = max(1, int(lo)) if lo is not None and not (isinstance(lo, float) and math.isnan(lo)) else 1
        hi = max(1, int(hi)) if hi is not None and not (isinstance(hi, float) and math.isnan(hi)) else lo
        return lo, hi
    except (ValueError, TypeError):
        return 1, 1
 
 
def _parse_box_spacing(needs: dict) -> Tuple[float, float]:
    # Parses distance_between_nest_boxes in metres.
    # Returns (dmin, dmax). Returns (0.0, 1e9) if not specified.
    raw = needs.get("distance_between_nest_boxes")
    if raw is None:
        return 0.0, 1e9
    s = str(raw).strip().lower()
    if not any(c.isdigit() for c in s):
        return 0.0, 1e9
    try:
        lo, hi = fpf.parse_range_float(raw, default=(0.0, 1e9))
        return float(lo), float(hi)
    except (ValueError, TypeError):
        return 0.0, 1e9
 
 
_SECTOR_ROWS = ["bottom", "middle", "top"]
_SECTOR_COLS = ["left",   "middle", "right"]
 
def _adjacent_sectors(sector_row: str, sector_col: str) -> list:
    # Returns strictly adjacent (non-diagonal) sector labels in the 3x3 grid.
    ri = _SECTOR_ROWS.index(sector_row)
    ci = _SECTOR_COLS.index(sector_col)
    adjacent = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ri + dr, ci + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            adjacent.append((_SECTOR_ROWS[nr], _SECTOR_COLS[nc]))
    return adjacent
 
 
def _place_solitary_boxes(
    building_dict: dict,
    wall_id: str,
    wall: dict,
    first_placement: dict,
    needs: dict,
    model,
    height_min: float,
    height_max: float,
    building_zero_z: float,
    usable_geom,
    model_type: str,
    xgb_encoders,
    max_boxes: int,
    dmin_box: float,
    dmax_box: float,
) -> list:
    # Places additional solitary nest boxes on the same wall after the first placement.
    # Returns a list of placement dicts (including the first one).
    # Rules:
    #   - Each new box must be dmin_box–dmax_box from ANY already placed box
    #   - Try same sector first, then adjacent sectors (strict adjacency, scored)
    #   - If new box lands in different sector → stop
    #   - If new box lands in same sector → try another box (up to max_boxes total)
 
    def _dist2(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
 
    placed = [first_placement]
    placed_uvs = [first_placement["uv"][0] if first_placement.get("uv") else None]
    current_sector_row = first_placement["section_row"]
    current_sector_col = first_placement["section_col"]
 
    while len(placed) < max_boxes:
        placed_uv_list = [p["uv"][0] for p in placed if p.get("uv")]
 
        # collect candidate sectors: same sector first, then adjacent
        candidate_sectors = [(current_sector_row, current_sector_col)] +                             _adjacent_sectors(current_sector_row, current_sector_col)
 
        new_placement = None
        new_sector_row = None
        new_sector_col = None
 
        # score all candidate sectors and try in ranked order
        scored_candidates = []
        for sr, sc in candidate_sectors:
            pts = _get_sector_feasible_points(
                building_dict=building_dict,
                wall_id=wall_id,
                sector_row=sr,
                sector_col=sc,
                min_height_m=height_min,
                max_height_m=height_max,
                usable_geom=usable_geom,
                needs=needs,
            )
            # filter to only points in valid distance range from any placed box
            valid_pts = []
            for pt in pts:
                c_uv = pt[1]
                too_close = any(_dist2(c_uv, p_uv) < dmin_box for p_uv in placed_uv_list if p_uv)
                in_range  = any(_dist2(c_uv, p_uv) <= dmax_box for p_uv in placed_uv_list if p_uv)
                if not too_close and in_range:
                    valid_pts.append(pt)
            if valid_pts:
                scored_candidates.append((sr, sc, valid_pts))
 
        if not scored_candidates:
            break
 
        # score sectors and try best first
        if len(scored_candidates) > 1:
            sector_feat_rows = [
                _build_sector_feature_row(
                    wall_id=wall_id, wall=wall,
                    sector_row=sr, sector_col=sc,
                    feasible_pts=vpts, needs=needs,
                )
                for sr, sc, vpts in scored_candidates
            ]
            df_sc = pd.DataFrame(sector_feat_rows)
            sc_scores = _score_rows(df_sc, model, model_type=model_type, xgb_encoders=xgb_encoders)
            order = list(np.argsort(sc_scores)[::-1])
        else:
            order = [0]
 
        for idx in order:
            sr, sc, valid_pts = scored_candidates[idx]
 
            # pick highest-scoring valid point
            pt_rows = [
                _build_point_feature_row(
                    pid=pt[0], uv=pt[1], xyz=pt[2],
                    wall_id=wall_id, wall=wall,
                    sector_row=sr, sector_col=sc,
                    wall_ground_z=building_zero_z,
                    needs=needs,
                )
                for pt in valid_pts
            ]
            df_pts = pd.DataFrame(pt_rows)
            pt_scores = _score_rows(df_pts, model, model_type=model_type, xgb_encoders=xgb_encoders)
            best_pt_idx = int(np.argmax(pt_scores))
            best_pt = valid_pts[best_pt_idx]
 
            wall_score = float(_score_rows(
                pd.DataFrame([_build_wall_feature_row(wall_id, wall, needs)]),
                model, model_type=model_type, xgb_encoders=xgb_encoders
            )[0])
 
            new_placement = {
                "wall_id":            wall_id,
                "wall_score":         round(wall_score, 4),
                "section_row":        sr,
                "section_col":        sc,
                "sector_score":       round(float(pt_scores[best_pt_idx]), 4),
                "placement_score":    round(float(pt_scores[best_pt_idx]), 4),
                "xyz":                [best_pt[2]],
                "uv":                 [best_pt[1]],
                "colony_size":        1,
                "species":            first_placement.get("species", ""),
                "selected_point_ids": [best_pt[0]],
                "sector_rank_used":   0,
            }
            new_sector_row = sr
            new_sector_col = sc
            break
 
        if new_placement is None:
            break
 
        placed.append(new_placement)
        placed_uv_list.append(new_placement["uv"][0])
 
        # if new box is in a different sector → stop
        if new_sector_row != current_sector_row or new_sector_col != current_sector_col:
            break
        # else continue in same sector
 
    return placed
 
 
def _try_place_on_wall(
    building_dict: dict,
    wall_id: str,
    needs: dict,
    model,
    height_min: float,
    height_max: float,
    building_zero_z: float,
    colony_min: int,
    colony_max: int,
    dmin_m: float,
    dmax_m: float,
    model_type: str,
    xgb_encoders,
    species_name: str = "",
    sector_rank: int = 0,
    exclude_point_ids: set | None = None,
) -> Optional[Dict[str, Any]]:
    # Attempts placement on a specific wall at a given sector rank (0=best, 1=second best).
    # Returns a placement dict or None.
    wall = building_dict[wall_id]
 
    usable_geom, _ = _wall_usable_geometry(wall, needs)
    if usable_geom is None:
        return None
 
    all_sectors = _get_all_sectors(building_dict, wall_id)
    if not all_sectors:
        return None
 
    feasible_sectors = []
    for sec_row, sec_col in all_sectors:
        pts = _get_sector_feasible_points(
            building_dict=building_dict,
            wall_id=wall_id,
            sector_row=sec_row,
            sector_col=sec_col,
            min_height_m=height_min,
            max_height_m=height_max,
            usable_geom=usable_geom,
            needs=needs,
        )
        if len(pts) >= colony_min:
            feasible_sectors.append((sec_row, sec_col, pts))
 
    if not feasible_sectors:
        return None
 
    # score sectors
    wall_obj = building_dict[wall_id]
    sector_rows = [
        _build_sector_feature_row(
            wall_id=wall_id, wall=wall_obj,
            sector_row=sr, sector_col=sc,
            feasible_pts=pts, needs=needs,
        )
        for sr, sc, pts in feasible_sectors
    ]
    df_sectors = pd.DataFrame(sector_rows)
    sector_scores_arr = _score_rows(
        df_sectors, model, model_type=model_type, xgb_encoders=xgb_encoders
    )
    df_sectors["_score"] = sector_scores_arr
    df_sectors["_sector_idx"] = list(range(len(feasible_sectors)))
    df_sectors = df_sectors.sort_values("_score", ascending=False).reset_index(drop=True)
 
    # determine which sector rank to use
    # if sector_rank >= number of available sectors, fall back to rank 0 with looser colony_min
    actual_colony_min = colony_min
    if sector_rank >= len(df_sectors):
        sector_rank = 0
        actual_colony_min = max(1, colony_min - 1)  # Option 2 tolerance
 
    # try sectors starting from sector_rank
    # for sector_rank > 0 (Option 2): try higher ranks first, then fall back to rank 0
    # with exclude_point_ids to ensure a different placement from Option 1
    if sector_rank == 0:
        sector_order = list(range(0, len(df_sectors)))
    else:
        sector_order = list(range(sector_rank, len(df_sectors))) + [0]
 
    for rank_pos in sector_order:
        row_data = df_sectors.iloc[rank_pos]
        idx = int(row_data["_sector_idx"])
        sec_row, sec_col, pts = feasible_sectors[idx]
        sec_score = float(row_data["_score"])
 
        placement = _place_colony_in_sector(
            feasible_pts=pts,
            wall_id=wall_id,
            wall=wall_obj,
            sector_row=sec_row,
            sector_col=sec_col,
            wall_ground_z=building_zero_z,
            needs=needs,
            model=model,
            colony_size_min=actual_colony_min,
            colony_size_max=colony_max,
            dmin_m=dmin_m,
            dmax_m=dmax_m,
            model_type=model_type,
            xgb_encoders=xgb_encoders,
            exclude_point_ids=exclude_point_ids,
        )
 
        if placement is not None:
            # score placed points
            placed_pts = list(zip(
                placement["selected_point_ids"],
                placement["uv"],
                placement["xyz"],
            ))
            placement_rows = [
                _build_point_feature_row(
                    pid=pid, uv=uv, xyz=xyz,
                    wall_id=wall_id, wall=wall_obj,
                    sector_row=sec_row,
                    sector_col=sec_col,
                    wall_ground_z=building_zero_z,
                    needs=needs,
                )
                for pid, uv, xyz in placed_pts
            ]
            df_placed = pd.DataFrame(placement_rows)
            pt_scores = _score_rows(
                df_placed, model, model_type=model_type, xgb_encoders=xgb_encoders
            )
            placement_score = float(np.mean(pt_scores))
 
            wall_score = float(_score_rows(
                pd.DataFrame([_build_wall_feature_row(wall_id, wall_obj, needs)]),
                model, model_type=model_type, xgb_encoders=xgb_encoders
            )[0])
 
            first_result = {
                "wall_id":            wall_id,
                "wall_score":         round(wall_score, 4),
                "section_row":        sec_row,
                "section_col":        sec_col,
                "sector_score":       round(sec_score, 4),
                "placement_score":    round(placement_score, 4),
                "xyz":                placement["xyz"],
                "uv":                 placement["uv"],
                "colony_size":        placement["colony_size"],
                "species":            species_name,
                "selected_point_ids": placement["selected_point_ids"],
                "sector_rank_used":   rank_pos,
            }
 
            # solitary multi-box: check if species needs multiple nest boxes
            min_boxes, max_boxes = _parse_solitary_boxes(needs)
            dmin_box, dmax_box   = _parse_box_spacing(needs)
 
            if max_boxes > 1:
                all_placements = _place_solitary_boxes(
                    building_dict=building_dict,
                    wall_id=wall_id,
                    wall=wall_obj,
                    first_placement=first_result,
                    needs=needs,
                    model=model,
                    height_min=height_min,
                    height_max=height_max,
                    building_zero_z=building_zero_z,
                    usable_geom=usable_geom,
                    model_type=model_type,
                    xgb_encoders=xgb_encoders,
                    max_boxes=max_boxes,
                    dmin_box=dmin_box,
                    dmax_box=dmax_box,
                )
                first_result["solitary_boxes"] = all_placements
 
            return first_result
 
    return None
 


def plan(
    model,
    building_dict: dict,
    species_name: str,
    needs: dict,
    n_options: int = 3,
    model_type: str = "lgbm",
    xgb_encoders: dict | None = None,
) -> List[Dict[str, Any]]:
    """
    AI-supported hierarchical planner. Returns up to n_options placement proposals,
    each on a different wall. No random candidate generation — the AI guides every
    decision within hard constraints.
 
    Parameters
    ──────────
    model         : trained LightGBM or XGBoost ranker model
    building_dict : building dict as loaded by fpf.load_building_dict
    species_name  : species string (for output labelling only)
    needs         : species needs dict (ecological parameters)
    n_options     : number of placement options to return (default 3)
    model_type    : "lgbm" or "xgb" (default "lgbm")
    xgb_encoders  : label encoders dict required when model_type="xgb"
 
    Returns
    ───────
    List of dicts, one per option:
        option_rank, wall_id, wall_score, section_row, section_col,
        sector_score, placement_score, xyz, uv, colony_size, species
    """
 
    # ensure wall orientation and climate are precomputed
    fpf.precompute_wall_climate_features(building_dict)
    fpf.precompute_wall_orientations(building_dict)
 
    # precompute wall features that exist in training but are not stored in building_dict:
    #   wall_free_area_m2, neighbor_umin, neighbor_umax, building_function
    building_function = str(building_dict.get("building_function") or "").strip().lower() or None
    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue
        # free area
        wall["wall_free_area_m2"] = fpf.free_wall_area(wall)
        # building function (same value for all walls)
        wall["building_function"] = building_function
        # neighbor floor functions per side
        try:
            nbs = fpf.neighbor_floor_functions(building_dict, wall_id)
        except Exception:
            nbs = []
        wall["neighbor_umin"] = "none"
        wall["neighbor_umax"] = "none"
        for nb in (nbs or []):
            side = (nb.get("matched_side") or "").strip().lower()
            ff   = str(nb.get("floor_function") or "none").strip().lower()
            if side == "umin":
                wall["neighbor_umin"] = ff
            elif side == "umax":
                wall["neighbor_umax"] = ff
 
    # parse hard constraint ranges once
    colony_min, colony_max = _parse_colony_size(needs)
    height_min, height_max = _parse_height_range(needs)
    dmin_m,     dmax_m     = _parse_spacing(needs)
 
    # compute building zero: lowest Z across all walls
    all_zs = []
    for _w in building_dict.values():
        if not isinstance(_w, dict):
            continue
        for _pd in (_w.get("grid") or {}).values():
            _xyz = _pd.get("point_on_wall")
            if _xyz and isinstance(_xyz[2], (int, float)):
                all_zs.append(float(_xyz[2]))
    building_zero_z = float(min(all_zs)) if all_zs else 0.0
    _cap = f"{height_max}" if (ENFORCE_MAX_HEIGHT and height_max > 0) else "no upper bound"
    print(f"  Building zero Z: {building_zero_z:.3f}  "
          f"min height: {height_min} m above each wall's own base, cap: {_cap}")
 
    # STEP 1: filter walls by hard constraints
    eligible_walls = []
    for wall_id, wall in building_dict.items():
        if _is_wall_hard_excluded(wall_id, wall):
            continue
        eligible_walls.append(wall_id)
 
    if not eligible_walls:
        raise ValueError("No walls passed hard constraints.")
 
    print(f"  {len(eligible_walls)} walls passed hard constraints.")
 
    # STEP 2: score and rank walls
    wall_rows = [
        _build_wall_feature_row(wall_id, building_dict[wall_id], needs)
        for wall_id in eligible_walls
    ]
    df_walls = pd.DataFrame(wall_rows)
    wall_scores = _score_rows(df_walls, model, model_type=model_type, xgb_encoders=xgb_encoders)
    df_walls["_score"] = wall_scores
    df_walls = df_walls.sort_values("_score", ascending=False).reset_index(drop=True)
    ranked_wall_ids = df_walls["wall_id"].tolist()
    print(f"  Walls ranked. Top wall: {ranked_wall_ids[0]}")
 
    # parse number of orientations
    num_ori_str  = _parse_num_orientations(needs)
    target_walls = _resolve_target_walls(num_ori_str, len(eligible_walls))
    print(f"  Orientation mode: {num_ori_str}  target walls: {target_walls}")
 
    # A solitary species whose orientation mode resolves to a single wall used to
    # get both options on that one wall, differing only by sector — so Option 2
    # was never a real alternative facade. Look one wall deeper so Option 2 can
    # be offered on the SECOND-best wall instead.
    solitary = _species_fields(needs).get("colonial") == 0
    walls_wanted = target_walls + 1 if (solitary and target_walls == 1) else target_walls

    # collect the top ranked walls that can actually produce a placement
    viable_walls: List[str] = []
    for wid in ranked_wall_ids:
        if len(viable_walls) >= walls_wanted:
            break
        # quick feasibility check — does this wall have any feasible sectors?
        wall = building_dict[wid]
        usable_geom, _ = _wall_usable_geometry(wall, needs)
        if usable_geom is None:
            continue
        all_sectors = _get_all_sectors(building_dict, wid)
        has_any = any(
            len(_get_sector_feasible_points(
                building_dict=building_dict, wall_id=wid,
                sector_row=sr, sector_col=sc,
                min_height_m=height_min, max_height_m=height_max,
                usable_geom=usable_geom, needs=needs,
            )) >= colony_min
            for sr, sc in all_sectors
        )
        if has_any:
            viable_walls.append(wid)
 
    if not viable_walls:
        raise ValueError("Planning failed: no viable walls found.")
 
    # STEP 3 + 4: build two options, as (walls, sector_rank) pairs.
    #
    #   solitary + single target wall -> Option 1 = best wall, Option 2 = second
    #                                    best wall, each in its own best sector
    #   otherwise                     -> both options span the same target walls,
    #                                    Option 1 = best sector, Option 2 = second
    #                                    best sector (with fallback)
    if solitary and target_walls == 1 and len(viable_walls) >= 2:
        option_specs = [([viable_walls[0]], 0), ([viable_walls[1]], 0)]
        rerank_by_score = False          # wall ranking already fixes the order
    else:
        walls = viable_walls[:target_walls]
        option_specs = [(walls, 0), (walls, 1)]
        rerank_by_score = target_walls == 1

    raw_options = []
    # tracks Option 1 point ids per wall so Option 2 can avoid duplicates
    opt1_ids_by_wall: Dict[str, set] = {}

    for option_no, (option_walls, sector_rank) in enumerate(option_specs, start=1):
        placements = []
        for wid in option_walls:
            result = _try_place_on_wall(
                building_dict=building_dict,
                wall_id=wid,
                needs=needs,
                model=model,
                height_min=height_min,
                height_max=height_max,
                building_zero_z=building_zero_z,
                colony_min=colony_min,
                colony_max=colony_max,
                dmin_m=dmin_m,
                dmax_m=dmax_m,
                model_type=model_type,
                xgb_encoders=xgb_encoders,
                species_name=species_name,
                sector_rank=sector_rank,
                exclude_point_ids=opt1_ids_by_wall.get(wid),
            )
            if result is not None:
                # merge solitary_boxes back into placement if present
                if result.get("solitary_boxes") and len(result["solitary_boxes"]) > 1:
                    all_boxes = result["solitary_boxes"]
                    result["xyz"]                = [b["xyz"][0] for b in all_boxes]
                    result["uv"]                 = [b["uv"][0] for b in all_boxes]
                    result["selected_point_ids"] = [b["selected_point_ids"][0] for b in all_boxes]
                    result["colony_size"]        = len(all_boxes)
                placements.append(result)
                # store Option 1 point ids for duplicate detection in Option 2
                if option_no == 1:
                    opt1_ids_by_wall[wid] = set(result.get("selected_point_ids") or [])
                print(f"  Option {option_no} wall={wid} "
                      f"sector=({result['section_row']},{result['section_col']}) "
                      f"colony_size={result['colony_size']} "
                      f"placement_score={result['placement_score']}")
            else:
                print(f"  Option {option_no} wall={wid}: placement failed — skipped.")
 
        if placements:
            best_idx = int(np.argmax([p["placement_score"] for p in placements]))
            raw_options.append({
                "placements":        placements,
                "best_placement_idx": best_idx,
            })
 
    if not raw_options:
        raise ValueError("Planning failed: no valid placement found on any wall.")
 
    # Same-wall options only: re-rank so the higher placement_score becomes
    # Option 1. Skipped when the two options are different walls — there the
    # wall ranking already decides which is second best.
    if rerank_by_score and len(raw_options) == 2:
        score_opt1 = raw_options[0]["placements"][0]["placement_score"]
        score_opt2 = raw_options[1]["placements"][0]["placement_score"]
        if score_opt2 > score_opt1:
            raw_options = [raw_options[1], raw_options[0]]
 
    # assign option_rank
    options = []
    for rank, opt in enumerate(raw_options, start=1):
        opt["option_rank"] = rank
        options.append(opt)
 
    return options
 
# ─────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────
 
def run_batch(
    building_jsons: list,
    species_names: list,
    excel_path: str,
    *,
    output_root: Path,
    model_dir: Path,
    models: list = ["lgbm"],  # "lgbm", "xgb", or both
    n_iterations: int = 1,
    n_options: int = 2,
    elev: float = 25,
    azim: float = 215,
    model_variant: str = "full",  # "full" or "reduced"
) -> pd.DataFrame:
    """
    Batch planner runner across multiple buildings and species.

    Parameters
    ──────────
    building_jsons  : list of Path or str to building JSON files
    species_names   : list of species name strings
    excel_path      : path to species Excel file
    output_root     : root output folder
    model_dir       : folder containing model .pkl files (the "full" model files
                      live directly in this folder; "reduced" ones in model_dir/reduced)
    models          : list of model types to run — "lgbm", "xgb", or both
    n_iterations    : number of planning runs per building/species/model combo
    n_options       : number of options per run (default 2)
    elev            : camera elevation angle for the PDF axonometric view (default 25)
    azim            : camera azimuth angle for the PDF axonometric view (default 215)
    model_variant   : "full" (default) or "reduced" — selects which trained models to load

    Output structure
    ────────────────
    output_root / model_name / building_id / species_name / species_name_iter_XXXX.pdf

    Returns
    ───────
    DataFrame with one row per option per placement, saved as Excel.
    """
    import uuid
    from multispecies_facades_planner_AI import facade_planner_visualisation as fpv
    from multispecies_facades_planner_AI import facade_planner_visAI as fpvAI

    if model_variant not in {"full", "reduced"}:
        raise ValueError(f"model_variant must be 'full' or 'reduced', got {model_variant!r}")

    if model_variant == "reduced":
        model_dir_resolved = Path(model_dir) / "reduced"
        suffix = "_reduced"
    else:
        model_dir_resolved = Path(model_dir)
        suffix = ""

    # load models
    loaded_models = {}
    if "lgbm" in models:
        lgbm_path = model_dir_resolved / f"nestworks_lgbm_ranker{suffix}.pkl"
        loaded_models["lgbm"] = {"model": joblib.load(lgbm_path), "encoders": None}
        print(f"Loaded LightGBM model from {lgbm_path}")
    if "xgb" in models:
        xgb_path = model_dir_resolved / f"nestworks_xgb_ranker{suffix}.pkl"
        enc_path  = model_dir_resolved / f"nestworks_xgb_encoders{suffix}.pkl"
        loaded_models["xgb"] = {
            "model":    joblib.load(xgb_path),
            "encoders": joblib.load(enc_path),
        }
        print(f"Loaded XGBoost model from {xgb_path}")
 
    all_rows = []
 
    for building_json in building_jsons:
        building_json = Path(building_json)
        building_dict = fpf.load_building_dict(str(building_json))
 
        # derive building id from filename e.g. building4336.json → 4336
        building_id = building_json.stem.replace("building", "")
 
        for species_name in species_names:
            try:
                needs = de.load_species_training_as_dict(
                    excel_path, species_name
                )[species_name]
            except Exception as e:
                print(f"  Skipping {species_name}: {e}")
                continue
 
            for model_name, model_info in loaded_models.items():
                model    = model_info["model"]
                encoders = model_info["encoders"]
 
                for i in range(n_iterations):
                    iter_id = uuid.uuid4().hex[:8]
 
                    print(f"\n{'='*50}")
                    print(f"Building: {building_id}  Species: {species_name}  "
                          f"Model: {model_name}  Iteration: {i+1}/{n_iterations}")
 
                    try:
                        options = plan(
                            model=model,
                            building_dict=building_dict,
                            species_name=species_name,
                            needs=needs,
                            n_options=n_options,
                            model_type=model_name,
                            xgb_encoders=encoders,
                        )
                    except Exception as e:
                        print(f"  Planning failed: {e}")
                        continue
 
                    # determine colonial/solitary mode
                    colonial_raw = str(needs.get("colonie") or "").strip().lower().replace('"', '')
                    mode = "colony" if colonial_raw == "yes" else "solitary"
 
                    # save PDF
                    pdf_dir = (
                        Path(output_root) / model_name / building_id / species_name
                    )
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    pdf_name = f"{species_name}_{iter_id}.pdf"
                    pdf_path = pdf_dir / pdf_name
 
                    try:
                        wall_label_pos = fpv.compute_wall_label_positions(building_dict)
                        fpvAI.export_plan_options_pdf(
                            building_dict=building_dict,
                            options=options,
                            wall_label_pos=wall_label_pos,
                            pdf_path=pdf_path,
                            elev=elev,
                            azim=azim,
                        )
                        print(f"  PDF saved: {pdf_path}")
                    except Exception as e:
                        print(f"  PDF export failed: {e}")
                        pdf_path = None
 
                    # collect rows for Excel
                    for opt in options:
                        for p_idx, p in enumerate(opt["placements"]):
                            wall_meta = building_dict.get(p["wall_id"]) or {}
                            wall_ori  = (wall_meta.get("orientation") or "").strip().upper()
                            is_best   = (p_idx == opt["best_placement_idx"])
 
                            all_rows.append({
                                "model":            model_name,
                                "building_id":      building_id,
                                "species":          species_name,
                                "mode":             mode,
                                "iteration_id":     iter_id,
                                "option_rank":      opt["option_rank"],
                                "placement_idx":    p_idx,
                                "is_best_placement": is_best,
                                "wall_id":          p["wall_id"],
                                "wall_orientation": wall_ori,
                                "section_row":      p["section_row"],
                                "section_col":      p["section_col"],
                                "nests_placed":     p["colony_size"],
                                "placement_score":  p["placement_score"],
                                "wall_score":       p.get("wall_score"),
                                "sector_score":     p.get("sector_score"),
                                "pdf_path":         str(pdf_path) if pdf_path else "",
                            })
 
    df = pd.DataFrame(all_rows)
 
    # save Excel
    excel_out = Path(output_root) / "nestworks_batch_results.xlsx"
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_out, index=False)
    print(f"\nResults saved to: {excel_out}")
 
    return df


# ─────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multispecies_facades_planner_AI import facade_planner_visualisation as fpv
    from multispecies_facades_planner_AI import facade_planner_visAI as fpvAI

    DATA_ROOT  = Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\tests\data_training")
    MODEL_DIR  = DATA_ROOT / "models"

    BUILDING_JSON = Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building4868.json")

    # load XGBoost model + encoders
    # model        = joblib.load(MODEL_DIR / "nestworks_xgb_ranker.pkl")
    # xgb_encoders = joblib.load(MODEL_DIR / "nestworks_xgb_encoders.pkl")
    # print("XGBoost model loaded.")

    # load LGBM model
    model        = joblib.load(MODEL_DIR / "nestworks_lgbm_ranker.pkl")

    # load building
    building_dict = fpf.load_building_dict(str(BUILDING_JSON))

    # load species needs
    needs = de.load_species_training_as_dict(excel_path, "black_redstart")["black_redstart"]


    # # run planner with XGBoost
    # options = plan(
    #     model=model,
    #     building_dict=building_dict,
    #     species_name="house_sparrow",
    #     needs=needs,
    #     model_type="xgb",
    #     xgb_encoders=xgb_encoders,
    # )

    # # run planner with LGBM
    # 'options = plan(
    #     model=model,
    #     building_dict=building_dict,
    #     species_name="black_redstart",
    #     needs=needs,
    #     model_type="lgbm",
    # )'

    # pdf_path = DATA_ROOT / "nestworks_plan_options_LGBM_solitary_0173.pdf"
    # pdf_path.parent.mkdir(parents=True, exist_ok=True)


    # wall_label_pos = fpv.compute_wall_label_positions(building_dict)
    # fpvAI.export_plan_options_pdf(
    #     building_dict=building_dict,
    #     options=options,
    #     wall_label_pos=wall_label_pos,
    #     pdf_path=pdf_path,
    # )

    BUILDINGS_DIR = Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export")

    #building_jsons = list(BUILDINGS_DIR.glob("building*.json"))

    building_jsons_test=[
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building3211.json",
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building2573.json",
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building0173.json",
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building0311.json",
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building4336.json",
        r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\buildings_export\building4868.json",
    ]

    # #species_names=["black_redstart", "blue_tit", "house_martin", "house_sparrow", "tree_sparrow", "common_pipistrelle"],

    # results_df = run_batch(
    #     building_jsons=building_jsons_test,
    #     species_names=["black_redstart", "blue_tit", "house_martin", "house_sparrow", "tree_sparrow", "common_pipistrelle"],
    #     excel_path=excel_path,
    #     output_root=Path(r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\tests\results\batch_results_full_model"),
    #     model_dir=MODEL_DIR,
    #     models=["lgbm","xgb"],
    #     n_iterations=1,
    #     azim = 150,
    #     model_variant = "full"
    # )


    # climate features must be precomputed before the overview can use them
    # fpf.precompute_wall_climate_features(building_dict)
    # fpf.precompute_wall_orientations(building_dict)

    # hottest_wall_id, _ = fpf.wall_hot_climate_median(building_dict)
    # wall_label_pos = fpv.compute_wall_label_positions(building_dict)

    # fpv.export_building_overview_pdf(
    #     building_dict=building_dict,
    #     hottest_wall_id=hottest_wall_id,
    #     wall_label_pos=wall_label_pos,
    #     pdf_path=DATA_ROOT / "building4868_climate_overview.pdf",
    #     azim=215,
    # )