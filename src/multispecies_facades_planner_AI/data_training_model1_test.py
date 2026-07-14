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
from multispecies_facades_planner_AI import data_extraction as de

# ─────────────────────────────────────────────────────────────────
# CONSTANTS  — mirror training pipeline exactly
# ─────────────────────────────────────────────────────────────────
excel_path = r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\data\Datasets\bird_species.xlsx"



ALL_ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
 
ALL_FEATURES = [
    "wall_free_area_m2", "door_count", "wall_shape",
    "orientation", "floor_function", "neighbor_umin", "neighbor_umax",
    "building_function", "wall_climate_median",
    "section_row", "section_col",
    "mean_height_m", "min_height_m", "max_height_m",
    "dist_to_top_edge_median", "dist_to_side_edge_median",
    "side_edge_label", "orientation_match", "sector_climate_median",
    "colonial", "colony_size_local_min", "colony_size_local_max",
    "noise_level", "human_tolerance_level", "dirt", "taxa",
    "is_day_active", "is_evening_active", "is_dusk_active",
    "nest_use_start_month", "nest_use_end_month", "nest_use_duration",
    "preferred_height_min_m", "preferred_height_max_m",
    "prefers_edges_proximity", "prefers_roof_proximity",
    "nest_temp_min_c", "nest_temp_max_c",
]
 
CATEGORICAL_COLS = [
    "orientation", "floor_function", "neighbor_umin", "neighbor_umax",
    "building_function", "section_row", "section_col",
    "side_edge_label", "wall_shape", "taxa",
]
 
 
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
 
    # exclude walls with negligible free area — too small to place anything
    free_area = wall.get("wall_free_area_m2")
    if free_area is not None:
        try:
            if float(free_area) < 2.0:
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
    ori = (wall.get("orientation") or "").strip().upper()
    preferred_raw = (needs.get("preferred_orientation") or "")
    preferred = [o.strip().upper() for o in str(preferred_raw).split(",") if o.strip()]
    orientation_match = 1 if (ori and ori in preferred) else 0
 
    row = {
        "wall_id": wall_id,
        "wall_free_area_m2":    wall.get("wall_free_area_m2"),
        "door_count":           wall.get("door_count"),
        "wall_shape":           wall.get("wall_shape"),
        "orientation":          ori or None,
        "floor_function":       wall.get("floor_function"),
        "neighbor_umin":        wall.get("neighbor_umin"),
        "neighbor_umax":        wall.get("neighbor_umax"),
        "building_function":    wall.get("building_function"),
        "wall_climate_median":  wall.get("wall_climate_median"),
        "orientation_match":    orientation_match,
        "section_row":          None,
        "section_col":          None,
        "mean_height_m":        None,
        "min_height_m":         None,
        "max_height_m":         None,
        "dist_to_top_edge_median":  None,
        "dist_to_side_edge_median": None,
        "side_edge_label":      None,
        "sector_climate_median": None,
        "colonial":                 needs.get("colonial"),
        "colony_size_local_min":    needs.get("colony_size_local_min"),
        "colony_size_local_max":    needs.get("colony_size_local_max"),
        "noise_level":              needs.get("noise_level"),
        "human_tolerance_level":    needs.get("human_tolerance_level"),
        "dirt":                     needs.get("dirt"),
        "taxa":                     needs.get("taxa"),
        "is_day_active":            needs.get("is_day_active"),
        "is_evening_active":        needs.get("is_evening_active"),
        "is_dusk_active":           needs.get("is_dusk_active"),
        "nest_use_start_month":     needs.get("nest_use_start_month"),
        "nest_use_end_month":       needs.get("nest_use_end_month"),
        "nest_use_duration":        needs.get("nest_use_duration"),
        "preferred_height_min_m":   needs.get("preferred_height_min_m"),
        "preferred_height_max_m":   needs.get("preferred_height_max_m"),
        "prefers_edges_proximity":  needs.get("prefers_edges_proximity"),
        "prefers_roof_proximity":   needs.get("prefers_roof_proximity"),
        "nest_temp_min_c":          needs.get("nest_temp_min_c"),
        "nest_temp_max_c":          needs.get("nest_temp_max_c"),
        "preferred_orientation":    preferred_raw,
        "avoided_orientation":      needs.get("avoided_orientation"),
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
    building_zero_z: float,
) -> list:
    # Returns grid points in this sector that pass height and usable area constraints.
    # sector_row/col are computed from UV via fpf.sector_3x3_labels — NOT read from
    # stored grid fields (they are not stored there).
    # Height is measured from building_zero_z (lowest Z across all building walls).
    wall = building_dict[wall_id]
    grid = wall.get("grid") or {}
    bbox = fpf.wall_uv_bbox_from_building(building_dict, wall_id)
    if bbox is None:
        return []
 
    pts = []
 
    for pid, pdata in grid.items():
        uv  = pdata.get("uv")
        xyz = pdata.get("point_on_wall")
        if not uv or not xyz:
            continue
 
        row_label, col_label = fpf.sector_3x3_labels(
            float(uv[0]), float(uv[1]), bbox
        )
        if row_label != sector_row or col_label != sector_col:
            continue
 
        z = float(xyz[2])
        rel_h = z - building_zero_z
 
        if rel_h < min_height_m:
            continue
        if max_height_m > 0 and rel_h > max_height_m:
            continue
 
        p = Point(float(uv[0]), float(uv[1]))
        if not usable_geom.contains(p):
            continue
 
        pts.append((pid, [float(uv[0]), float(uv[1])], [float(xyz[0]), float(xyz[1]), z]))
 
    return pts
 
 
def _wall_usable_geometry(wall: dict):
    # Computes the usable area polygon for this wall (openings removed, offset applied).
    # Returns usable_geom and wall_ground_z, or (None, None) if wall is unusable.
    free_geom   = fpf.derive_openings(wall)
    offset_geom = fpf.build_offset_area(wall)
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
 
    seen = set()
    for pdata in grid.values():
        uv = pdata.get("uv")
        if not uv or len(uv) < 2:
            continue
        row_label, col_label = fpf.sector_3x3_labels(
            float(uv[0]), float(uv[1]), bbox
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
    heights = [pt[2][2] for pt in feasible_pts]
 
    grid = wall.get("grid") or {}
    zs_all = [
        float(pdata["point_on_wall"][2])
        for pdata in grid.values()
        if pdata.get("point_on_wall")
    ]
    wall_ground_z = min(zs_all) if zs_all else 0.0
 
    rel_heights = [z - wall_ground_z for z in heights]
    mean_h = float(np.mean(rel_heights)) if rel_heights else None
    min_h  = float(np.min(rel_heights))  if rel_heights else None
    max_h  = float(np.max(rel_heights))  if rel_heights else None
 
    sec_climate_vals = []
    for pid, uv, xyz in feasible_pts:
        pdata = grid.get(pid) or {}
        cv = pdata.get("climate_value") or pdata.get("solar_radiation")
        if cv is not None:
            try:
                sec_climate_vals.append(float(cv))
            except (ValueError, TypeError):
                pass
    sector_climate = float(np.median(sec_climate_vals)) if sec_climate_vals else None
 
    ori = (wall.get("orientation") or "").strip().upper()
    preferred_raw = (needs.get("preferred_orientation") or "")
    preferred = [o.strip().upper() for o in str(preferred_raw).split(",") if o.strip()]
    orientation_match = 1 if (ori and ori in preferred) else 0
 
    row = {
        "wall_id":    wall_id,
        "sector_row": sector_row,
        "sector_col": sector_col,
        "wall_free_area_m2":    wall.get("wall_free_area_m2"),
        "door_count":           wall.get("door_count"),
        "wall_shape":           wall.get("wall_shape"),
        "orientation":          ori or None,
        "floor_function":       wall.get("floor_function"),
        "neighbor_umin":        wall.get("neighbor_umin"),
        "neighbor_umax":        wall.get("neighbor_umax"),
        "building_function":    wall.get("building_function"),
        "wall_climate_median":  wall.get("wall_climate_median"),
        "orientation_match":    orientation_match,
        "section_row":          sector_row,
        "section_col":          sector_col,
        "mean_height_m":        mean_h,
        "min_height_m":         min_h,
        "max_height_m":         max_h,
        "dist_to_top_edge_median":  wall.get("dist_to_top_edge_median"),
        "dist_to_side_edge_median": wall.get("dist_to_side_edge_median"),
        "side_edge_label":          wall.get("side_edge_label"),
        "sector_climate_median":    sector_climate,
        "colonial":                 needs.get("colonial"),
        "colony_size_local_min":    needs.get("colony_size_local_min"),
        "colony_size_local_max":    needs.get("colony_size_local_max"),
        "noise_level":              needs.get("noise_level"),
        "human_tolerance_level":    needs.get("human_tolerance_level"),
        "dirt":                     needs.get("dirt"),
        "taxa":                     needs.get("taxa"),
        "is_day_active":            needs.get("is_day_active"),
        "is_evening_active":        needs.get("is_evening_active"),
        "is_dusk_active":           needs.get("is_dusk_active"),
        "nest_use_start_month":     needs.get("nest_use_start_month"),
        "nest_use_end_month":       needs.get("nest_use_end_month"),
        "nest_use_duration":        needs.get("nest_use_duration"),
        "preferred_height_min_m":   needs.get("preferred_height_min_m"),
        "preferred_height_max_m":   needs.get("preferred_height_max_m"),
        "prefers_edges_proximity":  needs.get("prefers_edges_proximity"),
        "prefers_roof_proximity":   needs.get("prefers_roof_proximity"),
        "nest_temp_min_c":          needs.get("nest_temp_min_c"),
        "nest_temp_max_c":          needs.get("nest_temp_max_c"),
        "preferred_orientation":    preferred_raw,
        "avoided_orientation":      needs.get("avoided_orientation"),
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
    rel_h = float(xyz[2]) - wall_ground_z
 
    ori = (wall.get("orientation") or "").strip().upper()
    preferred_raw = (needs.get("preferred_orientation") or "")
    preferred = [o.strip().upper() for o in str(preferred_raw).split(",") if o.strip()]
    orientation_match = 1 if (ori and ori in preferred) else 0
 
    grid = wall.get("grid") or {}
    zs_all = [
        float(pdata["point_on_wall"][2])
        for pdata in grid.values()
        if pdata.get("point_on_wall")
    ]
    max_z = max(zs_all) if zs_all else float(xyz[2])
    dist_to_top = (max_z - float(xyz[2]))
 
    row = {
        "point_id":   pid,
        "wall_id":    wall_id,
        "wall_free_area_m2":    wall.get("wall_free_area_m2"),
        "door_count":           wall.get("door_count"),
        "wall_shape":           wall.get("wall_shape"),
        "orientation":          ori or None,
        "floor_function":       wall.get("floor_function"),
        "neighbor_umin":        wall.get("neighbor_umin"),
        "neighbor_umax":        wall.get("neighbor_umax"),
        "building_function":    wall.get("building_function"),
        "wall_climate_median":  wall.get("wall_climate_median"),
        "orientation_match":    orientation_match,
        "section_row":          sector_row,
        "section_col":          sector_col,
        "mean_height_m":        rel_h,
        "min_height_m":         rel_h,
        "max_height_m":         rel_h,
        "dist_to_top_edge_median":  dist_to_top,
        "dist_to_side_edge_median": wall.get("dist_to_side_edge_median"),
        "side_edge_label":          wall.get("side_edge_label"),
        "sector_climate_median":    wall.get("wall_climate_median"),
        "colonial":                 needs.get("colonial"),
        "colony_size_local_min":    needs.get("colony_size_local_min"),
        "colony_size_local_max":    needs.get("colony_size_local_max"),
        "noise_level":              needs.get("noise_level"),
        "human_tolerance_level":    needs.get("human_tolerance_level"),
        "dirt":                     needs.get("dirt"),
        "taxa":                     needs.get("taxa"),
        "is_day_active":            needs.get("is_day_active"),
        "is_evening_active":        needs.get("is_evening_active"),
        "is_dusk_active":           needs.get("is_dusk_active"),
        "nest_use_start_month":     needs.get("nest_use_start_month"),
        "nest_use_end_month":       needs.get("nest_use_end_month"),
        "nest_use_duration":        needs.get("nest_use_duration"),
        "preferred_height_min_m":   needs.get("preferred_height_min_m"),
        "preferred_height_max_m":   needs.get("preferred_height_max_m"),
        "prefers_edges_proximity":  needs.get("prefers_edges_proximity"),
        "prefers_roof_proximity":   needs.get("prefers_roof_proximity"),
        "nest_temp_min_c":          needs.get("nest_temp_min_c"),
        "nest_temp_max_c":          needs.get("nest_temp_max_c"),
        "preferred_orientation":    preferred_raw,
        "avoided_orientation":      needs.get("avoided_orientation"),
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
        "wall_free_area_m2", "wall_climate_median", "sector_climate_median",
        "mean_height_m", "min_height_m", "max_height_m",
        "dist_to_top_edge_median", "dist_to_side_edge_median",
        "nest_temp_min_c", "nest_temp_max_c",
        "preferred_height_min_m", "preferred_height_max_m",
        "colony_size_local_min", "colony_size_local_max",
        "orientation_match",
        "door_count", "colonial",
        "noise_level", "human_tolerance_level", "dirt",
        "is_day_active", "is_evening_active", "is_dusk_active",
        "nest_use_start_month", "nest_use_end_month", "nest_use_duration",
        "prefers_edges_proximity", "prefers_roof_proximity",
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
                building_zero_z=building_zero_z,
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
 
    usable_geom, _ = _wall_usable_geometry(wall)
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
            building_zero_z=building_zero_z,
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
    print(f"  Building zero Z: {building_zero_z:.3f}  "
          f"height range: {height_min}–{height_max} m above building zero")
 
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
 
    # collect the top target_walls ranked walls that can actually produce a placement
    viable_walls: List[str] = []
    for wid in ranked_wall_ids:
        if len(viable_walls) >= target_walls:
            break
        # quick feasibility check — does this wall have any feasible sectors?
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
            viable_walls.append(wid)
 
    if not viable_walls:
        raise ValueError("Planning failed: no viable walls found.")
 
    # STEP 3 + 4: build two options
    # Option 1 — sector_rank 0 (best sector) on each viable wall
    # Option 2 — sector_rank 1 (second best sector, with fallback) on each viable wall
    raw_options = []
    # tracks Option 1 point ids per wall so Option 2 can avoid duplicates
    opt1_ids_by_wall: Dict[str, set] = {}
 
    for sector_rank in [0, 1]:
        placements = []
        for wid in viable_walls:
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
                if sector_rank == 0:
                    opt1_ids_by_wall[wid] = set(result.get("selected_point_ids") or [])
                print(f"  Option {sector_rank+1} wall={wid} "
                      f"sector=({result['section_row']},{result['section_col']}) "
                      f"colony_size={result['colony_size']} "
                      f"placement_score={result['placement_score']}")
            else:
                print(f"  Option {sector_rank+1} wall={wid}: placement failed — skipped.")
 
        if placements:
            best_idx = int(np.argmax([p["placement_score"] for p in placements]))
            raw_options.append({
                "placements":        placements,
                "best_placement_idx": best_idx,
            })
 
    if not raw_options:
        raise ValueError("Planning failed: no valid placement found on any wall.")
 
    # for single-orientation species re-rank so higher placement_score = Option 1
    if target_walls == 1 and len(raw_options) == 2:
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