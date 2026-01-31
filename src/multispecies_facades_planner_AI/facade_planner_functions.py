import json
from pathlib import Path
from typing import List, Sequence
import math
import random
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import statistics
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, Any, Optional

def load_building_dict(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygon_area_2d(poly_uv: Sequence[Sequence[float]]) -> float:
    """
    Shoelace area for a 2D polygon given as [[u, v], ...].
    Returns absolute area.
    """
    if not poly_uv or len(poly_uv) < 3:
        return 0.0

    area2 = 0.0
    n = len(poly_uv)
    for i in range(n):
        x1, y1 = poly_uv[i]
        x2, y2 = poly_uv[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5

def free_wall_area(wall: dict) -> float:
    wall_area = polygon_area_2d(wall.get("boundary_uv", []))
    doors = wall.get("doors", {}) or {}
    door_area = sum(
        polygon_area_2d(door.get("hull_uv", []))
        for door in doors.values()
    )
    windows = wall.get("windows", {}) or {}
    window_area = sum(
        polygon_area_2d(win.get("hull_uv", []))
        for win in windows.values()
    )
    free_area = wall_area - door_area - window_area
    return max(0.0, free_area)  


def wall_with_max_free_area(building_dict: dict):
    """
    Returns (wall_id, max_free_area).
    If no walls found, returns (None, None).
    """
    biggest_wall_id = None
    biggest_area = None

    for wall_id, wall in building_dict.items():
        area = free_wall_area(wall)
        if biggest_area is None or area > biggest_area:
            biggest_area = area
            biggest_wall_id = wall_id

    return biggest_wall_id, biggest_area

def derive_openings(wall: dict):
    """
    Returns a Shapely geometry representing wall area minus windows AND doors.
    Result may be Polygon or MultiPolygon.
    """
    wall_uv = wall.get("boundary_uv")
    if not wall_uv or len(wall_uv) < 3:
        return Polygon()  # empty

    wall_poly = Polygon(wall_uv)

    opening_polys = []

    # windows
    for win in (wall.get("windows") or {}).values():
        uv = win.get("hull_uv")
        if uv and len(uv) >= 3:
            opening_polys.append(Polygon(uv))

    # doors
    for dr in (wall.get("doors") or {}).values():
        uv = dr.get("hull_uv")
        if uv and len(uv) >= 3:
            opening_polys.append(Polygon(uv))

    if not opening_polys:
        return wall_poly

    holes = unary_union(opening_polys)
    return wall_poly.difference(holes)


def build_offset_area(
    wall: dict,
    win_side_offset: float = 0.2,
    door_side_offset: float = 0.4,
    join_style: int = 2,
):
    wall_uv = wall.get("boundary_uv")
    if not wall_uv or len(wall_uv) < 3:
        return Polygon()

    wall_poly = Polygon(wall_uv)
    if wall_poly.is_empty:
        return wall_poly

    min_u, min_v, max_u, max_v = wall_poly.bounds

    def v_axis_points_up(wall: dict) -> bool:
        """
        Returns True if +V corresponds to +worldZ (up).
        Expects something like:
          wall["plane"]["yaxis"] = (x,y,z)
        Adapt this accessor to your actual stored structure.
        """
        pl = wall.get("plane") or {}
        y = pl.get("yaxis") or pl.get("YAxis")  # be flexible
        if not y or len(y) != 3:
            # Fallback: assume the old behavior (+V is up)
            return True
        # dot(yaxis, worldZ) > 0 means v increases upward
        return (y[2] > 0)

    v_up = v_axis_points_up(wall)

    def opening_offset_geom(opening_uv, side_offset):
        if not opening_uv or len(opening_uv) < 3:
            return None

        op = Polygon(opening_uv)
        if op.is_empty:
            return None
        if not op.is_valid:
            op = op.buffer(0)
        if op.is_empty:
            return None

        op_min_u, op_min_v, op_max_u, op_max_v = op.bounds

        # 1) keep your existing side-offset band
        band = op.buffer(side_offset, join_style=join_style)

        # 2) ABOVE strip: pick the correct "top" direction in V
        opening_top_v = op_max_v if v_up else op_min_v
        wall_top_v    = max_v    if v_up else min_v

        # If opening is already at/above top (or numerical weirdness), skip above strip safely
        if (v_up and opening_top_v >= wall_top_v) or ((not v_up) and opening_top_v <= wall_top_v):
            above_strip = Polygon()
        else:
            # Build strip from opening_top_v toward wall_top_v
            v0, v1 = sorted([opening_top_v, wall_top_v])
            above_strip = box(
                op_min_u - side_offset,
                v0,
                op_max_u + side_offset,
                v1
            )

        band_clipped  = band.intersection(wall_poly)
        above_clipped = above_strip.intersection(wall_poly)

        return band_clipped.union(above_clipped)

    pieces = []

    for win in (wall.get("windows") or {}).values():
        g = opening_offset_geom(win.get("hull_uv"), win_side_offset)
        if g and not g.is_empty:
            pieces.append(g)

    for dr in (wall.get("doors") or {}).values():
        g = opening_offset_geom(dr.get("hull_uv"), door_side_offset)
        if g and not g.is_empty:
            pieces.append(g)

    if not pieces:
        return Polygon()

    out = unary_union(pieces)
    if not out.is_valid:
        out = out.buffer(0)
    return out

def evaluate_climate_median(wall: dict):
    """
    Returns the median of climate values for one wall.
    """
    values = [
        pt["climate"]
        for pt in wall.get("grid", {}).values()
        if "climate" in pt and pt["climate"] is not None
    ]

    if not values:
        return None  # or float("nan")

    return float(statistics.median(values))

def wall_hot_climate_median(building_dict: dict):
    """
    Returns (wall_id, max_median_value).
    If no valid medians exist, returns (None, None).
    """
    best_wall_id = None
    best_median = None

    for wall_id, wall in building_dict.items():
        median = evaluate_climate_median(wall)
        if median is None:
            continue

        if best_median is None or median > best_median:
            best_median = median
            best_wall_id = wall_id

    return best_wall_id, best_median


@dataclass
class ScoreResult:
    score: float
    reasons: List[Tuple[str, float]]  # (rule_name, delta)
    meta: Dict

def evaluate_for_colony(building_dict: dict, wall_id: str) -> ScoreResult:
    wall = building_dict.get(wall_id)

    # unsuitable if not wall
    if not wall:
        return ScoreResult(
            score=0.0,
            reasons=[("unsuitable", -1)],
            meta={"unsuitable": True}
        )

    reasons = []
    score = 0.0

    hottest_wall_id, hottest_median = wall_hot_climate_median(building_dict)
    biggest_wall_id, biggest_free_area = wall_with_max_free_area(building_dict)

    #Rule 1: context / adjacency
    floor_fn = wall.get("floor_function")
    if floor_fn == "neigbor_building":
        return ScoreResult(
            score=0.0,
            reasons=[("unsuitable_neighbor_building", -1)],
            meta={
                "unsuitable": True,
                "floor_fn": floor_fn,
            },
        )

    if floor_fn == "pedestrian":
        score -= 2
        reasons.append(("floor_function=pedestrian", -2))
    elif floor_fn == "garden":
        score += 5
        reasons.append(("floor_function=garden", +5))

    #Rule 2: free area
    wall_free = free_wall_area(wall)

    if biggest_free_area and biggest_free_area > 0:
        rel = wall_free / biggest_free_area  # 0..1+
        delta = (rel - 0.5) * 2               # ~ -1 .. +1
        score += delta
        reasons.append(("free_area_relative", delta))

    if wall_free <= 3:
        score -= 3
        reasons.append(("free_area<=3", -3))

    #Rule 3: doors 
    door_count = len(wall.get("doors") or {})
    if door_count == 0:
        score += 1
        reasons.append(("door_count=0", +1))
    elif door_count > 2:
        score -= 1
        reasons.append(("door_count>2", -1))

    # Rule 4: hottest wall penalty 
    if hottest_wall_id == wall_id:
        score -= 3
        reasons.append(("hottest_wall_penalty", -3))

    # out-
    return ScoreResult(
        score=score,
        reasons=reasons,
        meta={
            "unsuitable": False,
            "wall_free": wall_free,
            "biggest_free_area": biggest_free_area,
            "hottest_median": hottest_median,
            "floor_fn": floor_fn,
            "door_count": door_count,
        },
    )


def parse_range_int(val, default=(0, 0)):
    if val is None:
        return int(default[0]), int(default[1])

    if isinstance(val, (list, tuple)) and len(val) == 2:
        return int(val[0]), int(val[1])

    s = str(val).strip()
    if not s:
        return int(default[0]), int(default[1])

    s2 = s.replace(" ", "")

    def _split_range(text):
        t = text.replace("–", "-").replace("—", "-").replace("to", "-").replace(",", "-").replace(";", "-")
        parts = [p.strip() for p in t.split("-") if p.strip()]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
        n = int(parts[0])
        return n, n

    if s2.startswith(">="):
        rest = s2[2:]
        return _split_range(rest)

    if s2.startswith("<="):
        rest = s2[2:]
        return _split_range(rest)

    if s2.startswith(">"):
        rest = s2[1:]
        return _split_range(rest)

    if s2.startswith("<"):
        rest = s2[1:]
        return _split_range(rest)

    if any(sep in s for sep in ["-", "to", ",", ";", "–", "—"]):
        return _split_range(s)

    n = int(s2)
    return n, n

def parse_range_float(s: str, default=(0.5, 1.0)):
    """'50-100' -> (50.0,100.0)."""
    if not s:
        return default
    s = str(s).replace('"', '').strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return float(a.strip()), float(b.strip())
    v = float(s.strip())
    return v, v

def parse_min_height_m(s: str, default=0.0):
    """
    Supports forms like '> 4', '>=4', '4', '2.5-4'.
    Returns minimum height in meters (float).
    """
    if not s:
        return float(default)

    s = str(s).replace('"', '').strip()
    s = s.replace("≥", ">=")

    if s.startswith(">="):
        return float(s[2:].strip())
    if s.startswith(">"):
        return float(s[1:].strip())

    # range like "2.5-4"
    if "-" in s:
        try:
            lo, _ = s.split("-", 1)
            return float(lo.strip())
        except ValueError:
            return float(default)

    # fallback: treat as numeric
    try:
        return float(s)
    except ValueError:
        return float(default)
    
def parse_max_height_m(s: str):
    """
    Supports forms like '2.5-4', '<4', '<=4'.
    Returns maximum height in meters (float) or None if not applicable.
    """
    if not s:
        return None

    s = str(s).replace('"', '').strip()
    s = s.replace("≤", "<=")

    # range like "2.5-4"
    if "-" in s:
        try:
            _, hi = s.split("-", 1)
            return float(hi.strip())
        except ValueError:
            return None

    # upper bound only
    if s.startswith("<="):
        return float(s[2:].strip())
    if s.startswith("<"):
        return float(s[1:].strip())

    # no max constraint (e.g. ">4", "4")
    return None


def dist_uv(a, b):
    du = a[0] - b[0]
    dv = a[1] - b[1]
    return math.sqrt(du*du + dv*dv)

def pick_points_chained_band(
    candidates,
    target_size,
    dmin_m,
    dmax_m,
    max_tries_per_step=400,
):
    """
    candidates: list of (pid, uv, xyz)
    Rules:
      - First point: random from candidates
      - Next point: distance to previous point in [dmin_m, dmax_m]
      - And distance to ALL earlier points >= dmin_m
    Returns list of selected (pid, uv, xyz)
    """

    if not candidates or target_size <= 0:
        return []

    # index for random sampling
    cand = list(candidates)

    # 1) pick first point randomly
    first = random.choice(cand)
    selected = [first]
    selected_uvs = [first[1]]

    # remove it
    cand = [c for c in cand if c[0] != first[0]]

    # 2) grow the chain
    while len(selected) < target_size and cand:
        prev_uv = selected_uvs[-1]

        # try random candidates until one fits the rules
        picked = None
        for _ in range(max_tries_per_step):
            c = random.choice(cand)
            uv = c[1]

            d_prev = dist_uv(uv, prev_uv)
            if d_prev < dmin_m or d_prev > dmax_m:
                continue

            # not closer than dmin to ANY already selected point
            too_close = False
            for suv in selected_uvs:
                if dist_uv(uv, suv) < dmin_m:
                    too_close = True
                    break
            if too_close:
                continue

            picked = c
            break
        if picked is None:
            # cannot find next point from current chain end -> stop
            break

        selected.append(picked)
        selected_uvs.append(picked[1])
        cand.remove(picked)

    return selected  

def _vec_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _vec_mul(a, s: float):
    return (a[0]*s, a[1]*s, a[2]*s)

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def plane_uv_to_xyz(plane: Any, u: float, v: float) -> Tuple[float, float, float]:
    """
    Supports:
    A) plane as dict: {"origin":[x,y,z], "xaxis":[x,y,z], "yaxis":[x,y,z]}
    B) Rhino Plane-like object: plane.Origin, plane.XAxis, plane.YAxis
    """
    if isinstance(plane, dict):
        o = tuple(plane["origin"])
        x = tuple(plane["xaxis"])
        y = tuple(plane["yaxis"])
    else:
        # Rhino.Geometry.Plane style
        o = (plane.OriginX, plane.OriginY, plane.OriginZ) if hasattr(plane, "OriginX") else (plane.Origin.X, plane.Origin.Y, plane.Origin.Z)
        x = (plane.XAxis.X, plane.XAxis.Y, plane.XAxis.Z)
        y = (plane.YAxis.X, plane.YAxis.Y, plane.YAxis.Z)

    return _vec_add(o, _vec_add(_vec_mul(x, u), _vec_mul(y, v)))

def boundary_uv_to_xyz(wall: dict) -> Optional[List[Tuple[float, float, float]]]:
    plane = wall.get("plane")
    buv = wall.get("boundary_uv")
    if plane is None or not buv or len(buv) < 3:
        return None
    return [plane_uv_to_xyz(plane, uv[0], uv[1]) for uv in buv]


# ---------------------------------------
# Extract "vertical" edge segments in 3D
# ---------------------------------------

def _u_bounds(boundary_uv: List[List[float]]) -> Tuple[float, float]:
    us = [p[0] for p in boundary_uv]
    return (min(us), max(us))

def _pick_vertical_edge_endpoints(
    boundary_uv: List[List[float]],
    boundary_xyz: List[Tuple[float, float, float]],
    which: str,
    u_snap_tol: float = 0.02,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    which: "umin" or "umax"
    We collect boundary points whose u is near the chosen u-bound,
    then take min-v and max-v among them and return their XYZ endpoints.
    """
    if not boundary_uv or not boundary_xyz or len(boundary_uv) != len(boundary_xyz):
        return None

    umin, umax = _u_bounds(boundary_uv)
    u_target = umin if which == "umin" else umax

    idx = [i for i, uv in enumerate(boundary_uv) if abs(uv[0] - u_target) <= u_snap_tol]
    if len(idx) < 2:
        return None

    # choose the "lowest" and "highest" in v among those points
    idx_sorted = sorted(idx, key=lambda i: boundary_uv[i][1])
    i0, i1 = idx_sorted[0], idx_sorted[-1]
    return (boundary_xyz[i0], boundary_xyz[i1])

def vertical_edges_xyz(wall: dict, u_snap_tol: float = 0.02):
    """
    Returns two vertical-ish edge endpoint pairs in XYZ: [((p0,p1), "umin"), ((p0,p1), "umax")]
    """
    buv = wall.get("boundary_uv")
    if not buv or len(buv) < 3:
        return []

    bxyz = boundary_uv_to_xyz(wall)
    if not bxyz:
        return []

    e1 = _pick_vertical_edge_endpoints(buv, bxyz, "umin", u_snap_tol=u_snap_tol)
    e2 = _pick_vertical_edge_endpoints(buv, bxyz, "umax", u_snap_tol=u_snap_tol)

    out = []
    if e1: out.append((e1, "umin"))
    if e2: out.append((e2, "umax"))
    return out


# ---------------------------------------
# Main: find neighbor floor functions
# ---------------------------------------

def neighbor_floor_functions(
    building_dict: Dict[str, dict],
    wall_id: str,
    *,
    edge_match_tol_m: float = 0.25,   # 20–30 cm is often good
    u_snap_tol: float = 0.02,         # UV snapping tolerance
) -> List[Dict[str, Any]]:
    """
    Returns a list of neighbor walls that share a vertical edge with wall_id and have floor_function.
    Each item: {"neighbor_id":..., "floor_function":..., "matched_side":..., "neighbor_side":...}
    """

    wall = building_dict.get(wall_id)
    if not isinstance(wall, dict):
        return []

    target_edges = vertical_edges_xyz(wall, u_snap_tol=u_snap_tol)
    if not target_edges:
        return []

    results = []

    for other_id, other_wall in building_dict.items():
        if other_id == wall_id:
            continue
        if not isinstance(other_wall, dict) or other_id.startswith("_"):
            continue

        # ignore walls without floor_function entirely (your rule)
        ff = other_wall.get("floor_function", None)
        if not ff:
            continue

        other_edges = vertical_edges_xyz(other_wall, u_snap_tol=u_snap_tol)
        if not other_edges:
            continue

        # Compare each vertical edge pair: endpoints can be swapped
        for (p0, p1), side in target_edges:
            for (q0, q1), qside in other_edges:
                d_direct = max(_dist(p0, q0), _dist(p1, q1))
                d_swap   = max(_dist(p0, q1), _dist(p1, q0))

                if min(d_direct, d_swap) <= edge_match_tol_m:
                    results.append({
                        "neighbor_id": other_id,
                        "floor_function": ff,
                        "matched_side": side,        # which side of target wall (umin/umax)
                        "neighbor_side": qside,      # which side of neighbor wall matched
                        "match_error_m": min(d_direct, d_swap),
                    })

    # de-duplicate by neighbor_id (keep best match)
    best = {}
    for r in results:
        nid = r["neighbor_id"]
        if nid not in best or r["match_error_m"] < best[nid]["match_error_m"]:
            best[nid] = r

    return sorted(best.values(), key=lambda x: x["match_error_m"])


def wall_uv_bbox_from_building(building_dict: dict, wall_id: str) -> Optional[Tuple[float, float, float, float]]:
    wall = building_dict.get(wall_id)
    if not isinstance(wall, dict):
        return None
    buv = wall.get("boundary_uv") or []
    if len(buv) < 3:
        return None
    us = [p[0] for p in buv]
    vs = [p[1] for p in buv]
    return (min(us), min(vs), max(us), max(vs))

def candidate_mean_uv(candidate: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    cuv = candidate.get("uv")
    if not cuv or len(cuv) == 0:
        return None
    u = float(np.mean([p[0] for p in cuv]))
    v = float(np.mean([p[1] for p in cuv]))
    return (u, v)

def sector_3x3(u: float, v: float, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int]:
    umin, vmin, umax, vmax = bbox
    du = max(umax - umin, 1e-9)
    dv = max(vmax - vmin, 1e-9)

    un = (u - umin) / du
    vn = (v - vmin) / dv

    # clamp to [0, 1)
    un = min(max(un, 0.0), 0.999999)
    vn = min(max(vn, 0.0), 0.999999)

    col = int(un * 3)          # 0..2
    row = int(vn * 3)          # 0..2
    sid = row * 3 + col        # 0..8
    return row, col, sid

def sector_features_3x3(building_dict: dict, candidate: Dict[str, Any]) -> Dict[str, Any]:
    wall_id = candidate["wall_id"]
    bbox = wall_uv_bbox_from_building(building_dict, wall_id)
    uv = candidate_mean_uv(candidate)

    if bbox is None or uv is None:
        return {"section_row": np.nan, "section_col": np.nan, "section_id": np.nan}

    row, col, sid = sector_3x3(uv[0], uv[1], bbox)
    return {"section_row": int(row), "section_col": int(col), "section_id": int(sid)}