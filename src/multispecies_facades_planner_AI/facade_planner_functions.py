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
import re 

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


def wall_height_m(wall: dict) -> Optional[float]:
    """
    Returns the wall's own vertical extent (max_z - min_z), in metres — e.g.
    10 or 20m. This is relative to the wall itself, not the building's
    absolute world Z coordinates (which can be arbitrary, e.g. 550m), so a
    model can learn "placed low *for this wall's height*" separately from
    "the wall itself is short".

    For triangular-top (gable) walls, the sampled grid can fall short of the
    actual roof apex, so the highest point is taken from the wall's mesh
    vertices instead — that's the true geometry, not just where nest points
    happen to be sampled.
    """
    grid = wall.get("grid") or {}
    zs = [
        float(pdata["point_on_wall"][2])
        for pdata in grid.values()
        if isinstance(pdata, dict) and pdata.get("point_on_wall")
    ]
    if not zs:
        return None

    z_min = min(zs)
    z_max = max(zs)

    wall_shape = str(wall.get("wall_shape") or "").strip().lower()
    if "triangular" in wall_shape:
        mesh_verts = (wall.get("mesh") or {}).get("vertices") or []
        mesh_zs = [float(v[2]) for v in mesh_verts if len(v) > 2]
        if mesh_zs:
            z_max = max(z_max, max(mesh_zs))

    return float(z_max - z_min)


def wall_with_max_free_area(building_dict: dict):
    """
    Returns (wall_id, max_free_area).
    If no walls found, returns (None, None).
    """
    biggest_wall_id = None
    biggest_area = None

    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict):
            continue
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

def _window_v_extent(hull_uv):
    if not hull_uv or len(hull_uv) < 3:
        return None
    vs = [p[1] for p in hull_uv]
    return min(vs), max(vs)


def _group_window_floors(windows: dict, v_up: bool, floor_gap_tol_m: float = 1.0):
    """
    Clusters windows on one wall into floor bands by vertical (V) position,
    ordered bottom -> top. A window joins the previous floor band if its
    V-center is within floor_gap_tol_m of it — same-floor windows share
    ~the same sill height, while floor-to-floor spacing is a few metres, so
    a 1m tolerance cleanly separates floors without needing real floor data.

    Returns a list of floor bands: [{"window_ids": [...], "tops": [...]}]
    "tops" holds each window's upper-edge V coordinate (op_max_v if v_up
    else op_min_v) — used to anchor the no-nest strip on the floor below at
    the actual top line of the windows above it, not a height offset.
    """
    entries = []
    for win_id, win in (windows or {}).items():
        ext = _window_v_extent(win.get("hull_uv"))
        if ext is None:
            continue
        v_min, v_max = ext
        v_center = (v_min + v_max) / 2.0
        order_key = v_center if v_up else -v_center
        top_v = v_max if v_up else v_min
        entries.append((order_key, win_id, top_v))

    entries.sort(key=lambda e: e[0])

    floors = []
    for order_key, win_id, top_v in entries:
        if floors and (order_key - floors[-1]["_order_key"]) <= floor_gap_tol_m:
            floor = floors[-1]
            floor["window_ids"].append(win_id)
            floor["tops"].append(top_v)
            floor["_order_key"] = order_key
        else:
            floors.append({
                "window_ids": [win_id],
                "tops": [top_v],
                "_order_key": order_key,
            })

    return floors


def _window_floor_index(floors, win_id):
    for i, floor in enumerate(floors):
        if win_id in floor["window_ids"]:
            return i
    return None


def v_axis_points_up(wall: dict) -> bool:
    """
    True if the wall's local +V axis corresponds to +worldZ (up).

    Roughly half of all walls store plane.yaxis = (x, y, -1), i.e. V increases
    *downwards* — opposite facades of a building get opposite parametrisation.
    Anything that reasons about "up", "top" or "the roofline" in UV space must
    consult this first, or it silently operates on the bottom of the wall.

    Falls back to True when no usable yaxis is stored.
    """
    pl = wall.get("plane") or {}
    y = pl.get("yaxis") or pl.get("YAxis")  # be flexible
    if not y or len(y) != 3:
        return True
    # dot(yaxis, worldZ) > 0 means v increases upward
    return bool(y[2] > 0)


WIN_SIDE_OFFSET_EDGE_M = 0.35     # window at the end of a floor band
WIN_SIDE_OFFSET_BETWEEN_M = 0.85  # window with neighbours on both sides
WIN_ABOVE_SIDE_OFFSET_M = 0.20    # sideways reach of the strip ABOVE a window
                                  # that has nothing above it — the same on every
                                  # such window, edge or between


def build_offset_area(
    wall: dict,
    win_side_offset: float = WIN_SIDE_OFFSET_EDGE_M,
    door_side_offset: float = 0.4,
    join_style: int = 2,
    floor_gap_tol_m: float = 1.0,
    colonial: bool = False,
    win_side_offset_between: float = WIN_SIDE_OFFSET_BETWEEN_M,
    win_above_side_offset: float = WIN_ABOVE_SIDE_OFFSET_M,
):
    """
    Hard-constraint no-nest area around windows and doors.

    Windows (all species):
      - side_offset (both sides) = win_side_offset (35cm by default).
      - the exclusion strip above a window normally only runs up to the top
        edge (V-coordinate) of the window directly above it on the next
        floor up (windows are clustered into floor bands by vertical
        position, see _group_window_floors) — not all the way to the roof.
      - for windows on the topmost floor band (no floor above), the strip
        runs all the way up to the wall's top boundary (the roof).

    colonial=True additionally applies the colony-species rules:
      - a window at either END of its floor band (nothing between it and the
        wall edge on that floor) keeps the narrow win_side_offset (0.35 m).
      - a window with neighbours on BOTH sides gets win_side_offset_between
        (0.80 m), because the pier it shares is flanked by two windows.
      - for a window with nothing above it in its own column, the strip running
        up to the roof reaches win_above_side_offset (0.20 m) to each side —
        the same for every such window, edge or between. The wider side band is
        cut off at that window's top edge so it cannot bleed into this region.

    Solitary species keep the flat 0.35 m behaviour (colonial defaults False),
    so nothing changes for them.
    """
    wall_uv = wall.get("boundary_uv")
    if not wall_uv or len(wall_uv) < 3:
        return Polygon()

    wall_poly = Polygon(wall_uv)
    if wall_poly.is_empty:
        return wall_poly

    min_u, min_v, max_u, max_v = wall_poly.bounds

    v_up = v_axis_points_up(wall)

    windows = wall.get("windows") or {}
    window_floors = _group_window_floors(windows, v_up, floor_gap_tol_m=floor_gap_tol_m)

    def opening_offset_geom(opening_uv, side_offset, next_floor_top_v=None,
                            above_side_offset=None, clip_band_at_top=False):
        # above_side_offset controls how far the "above" strip reaches sideways.
        # Defaults to side_offset; pass 0.0 to make the strip span only the
        # opening's own width (top-floor rule for colonial species).
        #
        # clip_band_at_top cuts the side band off at the opening's top edge. The
        # band is a buffer, so it otherwise also bulges side_offset ABOVE the
        # opening — which would widen the region between a top-floor window and
        # the roof. With this set, everything above the window top is exactly
        # the window's own width and nothing wider.
        if above_side_offset is None:
            above_side_offset = side_offset

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

        # cut the band at the opening's top edge so it cannot widen the area
        # above it (top-floor rule)
        if clip_band_at_top:
            pad = max(side_offset, 1.0) + 1.0
            keep = (
                box(min_u - pad, min_v - pad, max_u + pad, opening_top_v) if v_up
                else box(min_u - pad, opening_top_v, max_u + pad, max_v + pad)
            )
            band = band.intersection(keep)

        # If opening is already at/above top (or numerical weirdness), skip above strip safely
        if (v_up and opening_top_v >= wall_top_v) or ((not v_up) and opening_top_v <= wall_top_v):
            above_strip = Polygon()
        else:
            if next_floor_top_v is None:
                # top floor (or no floor above) -> exclusion runs to the roof
                strip_top_v = wall_top_v
            else:
                # exclusion runs up to the next floor window's own top edge,
                # clamped so it never falls short of this opening or past the wall
                strip_top_v = (
                    min(max(next_floor_top_v, opening_top_v), wall_top_v) if v_up
                    else max(min(next_floor_top_v, opening_top_v), wall_top_v)
                )

            v0, v1 = sorted([opening_top_v, strip_top_v])
            above_strip = box(
                op_min_u - above_side_offset,
                v0,
                op_max_u + above_side_offset,
                v1
            )

        band_clipped  = band.intersection(wall_poly)
        above_clipped = above_strip.intersection(wall_poly)

        return band_clipped.union(above_clipped)

    # Colony rule: which windows sit at the ends of their floor band? Those are
    # the ones with open wall between them and the wall edge, so they keep the
    # narrow offset. Everything else has a window on both sides.
    edge_window_ids = set()
    if colonial:
        for band in window_floors:
            spans = []
            for wid in band.get("window_ids") or []:
                hull = (windows.get(wid) or {}).get("hull_uv")
                if not hull or len(hull) < 3:
                    continue
                us = [p[0] for p in hull]
                spans.append((min(us), max(us), wid))
            if not spans:
                continue

            # Merge overlapping U spans into COLUMNS first. Several windows can
            # share one column — stacked panes, or floors close enough that the
            # banding tolerance groups them together. They sit at the same U, so
            # the union of their bands is what is actually seen: classifying them
            # individually let one window's 0.85 m swallow its neighbour's 0.35 m
            # and the outermost column stopped reading as an edge at all.
            spans.sort(key=lambda s: s[0])
            columns = []
            cur_u0, cur_u1, ids = spans[0][0], spans[0][1], [spans[0][2]]
            for u0, u1, wid in spans[1:]:
                if u0 <= cur_u1 + 1e-9:          # overlaps the column so far
                    cur_u1 = max(cur_u1, u1)
                    ids.append(wid)
                else:
                    columns.append(ids)
                    cur_u0, cur_u1, ids = u0, u1, [wid]
            columns.append(ids)

            edge_window_ids.update(columns[0])    # leftmost column on this floor
            edge_window_ids.update(columns[-1])   # rightmost column on this floor

    # U span + band index per window, so "is anything above THIS window?" can be
    # answered per column. Membership of the global top band is not the same
    # question: on a stepped roofline, or wherever the top band does not span
    # the whole facade, a window can be the highest one in its own column while
    # sitting on a lower band.
    win_span = {}
    for wid, w in windows.items():
        hull = w.get("hull_uv")
        if not hull or len(hull) < 3:
            continue
        us = [p[0] for p in hull]
        win_span[wid] = (min(us), max(us), _window_floor_index(window_floors, wid))

    def windows_above(win_id):
        me = win_span.get(win_id)
        if me is None or me[2] is None:
            return []
        u0, u1, bi = me
        return [
            other for other, (o0, o1, oi) in win_span.items()
            if other != win_id and oi is not None and oi > bi
            and o1 > u0 + 1e-9 and o0 < u1 - 1e-9
        ]

    def top_v_of(win_id):
        vs = [p[1] for p in windows[win_id]["hull_uv"]]
        return max(vs) if v_up else min(vs)

    pieces = []

    for win_id, win in windows.items():
        floor_idx = _window_floor_index(window_floors, win_id)

        if colonial:
            # anchor on the windows genuinely above this one, in its own column
            above_ids = windows_above(win_id)
            next_floor_top_v = None
            if above_ids:
                tops_above = [top_v_of(o) for o in above_ids]
                # NEAREST one, so the widened strip cannot overshoot past it and
                # widen the area above a window that is itself a top window
                next_floor_top_v = min(tops_above) if v_up else max(tops_above)
            # "last floor" = nothing above it in its column
            is_top_floor = not above_ids
        else:
            # unchanged behaviour for solitary species
            next_floor_top_v = None
            if floor_idx is not None and floor_idx < len(window_floors) - 1:
                next_tops = window_floors[floor_idx + 1]["tops"]
                if next_tops:
                    # the outermost top line among the next floor's windows, so
                    # the strip never falls short of any of them
                    next_floor_top_v = max(next_tops) if v_up else min(next_tops)
            is_top_floor = False

        if colonial:
            side = (win_side_offset if win_id in edge_window_ids
                    else win_side_offset_between)
            # Nothing above it in its column: from its top edge up to the roof
            # the exclusion is the window's width plus a fixed 0.20 m each side,
            # regardless of whether it is an edge or between window. The wider
            # side band is clipped at that top edge so only the 0.20 m applies.
            above_side = win_above_side_offset if is_top_floor else side
        else:
            side = win_side_offset
            above_side = None

        g = opening_offset_geom(
            win.get("hull_uv"), side,
            next_floor_top_v=next_floor_top_v,
            above_side_offset=above_side,
            clip_band_at_top=is_top_floor,
        )
        if g and not g.is_empty:
            pieces.append(g)

    for dr in (wall.get("doors") or {}).values():
        g = opening_offset_geom(dr.get("hull_uv"), door_side_offset, next_floor_top_v=None)
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
        if not isinstance(wall, dict):
            continue
        median = evaluate_climate_median(wall)
        if median is None:
            continue

        if best_median is None or median > best_median:
            best_median = median
            best_wall_id = wall_id

    return best_wall_id, best_median

def _clean_numeric_text(val) -> str:
    """
    Normalize numeric text from Excel:
    '2,5-4' -> '2.5-4'
    '2,5 – 4' -> '2.5-4'
    """
    return (
        str(val)
        .replace('"', '')
        .replace("\xa0", " ")
        .replace(",", ".")
        .replace("–", "-")
        .replace("—", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .strip()
    )

def parse_min_max_numeric(val):
    """
    Extracts numeric min/max from values like:
    '2,5-4', '2.5-4', '> 4', '15 - 31'
    """
    if val is None:
        return None, None

    s = _clean_numeric_text(val)

    nums = re.findall(r"\d+(?:\.\d+)?", s)

    if not nums:
        return None, None

    nums = [float(n) for n in nums]

    return min(nums), max(nums)

def parse_min_max_count(val, default=(np.nan, np.nan)):
    if val is None:
        return default

    s = str(val).strip().lower().replace(" ", "")
    if not s:
        return default

    s = s.replace("–", "-").replace("—", "-")

    if s.startswith(">"):
        n = int(float(s.lstrip(">").lstrip("=")))
        return n, None

    if "-" in s:
        a, b = s.split("-", 1)
        return int(float(a)), int(float(b))

    n = int(float(s))
    return n, n

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
    """
    Supports:
    '50-100', '0,5-1', '0.5-1'
    """
    if s is None or str(s).strip() == "":
        return default

    s = _clean_numeric_text(s)

    if "-" in s:
        a, b = s.split("-", 1)
        return float(a.strip()), float(b.strip())

    v = float(s.strip())
    return v, v

def parse_min_height_m(s: str, default=0.0):
    """
    Supports forms like:
    '> 4', '>=4', '4', '2.5-4', '2,5-4'
    Returns minimum height in meters.
    """
    if s is None or str(s).strip() == "":
        return float(default)

    s = _clean_numeric_text(s)

    if s.startswith(">="):
        return float(s[2:].strip())

    if s.startswith(">"):
        return float(s[1:].strip())

    if "-" in s:
        try:
            lo, _ = s.split("-", 1)
            return float(lo.strip())
        except ValueError:
            return float(default)

    try:
        return float(s)
    except ValueError:
        return float(default)
    
def parse_max_height_m(s: str):
    """
    Supports forms like:
    '2.5-4', '2,5-4', '<4', '<=4'
    Returns maximum height in meters, or None if no max constraint.
    """
    if s is None or str(s).strip() == "":
        return None

    s = _clean_numeric_text(s)

    if "-" in s:
        try:
            _, hi = s.split("-", 1)
            return float(hi.strip())
        except ValueError:
            return None

    if s.startswith("<="):
        return float(s[2:].strip())

    if s.startswith("<"):
        return float(s[1:].strip())

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
    rng=None,
):
    """
    candidates: list of (pid, uv, xyz)
    Rules:
      - First point: random from candidates
      - Next point: distance to previous point in [dmin_m, dmax_m]
      - And distance to ALL earlier points >= dmin_m
    Returns list of selected (pid, uv, xyz)

    rng: pass the caller's seeded random.Random so the placement is
    reproducible. This function decides where the nests actually go, so if it
    draws from the global `random` module instead, the seed threaded through
    the whole generator has no effect on the output. Defaults to the global
    module only to preserve behaviour for unseeded callers.
    """

    if not candidates or target_size <= 0:
        return []

    rng = rng if rng is not None else random

    # index for random sampling
    cand = list(candidates)

    # 1) pick first point randomly
    first = rng.choice(cand)
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
            c = rng.choice(cand)
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


def wall_grid_points(wall: dict):
    grid = wall.get("grid", {}) or {}
    pts = []

    for pdata in grid.values():
        if not isinstance(pdata, dict):
            continue

        uv = pdata.get("uv")
        xyz = pdata.get("point_on_wall")

        if not uv or not xyz:
            continue

        pts.append((
            float(uv[0]),
            float(uv[1]),
            (
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
            )
        ))

    return pts


def local_uv_side_edge_xyz(
    wall: dict,
    side: str,
    u_local_tol: float = 0.05,
):
    """
    Get approximate XYZ vertical edge from local UV side.

    side:
      'umin' = local left side
      'umax' = local right side
    """
    pts = wall_grid_points(wall)

    if not pts:
        return None

    us = [p[0] for p in pts]
    umin = min(us)
    umax = max(us)
    du = max(umax - umin, 1e-9)

    if side == "umin":
        side_pts = [
            xyz for u, v, xyz in pts
            if ((u - umin) / du) <= u_local_tol
        ]

    elif side == "umax":
        side_pts = [
            xyz for u, v, xyz in pts
            if ((u - umin) / du) >= (1.0 - u_local_tol)
        ]

    else:
        raise ValueError("side must be 'umin' or 'umax'.")

    if len(side_pts) < 2:
        return None

    bottom = min(side_pts, key=lambda p: p[2])
    top = max(side_pts, key=lambda p: p[2])

    return bottom, top


def edge_xy_midpoint(edge):
    p0, p1 = edge
    return (
        (p0[0] + p1[0]) / 2.0,
        (p0[1] + p1[1]) / 2.0,
    )


def edge_xy_distance(edge_a, edge_b):
    ax, ay = edge_xy_midpoint(edge_a)
    bx, by = edge_xy_midpoint(edge_b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def neighbor_floor_functions(
    building_dict: Dict[str, dict],
    wall_id: str,
    *,
    u_local_tol: float = 0.05,
    max_neighbor_distance_m: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns closest physical neighbor for local UV sides:
      - umin
      - umax

    Local UV defines the side.
    XYZ/XY geometry finds the closest neighboring wall.
    """

    wall = building_dict.get(wall_id)
    if not isinstance(wall, dict):
        return []

    target_edges = {
        "umin": local_uv_side_edge_xyz(
            wall,
            "umin",
            u_local_tol=u_local_tol,
        ),
        "umax": local_uv_side_edge_xyz(
            wall,
            "umax",
            u_local_tol=u_local_tol,
        ),
    }

    best_by_side = {
        "umin": None,
        "umax": None,
    }

    for side, target_edge in target_edges.items():
        if target_edge is None:
            continue

        for other_id, other_wall in building_dict.items():
            if other_id == wall_id:
                continue

            if not isinstance(other_wall, dict) or other_id.startswith("_"):
                continue

            ff = other_wall.get("floor_function")
            ff_norm = (
                str(ff).strip().lower()
                if ff is not None and str(ff).strip()
                else "none"
            )

            other_edges = {
                "umin": local_uv_side_edge_xyz(
                    other_wall,
                    "umin",
                    u_local_tol=u_local_tol,
                ),
                "umax": local_uv_side_edge_xyz(
                    other_wall,
                    "umax",
                    u_local_tol=u_local_tol,
                ),
            }

            for other_side, other_edge in other_edges.items():
                if other_edge is None:
                    continue

                dist = edge_xy_distance(target_edge, other_edge)

                if (
                    max_neighbor_distance_m is not None
                    and dist > max_neighbor_distance_m
                ):
                    continue

                candidate = {
                    "neighbor_id": other_id,
                    "floor_function": ff_norm,
                    "matched_side": side,
                    "neighbor_side": other_side,
                    "match_error_m": float(dist),
                }

                current = best_by_side[side]

                if current is None or dist < current["match_error_m"]:
                    best_by_side[side] = candidate

    return [
        best_by_side[side]
        for side in ["umin", "umax"]
        if best_by_side[side] is not None
    ]

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


def sector_3x3_labels(
    u: float,
    v: float,
    bbox: Tuple[float, float, float, float],
    v_up: bool = True,
) -> Tuple[str, str]:
    """
    Returns categorical sector labels:
      section_row: 'bottom' / 'middle' / 'top'
      section_col: 'left' / 'middle' / 'right'

    v_up says which V direction is physically up on this wall (see
    v_axis_points_up). On a wall whose V axis points down the normalised
    position is mirrored, otherwise a nest directly under the roof would be
    labelled 'bottom'.
    """
    umin, vmin, umax, vmax = bbox

    du = max(umax - umin, 1e-9)
    dv = max(vmax - vmin, 1e-9)

    un = (u - umin) / du
    vn = (v - vmin) / dv

    if not v_up:
        vn = 1.0 - vn

    # clamp to [0,1)
    un = min(max(un, 0.0), 0.999999)
    vn = min(max(vn, 0.0), 0.999999)

    col_idx = int(un * 3)   # 0..2
    row_idx = int(vn * 3)   # 0..2

    col_labels = ["left", "middle", "right"]
    row_labels = ["bottom", "middle", "top"]

    return row_labels[row_idx], col_labels[col_idx]


def sector_features_3x3(building_dict: dict, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns categorical 3x3 sector features.
    """
    wall_id = candidate["wall_id"]
    bbox = wall_uv_bbox_from_building(building_dict, wall_id)
    uv = candidate_mean_uv(candidate)

    if bbox is None or uv is None:
        return {
            "section_row": None,
            "section_col": None,
        }

    v_up = v_axis_points_up(building_dict.get(wall_id) or {})
    row_label, col_label = sector_3x3_labels(uv[0], uv[1], bbox, v_up=v_up)

    return {
        "section_row": row_label,
        "section_col": col_label,
    }


import statistics
from typing import Dict, Any


def precompute_wall_climate_features(building_dict: Dict[str, dict]) -> None:
    """
    Computes and stores wall-level and sector-level climate medians in-place.
    Grid point UV is read from pt["uv"] = [u, v] (world-scale UV space,
    same coordinate system as boundary_uv).
    """
    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue

        # ── whole-wall median ─────────────────────────────────────────────
        wall["wall_climate_median"] = evaluate_climate_median(wall)

        # ── sector medians ────────────────────────────────────────────────
        bbox = wall_uv_bbox_from_building(building_dict, wall_id)
        if bbox is None:
            wall["sector_climate_medians_3x3"] = {}
            continue

        sector_values: Dict[str, list] = {}
        v_up = v_axis_points_up(wall)

        for pt in (wall.get("grid") or {}).values():
            uv      = pt.get("uv")          # [u, v] in world-scale UV space
            climate = pt.get("climate")

            if not uv or len(uv) < 2 or climate is None:
                continue

            u, v = float(uv[0]), float(uv[1])

            row_label, col_label = sector_3x3_labels(u, v, bbox, v_up=v_up)
            key = f"{row_label}_{col_label}"

            sector_values.setdefault(key, []).append(climate)

        sector_medians = {}
        for key, values in sector_values.items():
            if values:
                sector_medians[key] = float(statistics.median(values))

        wall["sector_climate_medians_3x3"] = sector_medians

def candidate_sector_climate_median_3x3(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> float:
    """
    Returns the precomputed climate median for the 3x3 sector
    where the candidate is located.
    """

    wall_id = candidate["wall_id"]
    wall = building_dict.get(wall_id)

    if wall is None:
        return None

    bbox = wall_uv_bbox_from_building(building_dict, wall_id)
    uv = candidate_mean_uv(candidate)

    if bbox is None or uv is None:
        return None

    row_label, col_label = sector_3x3_labels(
        uv[0], uv[1], bbox, v_up=v_axis_points_up(wall)
    )
    key = f"{row_label}_{col_label}"

    sector_medians = wall.get("sector_climate_medians_3x3", {})
    return sector_medians.get(key)


import statistics
from typing import Dict, Any, Optional



def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    """
    Shortest distance from point P to segment AB.
    Uses perpendicular foot if it falls inside the segment,
    otherwise falls back to nearest endpoint.
    """
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        # degenerate segment — return distance to endpoint
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

    # parameter t of the foot along AB
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))   # clamp to [0, 1]

    foot_x = ax + t * dx
    foot_y = ay + t * dy

    return math.sqrt((px - foot_x) ** 2 + (py - foot_y) ** 2)


def point_to_top_edge_distance(
    boundary_uv,
    px: float,
    py: float,
    v_up: bool = True,
) -> Optional[float]:
    """
    Perpendicular distance from (px, py) to the nearest wall boundary edge
    that lies entirely above it.

    These "top" edges are the roofline as seen from this point — for a
    sloped/triangular wall this naturally picks up the diagonal gable edges,
    not just a flat top edge. Works for rectangular, triangular, and any
    polygon wall shape. Returns None if no boundary edge lies above the point.

    v_up says which V direction is physically up on this wall (see
    v_axis_points_up). On a wall whose V axis points down, "above" means
    *smaller* V — get this wrong and the whole function measures to the ground
    line instead of the roof.
    """
    if not boundary_uv or len(boundary_uv) < 3:
        return None

    loop = list(boundary_uv) + [boundary_uv[0]]
    segments = [(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]

    if v_up:
        top_edges = [(a, b) for a, b in segments if a[1] > py and b[1] > py]
    else:
        top_edges = [(a, b) for a, b in segments if a[1] < py and b[1] < py]

    if not top_edges:
        return None

    return min(
        _point_to_segment_dist(px, py, a[0], a[1], b[0], b[1])
        for a, b in top_edges
    )


ROOF_STRICT_BAND_M = 1.5


def is_distance_to_roof_strict(val) -> bool:
    """
    True if a species' distance_to_roof dataset value is "strict"
    (case-insensitive, tolerant of stray/extra whitespace, e.g. " Strict ").
    """
    if val is None:
        return False
    s = re.sub(r"\s+", "", str(val)).strip().lower()
    return s == "strict"


def is_within_roof_strict_band(
    boundary_uv,
    px: float,
    py: float,
    max_dist_m: float = ROOF_STRICT_BAND_M,
    v_up: bool = True,
) -> bool:
    """
    True if (px, py) is within max_dist_m of the nearest roof edge (see
    point_to_top_edge_distance) — the hard-constraint band for species whose
    distance_to_roof is "strict".

    Pass v_up from v_axis_points_up(wall); the default of True is only correct
    for walls whose V axis genuinely points up.
    """
    d = point_to_top_edge_distance(boundary_uv, px, py, v_up=v_up)
    return d is not None and d <= max_dist_m


def candidate_roof_edge_distance_median(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> Optional[float]:
    """
    Returns the median perpendicular distance from candidate nests to the
    nearest top edge of the wall.

    Top edges are defined as boundary segments where BOTH endpoints are
    above the nest's V position. For each nest the shortest perpendicular
    distance across all top edges is used. If the foot of the perpendicular
    falls outside the segment, the distance to the nearest endpoint is used
    instead (acceptable fallback, see design notes).

    Works for rectangular, triangular, and any polygon wall shape.
    """

    wall_id = candidate["wall_id"]
    wall    = building_dict.get(wall_id)
    if wall is None:
        return None

    buv = wall.get("boundary_uv") or []
    if len(buv) < 3:
        return None

    uvs = candidate.get("uv") or []
    if not uvs:
        return None

    v_up = v_axis_points_up(wall)

    dists = []
    for uv in uvs:
        if not uv or len(uv) < 2:
            continue
        d = point_to_top_edge_distance(buv, float(uv[0]), float(uv[1]), v_up=v_up)
        if d is not None:
            dists.append(d)

    if not dists:
        return None

    return float(statistics.median(dists))

def candidate_side_edge_distance_median(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns:
      - median distance from candidate nests to the nearest side boundary
        of the wall, measured as a horizontal ray in UV space.
      - dominant side label: 'umin' or 'umax'

    For each nest a horizontal ray is cast left and right at the nest's V
    position. It intersects the actual wall boundary_uv polygon to find the
    true wall edge at that height — correct for triangular, rectangular, and
    any polygon wall shape.

    If the ray finds no intersection on one side (e.g. nest is outside the
    wall polygon), that side is skipped.
    """

    wall_id = candidate["wall_id"]
    wall    = building_dict.get(wall_id)
    if wall is None:
        return {"side_edge_distance_median": None, "side_edge_label": None}

    buv = wall.get("boundary_uv") or []
    if len(buv) < 3:
        return {"side_edge_distance_median": None, "side_edge_label": None}

    # closed boundary loop as segments
    loop     = buv + [buv[0]]
    segments = [(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]

    # U extent for ray length
    us   = [p[0] for p in buv]
    umin = min(us)
    umax = max(us)

    uvs = candidate.get("uv") or []
    if not uvs:
        return {"side_edge_distance_median": None, "side_edge_label": None}

    def ray_u_intersections(px: float, py: float) -> list[float]:
        """
        Returns all U values where the horizontal line V=py intersects
        the wall boundary segments.
        """
        hits = []
        for (ax, ay), (bx, by) in segments:
            # skip horizontal segments
            if abs(by - ay) < 1e-9:
                continue
            t = (py - ay) / (by - ay)
            if 0.0 <= t <= 1.0:
                u_hit = ax + t * (bx - ax)
                hits.append(u_hit)
        return hits

    dists       = []
    side_labels = []

    for uv in uvs:
        if not uv or len(uv) < 2:
            continue
        px, py = float(uv[0]), float(uv[1])

        hits = ray_u_intersections(px, py)
        if not hits:
            continue

        # intersections to the left and right of the nest
        left_hits  = [u for u in hits if u <= px]
        right_hits = [u for u in hits if u >= px]

        d_left  = (px - max(left_hits))  if left_hits  else None
        d_right = (min(right_hits) - px) if right_hits else None

        if d_left is None and d_right is None:
            continue

        if d_left is None:
            dists.append(d_right)
            side_labels.append("umax")
        elif d_right is None:
            dists.append(d_left)
            side_labels.append("umin")
        elif d_left <= d_right:
            dists.append(d_left)
            side_labels.append("umin")
        else:
            dists.append(d_right)
            side_labels.append("umax")

    if not dists:
        return {"side_edge_distance_median": None, "side_edge_label": None}

    n_umin         = side_labels.count("umin")
    n_umax         = side_labels.count("umax")
    dominant_side  = "umin" if n_umin >= n_umax else "umax"

    return {
        "side_edge_distance_median": float(statistics.median(dists)),
        "side_edge_label": dominant_side,
    }

def _window_side_distance_at_point(windows: dict, px: float, py: float) -> Optional[float]:
    """
    Horizontal (side-only) distance from (px, py) to the nearest window whose
    vertical (V) extent contains py — a window directly above or below the
    point is ignored entirely; only windows level with it ("to the side")
    count. Returns None if no window's V-range contains py.
    """
    best = None
    for win in (windows or {}).values():
        huv = win.get("hull_uv")
        if not huv or len(huv) < 3:
            continue

        us = [p[0] for p in huv]
        vs = [p[1] for p in huv]
        v_min, v_max = min(vs), max(vs)

        if py < v_min or py > v_max:
            continue

        u_min, u_max = min(us), max(us)

        if u_min <= px <= u_max:
            d = 0.0
        elif px < u_min:
            d = u_min - px
        else:
            d = px - u_max

        if best is None or d < best:
            best = d

    return best


def candidate_window_side_distance(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> Optional[float]:
    """
    Distance from the candidate's closest nest point to the nearest window,
    measured only horizontally (side-to-side) — windows directly above or
    below a nest point don't count, only ones level with it do.

    For a colony this is "the distance from the closest nest placement to
    the closest window nearby"; for a solitary placement (one point) it's
    just that point's distance to the nearest window to its side.
    Returns None if the wall has no windows, or none share a nest point's
    height band.
    """
    wall_id = candidate["wall_id"]
    wall    = building_dict.get(wall_id)
    if wall is None:
        return None

    windows = wall.get("windows") or {}
    if not windows:
        return None

    uvs = candidate.get("uv") or []
    if not uvs:
        return None

    dists = []
    for uv in uvs:
        if not uv or len(uv) < 2:
            continue
        d = _window_side_distance_at_point(windows, float(uv[0]), float(uv[1]))
        if d is not None:
            dists.append(d)

    if not dists:
        return None

    return float(min(dists))


def candidate_window_side_distance_median(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> Optional[float]:
    """
    Median, across all of the candidate's nest points, of each nest's
    distance to its nearest same-height-band window — same per-point rule as
    candidate_window_side_distance (side-only; windows above/below a nest
    don't count), but median instead of minimum.

    Where candidate_window_side_distance answers "how close is the closest
    nest to a window," this answers "how close are the nests to windows
    typically" — robust to a single outlier nest, useful when a colony has
    several nests and only one happens to sit right next to a window.
    Returns None if the wall has no windows, or none share any nest point's
    height band.
    """
    wall_id = candidate["wall_id"]
    wall    = building_dict.get(wall_id)
    if wall is None:
        return None

    windows = wall.get("windows") or {}
    if not windows:
        return None

    uvs = candidate.get("uv") or []
    if not uvs:
        return None

    dists = []
    for uv in uvs:
        if not uv or len(uv) < 2:
            continue
        d = _window_side_distance_at_point(windows, float(uv[0]), float(uv[1]))
        if d is not None:
            dists.append(d)

    if not dists:
        return None

    return float(statistics.median(dists))


def candidate_edge_features(
    building_dict: dict,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns edge-distance features for one candidate.
    """

    top_dist = candidate_roof_edge_distance_median(building_dict, candidate)
    side_feats = candidate_side_edge_distance_median(building_dict, candidate)
    window_dist = candidate_window_side_distance(building_dict, candidate)
    window_dist_median = candidate_window_side_distance_median(building_dict, candidate)

    return {
        "dist_to_top_edge_median": top_dist,
        "dist_to_side_edge_median": side_feats["side_edge_distance_median"],
        "side_edge_label": side_feats["side_edge_label"],
        "distance_to_window": window_dist,
        "distance_to_window_median": window_dist_median,
    }

def orientation_from_xy_vector(x: float, y: float) -> tuple[Optional[str], Optional[float]]:
    """
    Converts XY vector to compass orientation.

    Assumption:
      +X = East
      -X = West
      +Y = North
      -Y = South
    """
    if x is None or y is None:
        return None, None

    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None, None

    angle = math.degrees(math.atan2(y, x))

    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(round(angle / 45.0)) % 8

    return directions[idx], float(angle)


def precompute_wall_orientations(
    building_dict: Dict[str, dict],
    *,
    print_debug: bool = False,
) -> None:
    """
    Computes wall orientation from wall['plane']['zaxis']
    and stores it in-place:

        wall['orientation'] = 'N' / 'NE' / ...
        wall['orientation_deg'] = angle
    """

    for wall_id, wall in building_dict.items():
        if not isinstance(wall, dict) or wall_id.startswith("_"):
            continue

        zaxis = wall.get("plane", {}).get("zaxis")
        floor_fn = wall.get("floor_function")

        if not zaxis or len(zaxis) < 2:
            wall["orientation"] = None
            wall["orientation_deg"] = None
            continue

        x = float(zaxis[0])
        y = float(zaxis[1])

        ori, angle = orientation_from_xy_vector(x, y)

        wall["orientation"] = ori
        wall["orientation_deg"] = angle

        # if print_debug:
        #     print(
        #         wall_id,
        #         "| floor_function =", floor_fn,
        #         "| zaxis =", zaxis,
        #         "->", ori,
        #         f"({angle:.1f}°)" if angle is not None else ""
        #     )

def candidate_local_height_stats(
    building_dict: dict,
    candidate: dict,
) -> Dict[str, float]:
    """
    Returns candidate height statistics relative to the wall lowest Z value.

    height_std_m is the standard deviation of the candidate's nest Z
    positions — low values mean the colony's nests sit at roughly the same
    height (a horizontal cluster), high values mean they're spread out
    vertically (stacked in a line). 0.0 for a single-point (solitary)
    candidate.
    """

    wall_id = candidate["wall_id"]
    wall = building_dict.get(wall_id, {})

    xyz = candidate.get("xyz") or []

    if not xyz:
        return {
            "mean_height_m": np.nan,
            "min_height_m": np.nan,
            "max_height_m": np.nan,
            "height_std_m": np.nan,
        }

    # wall local zero
    grid = wall.get("grid", {}) or {}

    wall_zs = [
        pdata["point_on_wall"][2]
        for pdata in grid.values()
        if isinstance(pdata, dict)
        and pdata.get("point_on_wall")
    ]

    if not wall_zs:
        return {
            "mean_height_m": np.nan,
            "min_height_m": np.nan,
            "max_height_m": np.nan,
            "height_std_m": np.nan,
        }

    wall_ground_z = float(np.min(wall_zs))

    zs_local = [
        float(p[2]) - wall_ground_z
        for p in xyz
    ]

    return {
        "mean_height_m": float(np.mean(zs_local)),
        "min_height_m": float(np.min(zs_local)),
        "max_height_m": float(np.max(zs_local)),
        "height_std_m": float(np.std(zs_local)),
    }