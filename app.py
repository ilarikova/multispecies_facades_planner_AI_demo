import re
import joblib
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from shapely.geometry import Point, Polygon, box
from shapely.ops import triangulate, unary_union

from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI import facade_planner_species as fps
from multispecies_facades_planner_AI import data_extraction as de
from multispecies_facades_planner_AI.data_training_model1_test import (
    plan,
    _parse_box_spacing,
    _parse_colony_size,
    _parse_solitary_boxes,
)
from multispecies_facades_planner_AI.data_training_model1_species_combination import plan_species_combination

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "demo_data"
ICONS_DIR = DATA_DIR / "icons"

EXCEL_PATH = DATA_DIR / "bird_species.xlsx"
# Final model: XGBoost trained on all 239 labelled iterations (5 buildings,
# 6 species, reduced feature set, 500 trees). Replaces the earlier model that
# had seen only 4 species. Written by xgboost 3.2.0, matching requirements.txt.
XGB_RANKER_PATH = DATA_DIR / "models" / "nestworks_xgb_ranker_final.pkl"
XGB_ENCODERS_PATH = DATA_DIR / "models" / "nestworks_encoders_final.pkl"
MODEL_TYPE = "xgb"

COLOR_A = "#F4A623"  # orange — best placement (single mode) / species A (combination mode)
COLOR_B = "#6B3A7D"  # purple — other placement(s) (single mode) / species B (combination mode)

# Max distance (meters) a window/door mesh may fall outside its own wall's mesh
# bounding box before it's treated as bad export data and skipped entirely —
# some buildings have openings whose baked mesh doesn't actually sit on the wall.
OPENING_FIT_TOLERANCE_M = 0.8

BUILDINGS = [
    {"file": "building4868.json", "street": "Preysingstraße", "house_number": "3", "zip_code": "85049"},
    {"file": "building5038.json", "street": "Münzbergstraße", "house_number": "16", "zip_code": "85049"},
    # No street address on record for these two — labelled by building id until
    # someone fills them in. They are the other two buildings the models were
    # trained on (export3107).
    {"file": "building4115.json", "street": "Building 4115", "house_number": "", "zip_code": "85049"},
    {"file": "building5128.json", "street": "Building 5128", "house_number": "", "zip_code": "85049"},
]


def building_address(b: dict) -> str:
    number = str(b.get("house_number") or "").strip()
    street = f"{b['street']} {number}".strip() if number else b["street"]
    return f"{street}, {b['zip_code']}"

ICON_ALIASES = {"house_sparrow": "sparrow"}

# The pairings that are ecologically wanted: house sparrow with swift, each of
# them with either bat, and nothing else. Both members must be in
# SINGLE_SPECIES_ALLOWLIST.
ALLOWED_SPECIES_PAIRS = [
    ("house_sparrow", "swift"),
    ("house_sparrow", "common_noctule"),
    ("house_sparrow", "common_pipistrelle"),
    ("swift", "common_noctule"),
    ("swift", "common_pipistrelle"),
    ("house_martin", "common_noctule"),
    ("house_martin", "common_pipistrelle"),
]

ORDINAL_WALL_LABELS = ["Best wall", "Second best wall", "Third best wall", "Fourth best wall"]


def ordinal_wall_label(rank: int) -> str:
    idx = rank - 1
    if 0 <= idx < len(ORDINAL_WALL_LABELS):
        return ORDINAL_WALL_LABELS[idx]
    return f"{rank}th best wall"


def color_dot_html(color: str) -> str:
    return (
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{color};margin-right:6px;'></span>"
    )


# ─────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_building(building_path: str) -> dict:
    building_dict = fpf.load_building_dict(building_path)
    # Both are also done inside plan(), but the radiation view is drawn before
    # anything is planned - without the climate medians it has nothing to shade,
    # and since that view is cached it stayed blank even after the first run.
    fpf.precompute_wall_climate_features(building_dict)
    fpf.precompute_wall_orientations(building_dict)
    return building_dict


@st.cache_resource
def load_model():
    model = joblib.load(XGB_RANKER_PATH)
    encoders = joblib.load(XGB_ENCODERS_PATH)
    return model, encoders


# The six species the deployed model was trained and validated on in export3107,
# in the order they are offered: birds first, bats last. Add a species only once
# it carries expert labels; without them the model has never seen its trait
# values in a labelled ranking group.
SPECIES_ORDER = [
    "black_redstart",
    "house_sparrow",
    "swift",
    "house_martin",
    "common_noctule",
    "common_pipistrelle",
]
SINGLE_SPECIES_ALLOWLIST = set(SPECIES_ORDER)

# How each species is placed, once the model has ranked walls and sectors:
#   roofline    — swift and house martin go in a horizontal line directly under
#                 the roof, the LBV's requirement for both.
#   per_facade  — the two bats get two boxes on every facade the hard
#                 constraints allow, at the spacing from the species sheet.
#   irregular / symmetric — of the rest, only the house sparrow is offered the
#                 regular-grid alternative to the clustered placement.
# Both are placed under the roofline; the house martin takes one row only -
# best sector, then the adjacent one - while the swift may open a second row
# below the first when the two sectors cannot carry the colony.
ROOFLINE_SPECIES = {"swift"}
ROOFLINE_SINGLE_ROW_SPECIES = {"house_martin"}
PER_FACADE_SPECIES = {"common_noctule", "common_pipistrelle"}
STYLE_CHOICE_SPECIES = {"house_sparrow"}

# Not yet labelled: spotted_flycatcher, robin, wagtail, great_tit, blue_tit,
# tree_sparrow, starling, jackdaw


@st.cache_data
def species_choices() -> list[str]:
    df = pd.read_excel(EXCEL_PATH)
    vals = df["specie_name_EN"].dropna().astype(str).str.strip()
    all_species = {v[: -len("_core")] for v in vals if v.endswith("_core")}
    return [s for s in SPECIES_ORDER if s in all_species]


@st.cache_data
def load_needs(species_name: str) -> dict:
    return de.load_species_training_as_dict(str(EXCEL_PATH), species_name)[species_name]


@st.cache_data
def load_species_icon_bytes(species_name: str) -> bytes | None:
    stem = ICON_ALIASES.get(species_name, species_name)
    p = ICONS_DIR / f"{stem}_core.png"
    return p.read_bytes() if p.exists() else None


# ─────────────────────────────────────────────────────────────────
# 3D VIEW HELPERS (generic over any building_dict)
# ─────────────────────────────────────────────────────────────────

def triangulate_faces(faces):
    I, J, K = [], [], []
    for f in faces or []:
        if len(f) < 3:
            continue
        a = f[0]
        for i in range(1, len(f) - 1):
            I.append(a)
            J.append(f[i])
            K.append(f[i + 1])
    return I, J, K


# Coordinates are rounded to the millimetre before they go into a figure.
# Plotly serialises floats in full double precision, which doubles the size of
# every payload the browser has to download for no visible difference.
COORD_DECIMALS = 3


def merge_meshes(meshes):
    """
    One set of vertex and face arrays from many meshes.

    A building exports as hundreds of separate little meshes - every window and
    door its own. Drawn as hundreds of Mesh3d traces they cost the browser far
    more than the geometry itself does, so everything sharing a colour is
    concatenated into a single trace.
    """
    X, I, J, K = [], [], [], []
    offset = 0
    for mesh in meshes:
        V = np.round(np.asarray(mesh["vertices"], dtype=float), COORD_DECIMALS)
        if V.size == 0:
            continue
        if "_tri" not in mesh:
            mesh["_tri"] = triangulate_faces(mesh["faces"])
        i, j, k = mesh["_tri"]
        X.append(V)
        I.extend(a + offset for a in i)
        J.extend(a + offset for a in j)
        K.extend(a + offset for a in k)
        offset += len(V)
    if not X:
        return None
    return (np.vstack(X),
            np.asarray(I, dtype=np.int32),
            np.asarray(J, dtype=np.int32),
            np.asarray(K, dtype=np.int32))


def merge_triangles(parts):
    """Concatenate (vertices, faces) pairs into one, renumbering the faces."""
    verts: list = []
    faces: list = []
    for V, F in parts:
        base = len(verts)
        verts.extend(V.tolist())
        faces.extend((np.asarray(F, dtype=np.int64) + base).tolist())
    if not faces:
        return None
    return (np.asarray(verts, dtype=float),
            np.asarray(faces, dtype=np.int32))


def add_triangles(fig, parts, name, opacity=0.15, color=None):
    merged = merge_triangles(parts)
    if merged is None:
        return
    V, F = merged
    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            name=name,
            opacity=opacity,
            color=color,
            showscale=False,
            hoverinfo="skip",
        )
    )


def mesh_outline_uv(wall: dict, snap_m: float = 1e-3):
    """
    The outline of a flat wall's mesh, in the wall's UV frame.

    A wall is exported as a dense tessellation — thousands of triangles for a
    flat surface. Its silhouette is the set of edges that belong to a single
    triangle, which recovers the true shape (notches, sloped tops and all) in
    one pass and replaces those thousands of triangles with a few. Vertices are
    snapped to the millimetre first, because the same corner is repeated under
    several indices in the export and the edges would otherwise not meet.

    Returns a shapely polygon, or None if the mesh does not resolve to one ring.
    """
    mesh = wall.get("mesh") or {}
    plane = wall.get("plane")
    V = mesh.get("vertices")
    if not V or not plane:
        return None

    o = np.asarray(plane["origin"], dtype=float)
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    rel = np.asarray(V, dtype=float) - o
    uv = np.round(np.stack([rel @ ux, rel @ uy], axis=1) / snap_m).astype(np.int64)

    keys = {}
    remap = np.empty(len(uv), dtype=np.int64)
    points = []
    for n, key in enumerate(map(tuple, uv)):
        at = keys.get(key)
        if at is None:
            at = keys[key] = len(points)
            points.append((key[0] * snap_m, key[1] * snap_m))
        remap[n] = at

    if "_tri" not in mesh:
        mesh["_tri"] = triangulate_faces(mesh["faces"])
    counts = {}
    for a, b, c in zip(*mesh["_tri"]):
        i, j, k = int(remap[a]), int(remap[b]), int(remap[c])
        if i == j or j == k or k == i:
            continue
        for e in ((i, j), (j, k), (k, i)):
            key = (min(e), max(e))
            counts[key] = counts.get(key, 0) + 1

    neighbours = {}
    for (i, j), n in counts.items():
        if n != 1:                       # interior edge, shared by two triangles
            continue
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    if not neighbours or any(len(v) != 2 for v in neighbours.values()):
        return None                      # branching outline: not a simple ring

    start = next(iter(neighbours))
    ring = [start]
    prev, cur = None, start
    while True:
        a, b = neighbours[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        ring.append(nxt)
        prev, cur = cur, nxt
        if len(ring) > len(neighbours):
            return None
    if len(ring) != len(neighbours) or len(ring) < 3:
        return None                      # more than one ring (a hole): leave it

    poly = Polygon([points[i] for i in ring]).simplify(snap_m)
    return poly if poly.is_valid and not poly.is_empty else None


def wall_outline_geom(wall: dict):
    """The wall's drawable outline: from its mesh, else the stored boundary."""
    poly = mesh_outline_uv(wall)
    if poly is not None:
        return poly
    bd = wall.get("boundary_uv") or []
    if len(bd) < 3:
        return None
    poly = Polygon([(float(p[0]), float(p[1])) for p in bd])
    return poly if poly.is_valid and not poly.is_empty else None


def add_merged_meshes(fig, meshes, name, opacity=0.15, color=None):
    merged = merge_meshes(meshes)
    if merged is None:
        return
    V, I, J, K = merged
    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=I,
            j=J,
            k=K,
            name=name,
            opacity=opacity,
            color=color,
            showscale=False,
            hoverinfo="skip",
        )
    )


def opening_mesh_fits_wall(wall_vmin: np.ndarray, wall_vmax: np.ndarray, opening_vertices, tol: float) -> bool:
    OV = np.asarray(opening_vertices, dtype=float)
    d = np.maximum(wall_vmin - OV, 0) + np.maximum(OV - wall_vmax, 0)
    return float(np.linalg.norm(d, axis=1).max()) <= tol


def building_ground_z(walls_data: dict) -> float:
    """Lowest point of the building, in world z."""
    zs = [v[2]
          for w in walls_data.values() if isinstance(w, dict)
          for v in ((w.get("mesh") or {}).get("vertices") or [])]
    return float(min(zs)) if zs else 0.0


def building_bounds(walls_data: dict):
    """(min, max) corner of the building, in world coordinates."""
    V = [v
         for w in walls_data.values() if isinstance(w, dict)
         for v in ((w.get("mesh") or {}).get("vertices") or [])]
    if not V:
        return None
    A = np.asarray(V, dtype=float)
    return A.min(axis=0), A.max(axis=0)


def _local_axis(lo: float, hi: float, title: str) -> dict:
    """Ticks counting from `lo`, labelled in metres, about five of them."""
    span = max(hi - lo, 1.0)
    step = next((s for s in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0) if span / s <= 6), 100.0)
    marks = [k * step for k in range(int(span / step) + 1)]
    return dict(
        tickmode="array",
        tickvals=[lo + m for m in marks],
        ticktext=[f"{m:.0f}" for m in marks],
        title=title,
    )


def local_axes(walls_data: dict) -> dict:
    """
    Scene axes measured from the building's own lowest corner.

    The exports carry city-model coordinates — building 4868 sits at y ≈ −105,
    z ≈ 367 m above sea level — which says nothing about the building. The
    geometry keeps those coordinates, since the planner works in them; only the
    tick labels are rewritten, so a 44 × 38 m, 19 m building reads 0–44, 0–38
    and 0–19.
    """
    bounds = building_bounds(walls_data)
    if bounds is None:
        return dict(zaxis=dict(title="height (m)"))
    lo, hi = bounds
    return dict(
        xaxis=_local_axis(lo[0], hi[0], "x (m)"),
        yaxis=_local_axis(lo[1], hi[1], "y (m)"),
        zaxis=_local_axis(lo[2], hi[2], "height (m)"),
    )


def wall_mesh_normal(wall: dict) -> np.ndarray:
    mesh = wall.get("mesh") or {}
    fn = mesh.get("face_normals")
    if fn and len(fn) > 0:
        n = np.mean(np.asarray(fn, dtype=float), axis=0)
    else:
        n = np.asarray((wall.get("plane") or {}).get("zaxis", [0, 0, 1]), dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    return n


def nice_species_label(stem: str) -> str:
    s = stem
    s = re.sub(r"^cre[_-]*", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\bcore\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else stem


def add_circles_on_plane(fig, centers, plane: dict, radius_m=0.10, n=48, name="", color=None):
    """All of one placement's nest rings as a single trace, split by None gaps."""
    if not centers:
        return
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    ts = np.linspace(0, 2 * np.pi, n, endpoint=True)
    ring = radius_m * (np.cos(ts)[:, None] * ux + np.sin(ts)[:, None] * uy)

    xs: list = []
    ys: list = []
    zs: list = []
    for c in centers:
        pts = np.round(np.asarray(c, dtype=float) + ring, COORD_DECIMALS)
        xs.extend(pts[:, 0].tolist() + [None])
        ys.extend(pts[:, 1].tolist() + [None])
        zs.extend(pts[:, 2].tolist() + [None])

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=4, color=color),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ─────────────────────────────────────────────────────────────────
# PLACEMENT AREA
# ─────────────────────────────────────────────────────────────────

# Intensity of the placement-area fill relative to the nest markers.
# 0.50 -> 0.35 (toned down 15pp) -> 0.50 (back up 15pp on request).
PLACEMENT_AREA_OPACITY = 0.50

# Only used if a wall's spacing cannot be read off its own grid.
GRID_SIZE_FALLBACK_M = 0.30


def wall_grid_size(wall: dict):
    """
    Grid spacing of this wall as (du, dv), measured from its own grid rather
    than assumed, so it follows automatically if the export grid ever changes.

    u and v spacing are returned separately because they differ slightly in the
    exports (e.g. 0.2976 vs 0.2956) — averaging them leaves hairline gaps
    between the cells that make up the placement area.
    """
    grid = wall.get("grid") or {}
    steps = []
    for axis in (0, 1):
        vals = sorted({round(float(p["uv"][axis]), 4)
                       for p in grid.values() if p.get("uv")})
        deltas = [b - a for a, b in zip(vals, vals[1:]) if b - a > 0.01]
        steps.append(float(np.median(deltas)) if deltas else GRID_SIZE_FALLBACK_M)
    return steps[0], steps[1]


def placement_area_geom(wall: dict, placement: dict, needs: dict):
    """
    Placeable area around one placement, in the wall's own UV frame.

    The outermost nests define an axis-aligned rectangle; it is grown by one
    grid cell on every side, then reduced to the grid positions that actually
    pass the hard constraints — window/door offsets, the species minimum
    height, and the 1.5 m roof band where the species requires it.

    Returns a shapely geometry, or None when nothing qualifies.
    """
    uvs = placement.get("uv") or []
    if not uvs:
        return None

    du, dv = wall_grid_size(wall)
    us = [float(p[0]) for p in uvs]
    vs = [float(p[1]) for p in uvs]
    envelope = box(min(us) - du, min(vs) - dv, max(us) + du, max(vs) + dv)
    e_u0, e_v0, e_u1, e_v1 = envelope.bounds

    colonial = fps.encode_species_traits(needs).get("colonial") == 1
    usable = fpf.derive_openings(wall).difference(
        fpf.build_offset_area(wall, colonial=colonial)
    )
    if usable.is_empty:
        return None

    grid = wall.get("grid") or {}
    zs = [float(p["point_on_wall"][2]) for p in grid.values() if p.get("point_on_wall")]
    ground = min(zs) if zs else 0.0
    min_h = fpf.parse_min_height_m(needs.get("nest_height"), 0.0)
    roof_strict = fpf.is_distance_to_roof_strict(needs.get("distance_to_roof"))
    boundary_uv = wall.get("boundary_uv") or []
    v_up = fpf.v_axis_points_up(wall)

    # 2% oversize so neighbouring cells overlap instead of leaving hairline
    # seams; the final intersection below restores the true outer boundary
    hu, hv = du * 0.51, dv * 0.51
    cells = []
    for p in grid.values():
        uv, xyz = p.get("uv"), p.get("point_on_wall")
        if not uv or not xyz:
            continue
        u, v = float(uv[0]), float(uv[1])
        if not (e_u0 <= u <= e_u1 and e_v0 <= v <= e_v1):
            continue
        if not usable.contains(Point(u, v)):
            continue
        if float(xyz[2]) - ground < min_h - 1e-6:
            continue
        if roof_strict and not fpf.is_within_roof_strict_band(
            boundary_uv, u, v, v_up=v_up
        ):
            continue
        cells.append(box(u - hu, v - hv, u + hu, v + hv))

    if not cells:
        return None

    area = unary_union(cells).intersection(envelope).intersection(usable)
    return None if area.is_empty else area


def ear_clip(coords):
    """
    Triangles of a simple polygon ring, as (points, index triples).

    The wall outlines are not all rectangles — gables and setbacks make them
    concave — and a Delaunay triangulation of their corners would bridge those
    notches, drawing wall where there is none. Ear clipping only ever cuts
    triangles that lie inside the ring. The rings here have at most a couple of
    dozen corners, so the simple quadratic version is more than fast enough.
    """
    pts = [(float(c[0]), float(c[1])) for c in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    n = len(pts)
    if n < 3:
        return pts, []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    order = list(range(n))
    signed = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                 for i in range(n))
    if signed < 0:                      # work anticlockwise
        order.reverse()

    tris = []
    guard = 0
    while len(order) > 2 and guard <= n * n:
        guard += 1
        for k in range(len(order)):
            i0, i1, i2 = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-12:         # reflex corner or a sliver
                continue
            if any(cross(a, b, pts[m]) >= -1e-12
                   and cross(b, c, pts[m]) >= -1e-12
                   and cross(c, a, pts[m]) >= -1e-12
                   for m in order if m not in (i0, i1, i2)):
                continue                        # another corner sits in the ear
            tris.append((i0, i1, i2))
            order.pop(k)
            break
        else:
            break                               # no ear found: ring is degenerate
    return pts, tris


def placement_area_mesh(wall: dict, geom):
    """Triangles of a wall-plane area, as (vertices, faces); None if empty."""
    plane = wall.get("plane")
    if plane is None or geom is None or geom.is_empty:
        return None

    o = np.asarray(plane["origin"], dtype=float)
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)

    verts, faces = [], []
    parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for poly in parts:
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        if poly.interiors:
            # a ring with holes in it — shapely's Delaunay covers the convex
            # hull, so triangles over a hole or a concavity are dropped by
            # testing their centre
            for tri in triangulate(poly):
                if not poly.contains(tri.centroid):
                    continue
                base = len(verts)
                for u, v in list(tri.exterior.coords)[:3]:
                    verts.append(o + float(u) * ux + float(v) * uy)
                faces.append((base, base + 1, base + 2))
            continue
        ring, tris = ear_clip(list(poly.exterior.coords))
        base = len(verts)
        verts.extend(o + float(u) * ux + float(v) * uy for u, v in ring)
        faces.extend((base + a, base + b, base + c) for a, b, c in tris)

    if not faces:
        return None
    return (np.round(np.asarray(verts, dtype=float), COORD_DECIMALS),
            np.asarray(faces, dtype=np.int32))


def add_placement_area(fig, wall: dict, geom, color: str, label: str, opacity=None):
    """Fill the placeable area on the wall plane, at half the nest intensity."""
    mesh = placement_area_mesh(wall, geom)
    if mesh is None:
        return
    V, faces = mesh
    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=PLACEMENT_AREA_OPACITY if opacity is None else opacity,
            flatshading=True,
            name=f"area_{label}",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _fmt_range(lo, hi, decimals=0):
    """'3' when both ends match, otherwise '3-10'."""
    def one(x):
        return f"{x:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{int(round(x))}"
    return one(lo) if abs(hi - lo) < 1e-9 else f"{one(lo)}–{one(hi)}"


def placement_field_caption(needs: dict, species_name: str = "") -> str:
    """
    How many boxes this species takes, and how far apart, straight from the
    species sheet — colony species use colonie_size_local + distance_to_next_nest,
    solitary ones number_of_individual_nest_boxes_on_building +
    distance_between_nest_boxes, matching what the planner itself reads.
    """
    traits = fps.encode_species_traits(needs)
    colonial = traits.get("colonial") == 1

    if colonial:
        lo, hi = _parse_colony_size(needs)
        d_lo, d_hi = traits.get("nest_distance_min_m"), traits.get("nest_distance_max_m")
    else:
        lo, hi = _parse_solitary_boxes(needs)
        d_lo, d_hi = _parse_box_spacing(needs)

    count = _fmt_range(lo, hi)

    def bad(x):
        return x is None or (isinstance(x, float) and (x != x or x >= 1e8))

    # the bats are not planned as one colony: every facade gets its own pair
    subject = ("2 nests per facade" if species_name in PER_FACADE_SPECIES
               else f"{count} nests")
    if bad(d_lo) or bad(d_hi):
        return f"{subject} could be placed in the shown fields."
    return (f"{subject} could be placed in the shown fields "
            f"within {_fmt_range(d_lo, d_hi, decimals=1)} m distance.")


def add_placement_points_and_circles(fig, placement: dict, walls_data: dict, color: str,
                                     radius_m=0.10, label="", needs=None):
    wall_id = placement["wall_id"]
    wall = walls_data.get(wall_id, {})
    plane = wall.get("plane")
    pts = placement.get("xyz") or []
    if not pts:
        return

    # area first, so the nest markers draw on top of it
    if needs is not None:
        add_placement_area(
            fig, wall, placement_area_geom(wall, placement, needs), color, label
        )
    P = np.asarray(pts, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers",
            marker=dict(size=8, color=color),
            name=label,
            showlegend=False,
        )
    )
    if plane:
        add_circles_on_plane(fig, pts, plane, radius_m=radius_m,
                             name=f"circles_{label}", color=color)


def add_wall_floor_function_labels(
    fig,
    walls_data: dict,
    *,
    offset_xy_m: float = 1.5,
    z_lift_m: float = 0.2,
    font_size: int | None = None,
    line_gap_m: float = 1.6,
    kinds: tuple = ("ff", "ori"),
):
    # estimate a "ground" z from all wall meshes
    zs = []
    for w in walls_data.values():
        if not isinstance(w, dict):
            continue
        m = w.get("mesh") or {}
        V = m.get("vertices") or []
        if V:
            zs.extend([v[2] for v in V])
    ground_z = float(min(zs)) if zs else 0.0
    label_z = ground_z + float(z_lift_m)

    # collected and emitted as one trace per kind: a text trace per wall is a
    # per-rerun cost for the browser that buys nothing
    labels = {"ff": [], "ori": []}
    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict):
            continue
        ff = wall.get("floor_function")
        orientation = wall.get("orientation")
        has_ff = "ff" in kinds and bool(ff and str(ff).strip())
        has_ori = "ori" in kinds and bool(orientation and str(orientation).strip())
        if not has_ff and not has_ori:
            continue

        mesh = wall.get("mesh") or {}
        V = mesh.get("vertices")
        if not V:
            continue

        V = np.asarray(V, dtype=float)
        c = V.mean(axis=0)  # centroid

        # wall normal -> XY direction
        n = wall_mesh_normal(wall)
        d = np.array([n[0], n[1], 0.0], dtype=float)
        dn = np.linalg.norm(d)
        if dn < 1e-9:
            # fallback: push in +Y if normal has no XY component
            d = np.array([0.0, 1.0, 0.0], dtype=float)
            dn = 1.0
        d = d / dn

        p = c + offset_xy_m * d
        p[2] = label_z  # force onto "XY plane"

        # orientation stacks directly below floor_function, both centered on the same
        # (x, y) so the two lines read as one label rather than a single \n-joined
        # string (Scatter3d text doesn't render embedded newlines as separate lines).
        half_gap = line_gap_m / 2.0 if (has_ff and has_ori) else 0.0
        if has_ff:
            labels["ff"].append((p[0], p[1], p[2] + half_gap, str(ff)))
        if has_ori:
            labels["ori"].append((p[0], p[1], p[2] - half_gap, str(orientation).upper()))

    font = dict(size=font_size) if font_size else None
    for kind, rows in labels.items():
        if not rows:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[r[0] for r in rows], y=[r[1] for r in rows], z=[r[2] for r in rows],
                mode="text",
                text=[r[3] for r in rows],
                textposition="middle center",
                textfont=font,
                showlegend=False,
                hoverinfo="skip",
                name=kind,
            )
        )


@st.cache_resource
def build_base_figure(_walls_data: dict, building_key: str) -> go.Figure:
    walls_data = _walls_data
    # copy first: the geometry figure is cached and shared with the climate
    # view, which labels itself its own way
    fig = go.Figure(_build_building_geometry(walls_data, building_key))
    add_wall_floor_function_labels(fig, walls_data, offset_xy_m=1.5, z_lift_m=0.2)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    return fig


@st.cache_resource
def _build_building_geometry(_walls_data: dict, building_key: str) -> go.Figure:
    """
    Walls, windows and doors only — no labels, so each view can label its own way.

    `_walls_data` is underscored so Streamlit does not hash it for the cache
    key; `building_key` identifies the building instead.
    """
    walls_data = _walls_data
    fig = go.Figure()
    roofs, wall_meshes, openings = [], [], []
    wall_parts = []
    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict) or "mesh" not in wall:
            continue

        if wall.get("type") == "roof":
            roofs.append(wall["mesh"])
            continue

        # Every wall is flat and the export stores its outline, so the wall is
        # drawn from that outline — a handful of points — rather than from the
        # thousands of triangles the exporter tessellated it into. Same surface,
        # a fraction of the data the browser has to carry on every rerun.
        outline = wall_outline_geom(wall)
        part = placement_area_mesh(wall, outline) if outline is not None else None
        if part is not None:
            wall_parts.append(part)
        else:
            wall_meshes.append(wall["mesh"])
        wins = wall.get("windows") or {}
        doors = wall.get("doors") or {}
        wall_V = np.asarray(wall["mesh"]["vertices"], dtype=float)
        wall_vmin, wall_vmax = wall_V.min(axis=0), wall_V.max(axis=0)
        # Drawn from each opening's own baked mesh (world-space, already correctly
        # positioned) rather than reconstructed from hull_uv + the wall's plane —
        # some walls' window/door hull_uv doesn't line up with their own plane,
        # which sent those openings flying off into space. The PDF exporter
        # (facade_planner_visAI._add_scene) already draws openings this same way.
        # Openings whose mesh still doesn't actually sit on the wall (bad export
        # data, e.g. building0173) are skipped entirely rather than drawn wrong.
        for group in (wins, doors):
            if not isinstance(group, dict):
                continue
            for m in (o.get("mesh") for o in group.values()):
                if m and m.get("vertices") and opening_mesh_fits_wall(
                    wall_vmin, wall_vmax, m["vertices"], OPENING_FIT_TOLERANCE_M
                ):
                    openings.append(m)

    add_merged_meshes(fig, roofs, name="roofs", opacity=1, color="lightgrey")
    add_triangles(fig, wall_parts, name="walls", opacity=0.3, color="lightblue")
    add_merged_meshes(fig, wall_meshes, name="walls_mesh", opacity=0.3, color="lightblue")
    add_merged_meshes(fig, openings, name="openings", opacity=0.45, color="royalblue")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    return fig


# ─────────────────────────────────────────────────────────────────
# CLIMATE SECTOR VIEW
# ─────────────────────────────────────────────────────────────────

# Matches the PDF overview: RdYlBu reversed, so warm sectors read red.
CLIMATE_COLORSCALE = "RdYlBu"
CLIMATE_SECTOR_OPACITY = 0.85
CLIMATE_VIEW_HEIGHT_PX = 420

# Plotly default 3D eye is 1.25; 2.5 pulls the camera twice as far out.
CAMERA_EYE = 2.5

# The 3D scene draws the building low in its canvas and leaves an empty band
# above it. Two knobs against that: a shorter window (cuts the band) and a
# camera that aims below the building's own centre, which lifts the building
# into what is left. More negative = higher in the frame.
MAIN_VIEW_HEIGHT_PX = 560
MAIN_VIEW_CENTER_Z = -0.35


@st.cache_resource
def build_climate_figure(_walls_data: dict, building_key: str) -> go.Figure:
    """
    The same building as the main view, with each wall's 3x3 climate sectors
    shaded by their stored median - the PDF's 'building climate overview'.

    Sectors are assembled from the wall's own grid points rather than from
    rectangles cut out of the UV bbox: the grid already carries the sector
    labels' geometry, so the shading lands exactly where the stored medians
    were measured, and irregular wall outlines clip themselves.
    """
    walls_data = _walls_data
    fig = go.Figure(_build_building_geometry(walls_data, building_key))
    # orientation only, small: this view is a quarter the width of the main one,
    # and what a reader needs from it is which way a warm wall faces
    add_wall_floor_function_labels(
        fig, walls_data,
        offset_xy_m=1.5, z_lift_m=0.2, font_size=9, line_gap_m=0.0,
        kinds=("ori",),
    )

    # one normalisation across the whole building, as the PDF does
    medians = [
        v for w in walls_data.values() if isinstance(w, dict)
        for v in (w.get("sector_climate_medians_3x3") or {}).values()
        if v is not None
    ]
    if not medians:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          scene=dict(aspectmode="data"))
        return fig
    vmin, vmax = float(min(medians)), float(max(medians))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0

    # every sector of every wall ends up in one mesh, each triangle carrying its
    # own sector colour - 70-280 separate meshes was the slowest thing in the app
    sector_verts: list = []
    sector_faces: list = []
    sector_values: list = []

    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict) or not wall.get("boundary_uv"):
            continue
        if str(wall.get("floor_function") or "").strip().lower() in {
            "neighbor_building", "neigbor_building"
        }:
            continue
        sm = wall.get("sector_climate_medians_3x3") or {}
        if not sm:
            continue
        bbox = fpf.wall_uv_bbox_from_building(walls_data, wall_id)
        if bbox is None:
            continue

        du, dv = wall_grid_size(wall)
        hu, hv = du * 0.51, dv * 0.51
        v_up = fpf.v_axis_points_up(wall)
        outline = wall_outline_geom(wall)

        # The sector is the extent of the grid points labelled with it, clipped
        # to the wall outline — one small rectangle each. Taking the union of
        # the individual grid cells instead produced tens of thousands of
        # triangles per building, which was most of this view's weight.
        spans = {}
        for p in (wall.get("grid") or {}).values():
            uv = p.get("uv")
            if not uv:
                continue
            u, v = float(uv[0]), float(uv[1])
            row, col = fpf.sector_3x3_labels(u, v, bbox, v_up=v_up)
            s = spans.setdefault("%s_%s" % (row, col), [u, v, u, v])
            s[0], s[1] = min(s[0], u), min(s[1], v)
            s[2], s[3] = max(s[2], u), max(s[3], v)

        for key, (u0, v0, u1, v1) in spans.items():
            val = sm.get(key)
            if val is None:
                continue
            rect = box(u0 - hu, v0 - hv, u1 + hu, v1 + hv)
            geom = rect.intersection(outline) if outline is not None else rect
            mesh = placement_area_mesh(wall, geom)
            if mesh is None:
                continue
            V, faces = mesh
            base = len(sector_verts)
            sector_verts.extend(V.tolist())
            sector_faces.extend((faces + base).tolist())
            sector_values.extend([float(val)] * len(faces))

    if sector_faces:
        V = np.asarray(sector_verts, dtype=float)
        F = np.asarray(sector_faces, dtype=np.int32)
        fig.add_trace(
            go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                i=F[:, 0], j=F[:, 1], k=F[:, 2],
                intensity=np.asarray(sector_values, dtype=float),
                intensitymode="cell",
                colorscale=CLIMATE_COLORSCALE,
                reversescale=True,          # warm sectors read red, as the PDF does
                cmin=vmin, cmax=vmax, showscale=False,
                opacity=CLIMATE_SECTOR_OPACITY,
                flatshading=True,
                name="climate_sectors",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode="data"), showlegend=False)
    return fig


def placement_caption(walls_data: dict, p: dict) -> str:
    wall = walls_data.get(p["wall_id"], {})
    orientation = wall.get("orientation") or "–"
    shared = f" · shared wall with {nice_species_label(p['shared_wall_with'])}" if p.get("shared_wall_with") else ""
    return (
        f"Orientation: {orientation} · "
        f"Sector: {p['section_row']}/{p['section_col']} · "
        f"Nests: {p['colony_size']} · Score: {p['placement_score']:.3f}{shared}"
    )


# ─────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("NestWorks – demo")

if not XGB_RANKER_PATH.exists() or not XGB_ENCODERS_PATH.exists():
    st.error(f"Model files missing under: {XGB_RANKER_PATH.parent}")
    st.stop()

st.sidebar.header("Buildings Ingolstadt")
building_labels = [building_address(b) for b in BUILDINGS]
picked_building_label = st.sidebar.selectbox("Building", building_labels, key="building_picker")
building = BUILDINGS[building_labels.index(picked_building_label)]
building_path = DATA_DIR / building["file"]

if not building_path.exists():
    st.error(f"Building file missing: {building_path}")
    st.stop()

# switching buildings invalidates any previously generated placements — their
# wall IDs and coordinates belong to a different building's geometry.
if st.session_state.get("prev_building_file") != building["file"]:
    st.session_state.prev_building_file = building["file"]
    for k in ["single_options", "combo_result"]:
        st.session_state.pop(k, None)

walls_data = load_building(str(building_path))
model, xgb_encoders = load_model()
species_list = species_choices()

mode = st.sidebar.radio("Planning mode", ["Single species", "Two species (combination)"])

base_fig = build_base_figure(walls_data, building["file"])
fig = go.Figure(base_fig)

# main placement view on the left, climate reference on the right. Both are
# live plotly scenes, so both rotate; zoom is disabled on the climate one so it
# keeps a stable framing.
view_col, climate_col = st.columns([3, 1])
with view_col:
    st.markdown("#### Nest placements")
    fig_ph = st.empty()

# drawn straight away, so picking a building shows the radiation immediately
# instead of waiting for options to be generated
with climate_col:
    st.markdown("#### Incident radiation")
    climate_fig = go.Figure(build_climate_figure(walls_data, building["file"]))
    climate_fig.update_layout(
        height=CLIMATE_VIEW_HEIGHT_PX,
        showlegend=False,
        scene=dict(
            aspectmode="data",
            **local_axes(walls_data),
            # Same camera as the placement view, and no dragmode="orbit": orbit
            # rotates freely about every axis, so the building tumbled and ended
            # up on its side. The default turntable keeps the vertical upright,
            # which is how the view beside it behaves.
            camera=dict(
                eye=dict(x=CAMERA_EYE, y=CAMERA_EYE, z=CAMERA_EYE * 0.6),
                center=dict(x=0, y=0, z=MAIN_VIEW_CENTER_Z),
            ),
        ),
    )
    # height must be given to Streamlit too: its own `height` defaults to
    # "content", which collapses a plotly 3D scene to nothing.
    st.plotly_chart(
        climate_fig,
        width="stretch",
        height=CLIMATE_VIEW_HEIGHT_PX,
        key="climate_3d",
        config={"scrollZoom": False, "displayModeBar": False,
                "doubleClick": False},
    )
    st.caption(
        "3×3 median solar exposure per wall. Red = warmest, blue = coolest. "
        "Rotate to inspect; no placements shown."
    )

LAYOUT_LABELS = ["Irregular placement", "Symmetrical placement"]
DEFAULT_STYLE = "irregular"


def layout_from_label(label: str) -> str:
    return "symmetric" if label.startswith("Symmetrical") else "irregular"


def layout_for(species_name: str, style: str) -> str:
    if species_name in ROOFLINE_SPECIES:
        return "roofline"
    if species_name in ROOFLINE_SINGLE_ROW_SPECIES:
        return "roofline_single"
    if species_name in PER_FACADE_SPECIES:
        return "per_facade"
    return style if species_name in STYLE_CHOICE_SPECIES else "irregular"


def run_single_plan(species_name: str, layout: str) -> list:
    with st.spinner(f"Planning placements for {nice_species_label(species_name)}..."):
        return plan(
            model=model,
            building_dict=walls_data,
            species_name=species_name,
            needs=load_needs(species_name),
            n_options=2,
            model_type=MODEL_TYPE,
            xgb_encoders=xgb_encoders,
            layout=layout,
        )


def run_combo_plan(species_a: str, species_b: str, style: str) -> dict:
    with st.spinner(
        f"Planning placements for {nice_species_label(species_a)} + "
        f"{nice_species_label(species_b)}..."
    ):
        return plan_species_combination(
            model=model,
            building_dict=walls_data,
            species1_name=species_a,
            needs1=load_needs(species_a),
            species2_name=species_b,
            needs2=load_needs(species_b),
            model_type=MODEL_TYPE,
            xgb_encoders=xgb_encoders,
            layout1=layout_for(species_a, style),
            layout2=layout_for(species_b, style),
        )


icon_bytes_row: list[bytes] = []

if mode == "Single species":
    st.sidebar.header("Species selection")
    species_name = st.sidebar.selectbox("Species", species_list, format_func=nice_species_label)
    run = st.sidebar.button("Generate options", key="generate_single")

    if run:
        layout = layout_for(species_name, st.session_state.get("single_style", DEFAULT_STYLE))
        st.session_state.single_species = species_name
        st.session_state.single_layout = layout
        st.session_state.single_options = run_single_plan(species_name, layout)
        st.session_state.single_option_idx = 0

    if st.session_state.get("single_options"):
        options = st.session_state.single_options
        shown_species = st.session_state.single_species
        option_labels = ["Option 1 (best)", "Option 2"][: len(options)]

        st.sidebar.header("Results")
        st.sidebar.caption(placement_field_caption(load_needs(shown_species), shown_species))
        current_idx = min(st.session_state.get("single_option_idx", 0), len(options) - 1)
        picked = st.sidebar.radio(
            "Show option", option_labels, index=current_idx, horizontal=True, key="single_option_radio"
        )
        pick_idx = option_labels.index(picked)
        st.session_state.single_option_idx = pick_idx

        # Placement style belongs with the results, not with the inputs: it
        # rearranges a colony that has already been placed, so switching it
        # re-plans straight away instead of waiting for another button press.
        if shown_species in STYLE_CHOICE_SPECIES:
            current_style = st.session_state.get("single_style", DEFAULT_STYLE)
            picked_style = layout_from_label(
                st.sidebar.radio(
                    "Placement style",
                    LAYOUT_LABELS,
                    index=LAYOUT_LABELS.index(
                        "Symmetrical placement" if current_style == "symmetric"
                        else "Irregular placement"
                    ),
                    key="layout_single",
                    help=(
                        "Irregular groups the nest boxes as a cluster; symmetrical "
                        "arranges them on a regular grid, with spacing and "
                        "orientation chosen by the model."
                    ),
                )
            )
            if picked_style != current_style:
                st.session_state.single_style = picked_style
                st.session_state.single_layout = picked_style
                st.session_state.single_options = run_single_plan(shown_species, picked_style)
                st.rerun()
            options = st.session_state.single_options

        option = options[pick_idx]
        placements = option["placements"]
        rank_order = sorted(range(len(placements)), key=lambda i: placements[i]["placement_score"], reverse=True)
        for rank, p_idx in enumerate(rank_order, start=1):
            placement = placements[p_idx]
            color = COLOR_A if rank == 1 else COLOR_B
            label = ordinal_wall_label(rank)
            add_placement_points_and_circles(
                fig, placement, walls_data, color=color, label=label,
                needs=load_needs(shown_species),
            )
            st.sidebar.markdown(f"{color_dot_html(color)}**{label}**", unsafe_allow_html=True)
            st.sidebar.caption(placement_caption(walls_data, placement))

        icon_bytes = load_species_icon_bytes(shown_species)
        if icon_bytes:
            icon_bytes_row = [icon_bytes]

else:
    st.sidebar.header("Species selection")
    pair_labels = [f"{nice_species_label(a)} + {nice_species_label(b)}" for a, b in ALLOWED_SPECIES_PAIRS]
    picked_pair_label = st.sidebar.selectbox("Species pair", pair_labels, key="species_pair")
    species_a, species_b = ALLOWED_SPECIES_PAIRS[pair_labels.index(picked_pair_label)]
    run = st.sidebar.button("Generate options", key="generate_combo")

    if run:
        style = st.session_state.get("combo_style", DEFAULT_STYLE)
        st.session_state.combo_species = (species_a, species_b)
        st.session_state.combo_style = style
        st.session_state.combo_result = run_combo_plan(species_a, species_b, style)

    if st.session_state.get("combo_result"):
        sp_a, sp_b = st.session_state.combo_species
        combination_result = st.session_state.combo_result

        st.sidebar.header("Results")

        # only the house sparrow has a style to choose; the swift's roofline and
        # the bats' per-facade pairs are fixed, so the radio is shown when the
        # pair contains a sparrow and applies to it alone
        if STYLE_CHOICE_SPECIES & {sp_a, sp_b}:
            current_style = st.session_state.get("combo_style", DEFAULT_STYLE)
            picked_style = layout_from_label(
                st.sidebar.radio(
                    "Placement style",
                    LAYOUT_LABELS,
                    index=LAYOUT_LABELS.index(
                        "Symmetrical placement" if current_style == "symmetric"
                        else "Irregular placement"
                    ),
                    key="layout_combo",
                    help=(
                        "Applies to the house sparrow. Irregular groups its nest "
                        "boxes as a cluster; symmetrical arranges them on a "
                        "regular grid, with spacing and orientation chosen by "
                        "the model."
                    ),
                )
            )
            if picked_style != current_style:
                st.session_state.combo_style = picked_style
                st.session_state.combo_result = run_combo_plan(sp_a, sp_b, picked_style)
                st.rerun()
            combination_result = st.session_state.combo_result

        for sp_name, color in [(sp_a, COLOR_A), (sp_b, COLOR_B)]:
            st.sidebar.markdown(f"{color_dot_html(color)}**{nice_species_label(sp_name)}**", unsafe_allow_html=True)
            st.sidebar.caption(placement_field_caption(load_needs(sp_name), sp_name))
            placements = combination_result.get(sp_name, [])
            if not placements:
                st.sidebar.caption("No placement found.")
                continue
            for placement in placements:
                add_placement_points_and_circles(
                    fig, placement, walls_data, color=color,
                    label=nice_species_label(sp_name), needs=load_needs(sp_name),
                )
                st.sidebar.caption(placement_caption(walls_data, placement))

        for sp_name in (sp_a, sp_b):
            icon_bytes = load_species_icon_bytes(sp_name)
            if icon_bytes:
                icon_bytes_row.append(icon_bytes)

# --- 3D VIEW ---
fig.update_layout(
    height=MAIN_VIEW_HEIGHT_PX,
    scene=dict(
        aspectmode="data",
        **local_axes(walls_data),
        camera=dict(
            eye=dict(x=CAMERA_EYE, y=CAMERA_EYE, z=CAMERA_EYE * 0.6),
            center=dict(x=0, y=0, z=MAIN_VIEW_CENTER_Z),
        ),
    ),
)
fig_ph.plotly_chart(fig, width="stretch", height=MAIN_VIEW_HEIGHT_PX, key="main_3d")

# --- ADVISORY NOTE ---
st.markdown(
    "<div style='text-align:center;opacity:0.7;font-size:0.9rem;"
    "margin-top:0.25rem;'>Proposed placement serves as a guideline, "
    "please consult ecologists for the final approval.</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

# --- ICON ROW BELOW THE 3D VIEW ---
with st.container():
    if icon_bytes_row:
        cols = st.columns([3] + [1] * len(icon_bytes_row) + [3])
        for col, icon_bytes in zip(cols[1:-1], icon_bytes_row):
            with col:
                st.image(icon_bytes, width=72)
    else:
        st.caption("")
