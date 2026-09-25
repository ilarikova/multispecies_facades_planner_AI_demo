import re
import joblib
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from shapely.geometry import Point, box
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
ROOFLINE_SPECIES = {"swift", "house_martin"}
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


def add_mesh(fig, mesh, name, opacity=0.15, color=None):
    V = np.asarray(mesh["vertices"], dtype=float)
    if "_tri" not in mesh:
        mesh["_tri"] = triangulate_faces(mesh["faces"])
    I, J, K = mesh["_tri"]
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
        )
    )


def opening_mesh_fits_wall(wall_vmin: np.ndarray, wall_vmax: np.ndarray, opening_vertices, tol: float) -> bool:
    OV = np.asarray(opening_vertices, dtype=float)
    d = np.maximum(wall_vmin - OV, 0) + np.maximum(OV - wall_vmax, 0)
    return float(np.linalg.norm(d, axis=1).max()) <= tol


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


def add_circle_on_plane(fig, center_xyz, plane: dict, radius_m=0.10, n=48, name="", color=None):
    c = np.asarray(center_xyz, dtype=float)
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    ts = np.linspace(0, 2 * np.pi, n, endpoint=True)
    pts = [c + radius_m * np.cos(t) * ux + radius_m * np.sin(t) * uy for t in ts]
    pts = np.asarray(pts, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(width=4, color=color),
            name=name,
            showlegend=False,
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


def add_placement_area(fig, wall: dict, geom, color: str, label: str, opacity=None):
    """Fill the placeable area on the wall plane, at half the nest intensity."""
    plane = wall.get("plane")
    if plane is None or geom is None or geom.is_empty:
        return

    o = np.asarray(plane["origin"], dtype=float)
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)

    verts, faces = [], []
    for tri in triangulate(geom):
        # triangulate() covers the convex hull, so drop anything that falls in a
        # concavity or a hole cut by the hard constraints
        if not geom.contains(tri.centroid):
            continue
        base = len(verts)
        for u, v in list(tri.exterior.coords)[:3]:
            verts.append(o + float(u) * ux + float(v) * uy)
        faces.append((base, base + 1, base + 2))

    if not faces:
        return

    V = np.asarray(verts, dtype=float)
    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
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
        for p in pts:
            add_circle_on_plane(fig, p, plane, radius_m=radius_m, name=f"circle_{label}", color=color)


def add_wall_floor_function_labels(
    fig,
    walls_data: dict,
    *,
    offset_xy_m: float = 1.5,
    z_lift_m: float = 0.2,
    font_size: int | None = None,
    line_gap_m: float = 1.6,
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

    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict):
            continue
        ff = wall.get("floor_function")
        orientation = wall.get("orientation")
        has_ff = bool(ff and str(ff).strip())
        has_ori = bool(orientation and str(orientation).strip())
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
        font = dict(size=font_size) if font_size else None
        if has_ff:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2] + half_gap],
                    mode="text",
                    text=[str(ff)],
                    textposition="middle center",
                    textfont=font,
                    showlegend=False,
                    name=f"ff_{wall_id}",
                )
            )
        if has_ori:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2] - half_gap],
                    mode="text",
                    text=[str(orientation).upper()],
                    textposition="middle center",
                    textfont=font,
                    showlegend=False,
                    name=f"ori_{wall_id}",
                )
            )


@st.cache_resource
def build_base_figure(walls_data: dict) -> go.Figure:
    # copy first: the geometry figure is cached and shared with the climate
    # view, which labels itself its own way
    fig = go.Figure(_build_building_geometry(walls_data))
    add_wall_floor_function_labels(fig, walls_data, offset_xy_m=1.5, z_lift_m=0.2)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    return fig


@st.cache_resource
def _build_building_geometry(walls_data: dict) -> go.Figure:
    """Walls, windows and doors only — no labels, so each view can label its own way."""
    fig = go.Figure()
    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict) or "mesh" not in wall:
            continue

        if wall.get("type") == "roof":
            add_mesh(fig, wall["mesh"], name=wall_id, opacity=1, color="lightgrey")
            continue

        add_mesh(fig, wall["mesh"], name=wall_id, opacity=0.3, color="lightblue")
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
        if isinstance(wins, dict):
            for win_id, win in wins.items():
                m = win.get("mesh")
                if m and m.get("vertices") and opening_mesh_fits_wall(
                    wall_vmin, wall_vmax, m["vertices"], OPENING_FIT_TOLERANCE_M
                ):
                    add_mesh(fig, m, name=f"{wall_id}:{win_id}", opacity=0.45, color="royalblue")
        if isinstance(doors, dict):
            for door_id, door in doors.items():
                m = door.get("mesh")
                if m and m.get("vertices") and opening_mesh_fits_wall(
                    wall_vmin, wall_vmax, m["vertices"], OPENING_FIT_TOLERANCE_M
                ):
                    add_mesh(fig, m, name=f"{wall_id}:{door_id}", opacity=0.45, color="royalblue")
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


def _sector_climate_color(t: float) -> str:
    """t in 0..1 -> colour. Sampled at 1-t so high climate reads red (RdYlBu_r)."""
    t = 0.0 if t != t else min(1.0, max(0.0, float(t)))
    return pc.sample_colorscale(CLIMATE_COLORSCALE, [1.0 - t])[0]


@st.cache_resource
def build_climate_figure(walls_data: dict) -> go.Figure:
    """
    The same building as the main view, with each wall's 3x3 climate sectors
    shaded by their stored median - the PDF's 'building climate overview'.

    Sectors are assembled from the wall's own grid points rather than from
    rectangles cut out of the UV bbox: the grid already carries the sector
    labels' geometry, so the shading lands exactly where the stored medians
    were measured, and irregular wall outlines clip themselves.
    """
    fig = go.Figure(_build_building_geometry(walls_data))
    # very small labels, sitting lower — this view is a quarter the width of the
    # main one, so the default label block dominates it otherwise
    add_wall_floor_function_labels(
        fig, walls_data,
        offset_xy_m=1.5, z_lift_m=-1.2, font_size=2, line_gap_m=0.27,
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

        cells = {}
        for p in (wall.get("grid") or {}).values():
            uv = p.get("uv")
            if not uv:
                continue
            u, v = float(uv[0]), float(uv[1])
            row, col = fpf.sector_3x3_labels(u, v, bbox, v_up=v_up)
            cells.setdefault("%s_%s" % (row, col), []).append(
                box(u - hu, v - hv, u + hu, v + hv)
            )

        for key, boxes in cells.items():
            val = sm.get(key)
            if val is None:
                continue
            geom = unary_union(boxes)
            add_placement_area(
                fig, wall, geom,
                _sector_climate_color((float(val) - vmin) / (vmax - vmin)),
                "climate_%s_%s" % (wall_id, key),
                opacity=CLIMATE_SECTOR_OPACITY,
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

base_fig = build_base_figure(walls_data)
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
    climate_fig = go.Figure(build_climate_figure(walls_data))
    climate_fig.update_layout(
        height=CLIMATE_VIEW_HEIGHT_PX,
        showlegend=False,
        scene=dict(
            aspectmode="data",
            dragmode="orbit",
            camera=dict(eye=dict(x=CAMERA_EYE, y=CAMERA_EYE, z=CAMERA_EYE * 0.6)),
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
