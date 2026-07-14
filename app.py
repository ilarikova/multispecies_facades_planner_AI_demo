import re
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI import data_extraction as de
from multispecies_facades_planner_AI.data_training_model1_test import plan
from multispecies_facades_planner_AI.data_training_model1_species_combination import plan_species_combination

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "demo_data"
ICONS_DIR = DATA_DIR / "icons"

EXCEL_PATH = DATA_DIR / "bird_species.xlsx"
XGB_RANKER_PATH = DATA_DIR / "models" / "nestworks_xgb_ranker_reduced.pkl"
XGB_ENCODERS_PATH = DATA_DIR / "models" / "nestworks_xgb_encoders_reduced.pkl"
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
    {"file": "building0173.json", "street": "Schäffbräustraße", "house_number": "23", "zip_code": "85049"},
]


def building_address(b: dict) -> str:
    return f"{b['street']} {b['house_number']}, {b['zip_code']}"

ICON_ALIASES = {"house_sparrow": "sparrow"}

ALLOWED_SPECIES_PAIRS = [
    ("house_sparrow", "swift"),
    ("black_redstart", "house_martin"),
    ("starling", "common_noctule"),
    ("starling", "common_pipistrelle"),
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
    fpf.precompute_wall_orientations(building_dict)
    return building_dict


@st.cache_resource
def load_model():
    model = joblib.load(XGB_RANKER_PATH)
    encoders = joblib.load(XGB_ENCODERS_PATH)
    return model, encoders


@st.cache_data
def species_choices() -> list[str]:
    df = pd.read_excel(EXCEL_PATH)
    vals = df["specie_name_EN"].dropna().astype(str).str.strip()
    return sorted({v[: -len("_core")] for v in vals if v.endswith("_core")})


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


def add_placement_points_and_circles(fig, placement: dict, walls_data: dict, color: str, radius_m=0.10, label=""):
    wall_id = placement["wall_id"]
    wall = walls_data.get(wall_id, {})
    plane = wall.get("plane")
    pts = placement.get("xyz") or []
    if not pts:
        return
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
        line_gap_m = 1.6
        half_gap = line_gap_m / 2.0 if (has_ff and has_ori) else 0.0
        if has_ff:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2] + half_gap],
                    mode="text",
                    text=[str(ff)],
                    textposition="middle center",
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
                    showlegend=False,
                    name=f"ori_{wall_id}",
                )
            )


@st.cache_resource
def build_base_figure(walls_data: dict) -> go.Figure:
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
    add_wall_floor_function_labels(fig, walls_data, offset_xy_m=1.5, z_lift_m=0.2)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
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
fig_ph = st.empty()
icon_bytes_row: list[bytes] = []

if mode == "Single species":
    st.sidebar.header("Species selection")
    species_name = st.sidebar.selectbox("Species", species_list, format_func=nice_species_label)
    run = st.sidebar.button("Generate options", key="generate_single")

    if run:
        needs = load_needs(species_name)
        with st.spinner(f"Planning placements for {nice_species_label(species_name)}..."):
            options = plan(
                model=model,
                building_dict=walls_data,
                species_name=species_name,
                needs=needs,
                n_options=2,
                model_type=MODEL_TYPE,
                xgb_encoders=xgb_encoders,
            )
        st.session_state.single_species = species_name
        st.session_state.single_options = options
        st.session_state.single_option_idx = 0

    if st.session_state.get("single_options"):
        options = st.session_state.single_options
        option_labels = ["Option 1 (best)", "Option 2"][: len(options)]

        st.sidebar.header("Results")
        current_idx = min(st.session_state.get("single_option_idx", 0), len(options) - 1)
        picked = st.sidebar.radio(
            "Show option", option_labels, index=current_idx, horizontal=True, key="single_option_radio"
        )
        pick_idx = option_labels.index(picked)
        st.session_state.single_option_idx = pick_idx

        option = options[pick_idx]
        placements = option["placements"]
        rank_order = sorted(range(len(placements)), key=lambda i: placements[i]["placement_score"], reverse=True)
        for rank, p_idx in enumerate(rank_order, start=1):
            placement = placements[p_idx]
            color = COLOR_A if rank == 1 else COLOR_B
            label = ordinal_wall_label(rank)
            add_placement_points_and_circles(fig, placement, walls_data, color=color, label=label)
            st.sidebar.markdown(f"{color_dot_html(color)}**{label}**", unsafe_allow_html=True)
            st.sidebar.caption(placement_caption(walls_data, placement))

        icon_bytes = load_species_icon_bytes(st.session_state.single_species)
        if icon_bytes:
            icon_bytes_row = [icon_bytes]

else:
    st.sidebar.header("Species selection")
    pair_labels = [f"{nice_species_label(a)} + {nice_species_label(b)}" for a, b in ALLOWED_SPECIES_PAIRS]
    picked_pair_label = st.sidebar.selectbox("Species pair", pair_labels, key="species_pair")
    species_a, species_b = ALLOWED_SPECIES_PAIRS[pair_labels.index(picked_pair_label)]
    run = st.sidebar.button("Generate options", key="generate_combo")

    if run:
        needs_a = load_needs(species_a)
        needs_b = load_needs(species_b)
        with st.spinner(
            f"Planning placements for {nice_species_label(species_a)} + {nice_species_label(species_b)}..."
        ):
            combination_result = plan_species_combination(
                model=model,
                building_dict=walls_data,
                species1_name=species_a,
                needs1=needs_a,
                species2_name=species_b,
                needs2=needs_b,
                model_type=MODEL_TYPE,
                xgb_encoders=xgb_encoders,
            )
        st.session_state.combo_species = (species_a, species_b)
        st.session_state.combo_result = combination_result

    if st.session_state.get("combo_result"):
        sp_a, sp_b = st.session_state.combo_species
        combination_result = st.session_state.combo_result

        st.sidebar.header("Results")
        for sp_name, color in [(sp_a, COLOR_A), (sp_b, COLOR_B)]:
            st.sidebar.markdown(f"{color_dot_html(color)}**{nice_species_label(sp_name)}**", unsafe_allow_html=True)
            placements = combination_result.get(sp_name, [])
            if not placements:
                st.sidebar.caption("No placement found.")
                continue
            for placement in placements:
                add_placement_points_and_circles(
                    fig, placement, walls_data, color=color, label=nice_species_label(sp_name)
                )
                st.sidebar.caption(placement_caption(walls_data, placement))

        for sp_name in (sp_a, sp_b):
            icon_bytes = load_species_icon_bytes(sp_name)
            if icon_bytes:
                icon_bytes_row.append(icon_bytes)

# --- 3D VIEW ---
fig_ph.plotly_chart(fig, use_container_width=True, key="main_3d")
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
