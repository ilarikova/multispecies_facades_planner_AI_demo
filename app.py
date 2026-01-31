import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import re
from joblib import load
import multispecies_facades_planner_AI as mcfp
import inspect
import streamlit as st


from multispecies_facades_planner_AI.facade_planner import (
    generate_colony_candidates,
    generate_solitary_candidates,
)
from multispecies_facades_planner_AI.facade_planner_trainer_global import score_with_global_ranker_new

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "demo_data"
SPECIES_DIR = DATA_DIR / "species"
SPECIES_TRAIN_DIR = DATA_DIR / "species_training"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "ranker_global_2026-01-31_16-15-48.joblib"
ICONS_DIR = DATA_DIR / "icons"

@st.cache_resource
def load_model_pack(path: Path) -> dict:
    return load(path)

def render_tag_chips(tags, *, max_tags=4):
    if not tags:
        st.caption("Reasons: —")
        return
    if isinstance(tags, str):
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, list):
        tags_list = [str(t).strip() for t in tags if str(t).strip()]
    else:
        tags_list = []
    if not tags_list:
        st.caption("Reasons: —")
        return
    tags_list = tags_list[:max_tags]
    chips = " ".join(
        [
            f"<span style='display:inline-block; padding:2px 8px; margin:2px 4px 2px 0; "
            f"border-radius:999px; background:#eef2ff; color:#1f2937; font-size:12px;'>"
            f"{t}</span>"
            for t in tags_list
        ]
    )
    st.markdown(f"**Reasons:** {chips}", unsafe_allow_html=True)

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

def wall_mesh_normal(wall: dict) -> np.ndarray:
    mesh = wall.get("mesh") or {}
    fn = mesh.get("face_normals")
    if fn and len(fn) > 0:
        n = np.mean(np.asarray(fn, dtype=float), axis=0)
    else:
        n = np.asarray((wall.get("plane") or {}).get("zaxis", [0, 0, 1]), dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    return n

def uv_to_xyz(plane: dict, uv, offset_m: float = 0.0, offset_dir=None):
    o = np.array(plane["origin"], dtype=float)
    x = np.array(plane["xaxis"], dtype=float)
    y = np.array(plane["yaxis"], dtype=float)
    u, v = float(uv[0]), float(uv[1])
    p = o + u * x + v * y
    if offset_m != 0.0 and offset_dir is not None:
        n = np.asarray(offset_dir, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        p = p + offset_m * n
    return p

def triangulate_convex_polygon(points_uv):
    n = len(points_uv)
    if n < 3:
        return []
    return [(0, i, i + 1) for i in range(1, n - 1)]

def add_filled_uv_polygon(
    fig,
    plane,
    poly_uv,
    name="window",
    opacity=0.35,
    color="royalblue",
    offset_m=0.002,
    offset_dir=None,
):
    P = np.asarray(
        [uv_to_xyz(plane, uv, offset_m=offset_m, offset_dir=offset_dir) for uv in poly_uv],
        dtype=float,
    )
    faces = triangulate_convex_polygon(poly_uv)
    if not faces:
        return
    i = [a for (a, b, c) in faces]
    j = [b for (a, b, c) in faces]
    k = [c for (a, b, c) in faces]
    fig.add_trace(
        go.Mesh3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            i=i,
            j=j,
            k=k,
            opacity=opacity,
            color=color,
            name=name,
            showlegend=False,
        )
    )

def nice_species_label(stem: str) -> str:
    s = stem
    s = re.sub(r"^cre[_-]*", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\bcore\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else stem

def add_circle_on_plane(fig, center_xyz, plane: dict, radius_m=0.10, n=48, name=""):
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
            line=dict(width=4),
            name=name,
            showlegend=False,
        )
    )

def add_candidate_points_and_circles(fig, candidate: dict, walls_data: dict, radius_m=0.10, label="1"):
    wall_id = candidate["wall_id"]
    wall = walls_data.get(wall_id, {})
    plane = wall.get("plane")
    pts = candidate.get("xyz") or []
    if not pts:
        return
    P = np.asarray(pts, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers+text",
            marker=dict(size=8),
            name=f"Option {label}",
            showlegend=False,
        )
    )
    if plane:
        for p in pts:
            add_circle_on_plane(fig, p, plane, radius_m=radius_m, name=f"circle_{label}")

def load_json_path(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def unwrap_species_core(species_wrapped: dict, species_key: str) -> dict:
    if isinstance(species_wrapped, dict):
        if species_key in species_wrapped and isinstance(species_wrapped[species_key], dict):
            return species_wrapped[species_key]
        if len(species_wrapped) == 1:
            only_key = next(iter(species_wrapped.keys()))
            if isinstance(species_wrapped[only_key], dict):
                return species_wrapped[only_key]
    return species_wrapped

def truthy_yes(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) != 0.0
    if isinstance(v, str):
        return v.strip().lower() in {"yes", "y", "true", "1", "ja"}
    return False

def truthy_no(v) -> bool:
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return float(v) == 0.0
    if isinstance(v, str):
        return v.strip().lower() in {"no", "n", "false", "0", "nein"}
    return False

def species_mode_from_core(species_core: dict) -> str:
    if not isinstance(species_core, dict):
        return "solitary"
    for k in ["is_colonial", "is_colonial_bool", "colonial", "is_colony"]:
        if k in species_core:
            return "colony" if truthy_yes(species_core.get(k)) else "solitary"
    for k in ["colonie", "colony"]:
        if k in species_core:
            v = species_core.get(k)
            if truthy_yes(v):
                return "colony"
            if truthy_no(v):
                return "solitary"
    if species_core.get("colonie_size") is not None or species_core.get("colony_size") is not None:
        return "colony"
    if "if_colonial_distance_to_next_nest" in species_core:
        return "colony"
    return "solitary"

def find_training_species_file(species_id: str, species_key: str) -> Path | None:
    candidates = [
        f"{species_id}_core_training",
        f"{species_key}_core_training",
        f"{species_id}_training",
        f"{species_key}_training",
    ]
    for name in candidates:
        p_json = SPECIES_TRAIN_DIR / f"{name}.json"
        p_noext = SPECIES_TRAIN_DIR / name
        if p_json.exists():
            return p_json
        if p_noext.exists():
            return p_noext
    if SPECIES_TRAIN_DIR.exists():
        all_files = list(SPECIES_TRAIN_DIR.glob("*.json")) + [
            p for p in SPECIES_TRAIN_DIR.iterdir() if p.is_file() and p.suffix == ""
        ]
        sid = (species_id or "").lower()
        sk = (species_key or "").lower()
        scored = []
        for p in all_files:
            st_ = p.stem.lower()
            if "training" not in st_:
                continue
            score = 0
            if sid and sid in st_:
                score += 2
            if sk and sk in st_:
                score += 1
            if score > 0:
                scored.append((score, p))
        if scored:
            scored.sort(key=lambda t: (-t[0], len(t[1].stem)))
            return scored[0][1]
    return None

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
        m = w.get("mesh") or {}
        V = m.get("vertices") or []
        if V:
            zs.extend([v[2] for v in V])
    ground_z = float(min(zs)) if zs else 0.0
    label_z = ground_z + float(z_lift_m)

    for wall_id, wall in walls_data.items():
        ff = wall.get("floor_function", None)
        if ff is None or str(ff).strip() == "":
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

        fig.add_trace(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="text",
                text=[str(ff)],
                textposition="middle center",
                showlegend=False,
                name=f"ff_{wall_id}",
            )
        )

@st.cache_data
def load_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_json(filename: str) -> dict:
    return load_json_file(DATA_DIR / filename)

@st.cache_data
def load_species(species_id: str) -> dict:
    path = SPECIES_DIR / f"{species_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_json_file(path)

@st.cache_data
def load_species_icon_bytes(species_id: str) -> bytes | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = ICONS_DIR / f"{species_id}{ext}"
        if p.exists():
            return p.read_bytes()
    return None

@st.cache_resource
def build_base_figure(walls_data: dict, roof_mesh: dict) -> go.Figure:
    fig = go.Figure()
    for wall_id, wall in walls_data.items():
        if "mesh" not in wall:
            continue
        add_mesh(fig, wall["mesh"], name=wall_id, opacity=0.3, color="lightblue")
        plane = wall.get("plane")
        wins = wall.get("windows") or {}
        doors = wall.get("doors") or {}
        n_wall = wall_mesh_normal(wall)
        if plane and isinstance(wins, dict):
            for win_id, win in wins.items():
                hull_uv = win.get("hull_uv")
                if hull_uv and len(hull_uv) >= 3:
                    add_filled_uv_polygon(
                        fig,
                        plane,
                        hull_uv,
                        name=f"{wall_id}:{win_id}",
                        opacity=0.3,
                        offset_m=0.05,
                        offset_dir=n_wall,
                        color="royalblue",
                    )
        if plane and isinstance(doors, dict):
            for door_id, door in doors.items():
                hull_uv = door.get("hull_uv")
                if hull_uv and len(hull_uv) >= 3:
                    add_filled_uv_polygon(
                        fig,
                        plane,
                        hull_uv,
                        name=f"{wall_id}:{door_id}",
                        opacity=0.35,
                        offset_m=0.002,
                        offset_dir=n_wall,
                        color="royalblue",
                    )
    add_mesh(fig, roof_mesh, name="roof", opacity=1, color="lightgrey")
    add_wall_floor_function_labels(fig, walls_data, offset_xy_m=1.5, z_lift_m=0.2)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    return fig

st.set_page_config(layout="wide")
st.title("NestWorks– demo")

if not MODEL_PATH.exists():
    st.error(f"Model file missing: {MODEL_PATH}")
    st.stop()
_ = load_model_pack(MODEL_PATH)

BUILDING_ADDRESS = "Am Bachl 30, 85049 Ingolstadt"
st.sidebar.header("Building address:")
st.sidebar.caption(BUILDING_ADDRESS)
st.sidebar.header("Species selection")

species_files = sorted(p.stem for p in SPECIES_DIR.glob("*.json"))
if not species_files:
    st.sidebar.error(f"No species .json files found in: {SPECIES_DIR}")
    st.stop()

species_id = st.sidebar.selectbox(
    "Species",
    species_files,
    format_func=nice_species_label,
    key="species_selectbox",
)

if "prev_species_id" not in st.session_state:
    st.session_state.prev_species_id = species_id
if st.session_state.prev_species_id != species_id:
    st.session_state.prev_species_id = species_id
    for k in ["top3", "selected_option_idx", "top3_meta"]:
        if k in st.session_state:
            del st.session_state[k]

run = st.sidebar.button("Generate options", key="generate_btn")

_ = load_species(species_id)
species_key = nice_species_label(species_id).lower().replace(" ", "_")

walls_data = load_json("building.json")
roof_mesh = load_json("building5128_roofs.json")

base_fig = build_base_figure(walls_data, roof_mesh)
fig = go.Figure(base_fig)

fig_ph = st.empty()

if run:
    try:
        training_path = find_training_species_file(species_id=species_id, species_key=species_key)
        if training_path is None:
            available = sorted([p.name for p in SPECIES_TRAIN_DIR.iterdir()]) if SPECIES_TRAIN_DIR.exists() else []
            st.sidebar.error(
                "Could not find training species file.\n\n"
                f"species_id: {species_id}\n"
                f"species_key: {species_key}\n\n"
                f"Available in species_training:\n{available}"
            )
            st.stop()
        species_training_wrapped = load_json_path(training_path)
        species_training_core = unwrap_species_core(species_training_wrapped, species_key)
    except Exception as e:
        st.sidebar.error(f"Failed loading training species file: {e}")
        st.stop()

    mode = species_mode_from_core(species_training_core)
    specie_needs_for_generation = {species_key: species_training_core}

    with st.spinner(f"Generating placement options for {nice_species_label(species_id)} ({mode} specie)..."):
        if mode == "colony":
            candidates = generate_colony_candidates(
                building_dict=walls_data,
                specie_needs=specie_needs_for_generation,
                specie_colonies=1,
                attempts_per_wall=35,
                low_band_m=3.0,
                high_band_m=10.0,
                max_candidates_per_wall=10,
                base_seed=42,
            )
        else:
            candidates = generate_solitary_candidates(
                building_dict=walls_data,
                specie_needs=specie_needs_for_generation,
                attempts_per_wall=35,
                low_band_m=3.0,
                high_band_m=10.0,
                max_candidates_per_wall=20,
                base_seed=42,
                min_nests_per_wall=1,
                max_nests_per_wall=10,
                min_pair_distance_m=10.0,
            )

    if not candidates:
        st.warning("No candidates generated.")
        st.stop()

    try:
        from multispecies_facades_planner_AI.facade_planner_feature import build_feature_table
    except Exception as e:
        st.error(f"Could not import build_feature_table: {e}")
        st.stop()

    with st.spinner("Building feature table..."):
        df_features = build_feature_table(
            candidates=candidates,
            building_dict=walls_data,
            species_dict={species_key: species_training_core},
            iteration_id="demo",
            building_id="5128",
        )

    if df_features is None or len(df_features) == 0:
        st.warning("Feature table is empty.")
        st.stop()

    with st.spinner("Scoring with global ranker..."):
        scored_df = score_with_global_ranker_new(
            df_features=df_features,
            model_path=MODEL_PATH,
            top_k_tags=4,
            tag_min_p=0.1,
        )

    score_col = None
    for c in ["ml_score", "score", "rank_score", "pred", "prediction"]:
        if c in scored_df.columns:
            score_col = c
            break
    if score_col is None:
        st.error(f"Could not find a score column in ranker output. Columns: {list(scored_df.columns)}")
        st.stop()

    if "candidate_id" not in scored_df.columns:
        st.error(f"Ranker output is missing 'candidate_id'. Columns: {list(scored_df.columns)}")
        st.stop()

    cand_by_id = {c["candidate_id"]: c for c in candidates if "candidate_id" in c}
    top3_ids = scored_df.sort_values(score_col, ascending=False).head(3)["candidate_id"].tolist()
    top3 = [cand_by_id.get(cid) for cid in top3_ids if cid in cand_by_id]

    if not top3:
        st.warning("Top-3 candidates could not be matched back to generated candidates.")
        st.stop()

    # CHANGE: update icon species ONLY when Generate options is clicked
    st.session_state.generated_species_id = species_id
    # (optional) cache icon bytes for that generated species, to avoid re-reading file on reruns
    st.session_state.generated_species_icon_bytes = load_species_icon_bytes(species_id)

    st.session_state.top3 = top3
    st.session_state.selected_option_idx = 0

    meta = {}
    for cid in top3_ids:
        row = scored_df[scored_df["candidate_id"] == cid].head(1)
        if len(row) == 0:
            continue
        r0 = row.iloc[0]
        meta[cid] = {
            "score": float(r0[score_col]) if score_col in r0 else None,
            "predicted_tags": r0.get("predicted_tags", []),
            "predicted_tags_str": r0.get("predicted_tags_str", ""),
        }
    st.session_state.top3_meta = meta

if "top3" in st.session_state and st.session_state.top3:
    st.sidebar.header("Results")
    option_labels = ["Option 1 (best)", "Option 2", "Option 3"]

    current_idx = int(st.session_state.get("selected_option_idx", 0))
    picked = st.sidebar.radio(
        "Show option",
        option_labels[: len(st.session_state.top3)],
        index=min(current_idx, len(st.session_state.top3) - 1),
        horizontal=True,
        key="results_option_radio",
    )
    pick_idx = option_labels.index(picked)
    st.session_state.selected_option_idx = pick_idx

    sel = st.session_state.top3[pick_idx]

    meta = st.session_state.get("top3_meta", {}) or {}
    cid = sel.get("candidate_id")
    m = meta.get(cid, {}) if cid else {}

    if m.get("score") is not None:
        st.sidebar.metric("ML score", f"{m['score']:.3f}")
    else:
        st.sidebar.caption("ML score: —")

    st.sidebar.write("")
    tags_payload = m.get("predicted_tags") or m.get("predicted_tags_str", "")
    with st.sidebar:
        render_tag_chips(tags_payload, max_tags=4)

    add_candidate_points_and_circles(fig, sel, walls_data, label=str(pick_idx + 1), radius_m=0.10)

# --- 3D VIEW ---
fig_ph.plotly_chart(fig, use_container_width=True, key="main_3d")
st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

# --- ICON ALWAYS BELOW THE 3D VIEW ---
with st.container():
    # CHANGE: use generated species icon only (updates only after Generate options)
    icon_bytes = st.session_state.get("generated_species_icon_bytes", None)
    if icon_bytes:
        c1, c2, c3 = st.columns([3, 1, 3])
        with c2:
            st.image(icon_bytes, width=72)
    else:
        st.caption("")