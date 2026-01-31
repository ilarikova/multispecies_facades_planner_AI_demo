import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier


# -----------------------------
# Helpers: loading data
# -----------------------------

def load_all_features_global(output_base: Path) -> pd.DataFrame:
    rows = []

    for building_dir in output_base.iterdir():
        if not building_dir.is_dir():
            continue

        for iter_dir in building_dir.iterdir():
            if not iter_dir.is_dir():
                continue

            fcsv = iter_dir / "features.csv"
            if not fcsv.exists():
                continue

            df = pd.read_csv(fcsv)
            rows.append(df)

    if not rows:
        raise RuntimeError("No features.csv found across buildings.")

    return pd.concat(rows, ignore_index=True)


def load_all_feedback_global(output_base: Path) -> pd.DataFrame:
    records = []

    for building_dir in output_base.iterdir():
        if not building_dir.is_dir():
            continue

        fb = building_dir / "feedback.jsonl"
        if not fb.exists():
            continue

        with fb.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))

    if not records:
        raise RuntimeError("No feedback.jsonl found across buildings.")

    return pd.DataFrame(records)


# -----------------------------
# Tag normalization (light safety net)
# -----------------------------

TAG_ALIASES = {
    "pedestrial_local": "pedestrian_local",
}

def normalize_tags(tags: list[str]) -> list[str]:
    """
    Final safety net.
    Assumes tags are already a list of strings.
    - lowercases
    - trims whitespace
    - fixes common alias typos
    - removes empty tokens
    - de-dups while preserving order
    """
    if not tags:
        return []

    out = []
    for t in tags:
        t = str(t).strip().lower()
        if not t:
            continue
        t = TAG_ALIASES.get(t, t)
        # keep only tokens that contain at least one [a-z0-9_]
        if not re.search(r"[a-z0-9_]", t):
            continue
        out.append(t)

    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


# -----------------------------
# Training: Global ranker + tag model
# -----------------------------

def train_ranker_global(
    *,
    output_base: Path,
    model_out_dir: Path,
    min_tag_count: int = 3,
    train_tag_model: bool = True,
):
    """
    Trains:
      1) global ranker (binary accept/reject)
      2) optional tag model (multi-label, trained on ACCEPTED only)

    Saves a joblib pack containing:
      - model, feature_columns
      - (optional) tag_model, tag_classes, tag_min_count
      - metadata: trained_at, n_samples, buildings, species
    """
    # 1) Load everything
    df_feat = load_all_features_global(output_base)
    df_fb = load_all_feedback_global(output_base)

    # --- IMPORTANT: make merge keys consistent types (prevents int64 vs object merge errors) ---
    for k in ["building_id", "iteration_id", "candidate_id", "species"]:
        if k in df_feat.columns:
            df_feat[k] = df_feat[k].astype(str)
        if k in df_fb.columns:
            df_fb[k] = df_fb[k].astype(str)

    # 2) Join on (iteration_id, candidate_id, building_id, species)
    df = df_feat.merge(
        df_fb,
        on=["iteration_id", "candidate_id", "building_id", "species"],
        how="inner",
    )

    if df.empty:
        raise RuntimeError(
            "No matched feature ↔ feedback rows. Check that candidate_id/iteration_id/building_id/species match."
        )

    # --- Backward compatibility (older logs won't have these columns) ---
    if "source" not in df.columns:
        df["source"] = "manual"
    if "strength" not in df.columns:
        df["strength"] = 2
    if "tags" not in df.columns:
        df["tags"] = [[] for _ in range(len(df))]

    # Ensure tags are lists (your exporter should already do this)
    # If something slips through, keep it safe:
    def _coerce_tags(x):
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        # if a stray string got in, don't try to parse here; just drop
        return []

    df["tags"] = df["tags"].apply(_coerce_tags)
    df["tags_norm"] = df["tags"].apply(normalize_tags)

    # --- Train only on explicit feedback (manual + strong) ---
    df = df[(df["source"] == "manual") & (df["strength"] == 2)].copy()

    if df.empty:
        raise RuntimeError(
            "After filtering to explicit feedback (source=manual, strength=2), "
            "no training rows remain. Did you log explicit rejects?"
        )

    # 3) Labels
    df["label"] = (df["decision"] == "accept").astype(int)

    # 4) Feature selection
    drop_cols = {
        "candidate_id",
        "iteration_id",
        "decision",
        "comment",
        "tags",
        "tags_norm",
        "timestamp",
        "source",
        "strength",
        "label",
    }

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and df[c].dtype != "object"
    ]

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found after filtering drop_cols.")

    X = df[feature_cols].fillna(0.0)
    y = df["label"].astype(int)

    # 5) Ranker model (simple + stable)
    ranker_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=400,
            class_weight="balanced",
        )),
    ])
    ranker_model.fit(X, y)

    # 6) Optional tag model (multi-label)
    tag_model = None
    tag_classes: list[str] = []
    mlb = None

    if train_tag_model:
        # Train tag model on ACCEPTED only (positive explanations)
        df_acc = df[df["label"] == 1].copy()

        # compute tag counts
        tag_counts = {}
        for tags in df_acc["tags_norm"]:
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        keep = {t for t, c in tag_counts.items() if c >= int(min_tag_count)}
        df_acc["tags_filt"] = df_acc["tags_norm"].apply(lambda arr: [t for t in arr if t in keep])

        mlb = MultiLabelBinarizer()
        Y_tags = mlb.fit_transform(df_acc["tags_filt"])

        # Only train if we have at least 1 tag class and enough rows
        if Y_tags.shape[1] >= 1 and len(df_acc) >= 5:
            tag_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", OneVsRestClassifier(
                    LogisticRegression(max_iter=500, class_weight="balanced")
                )),
            ])
            tag_model.fit(df_acc[feature_cols].fillna(0.0), Y_tags)
            tag_classes = list(mlb.classes_)
        else:
            tag_model = None
            tag_classes = []

    # 7) Save with timestamp (prevents overwriting)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = model_out_dir / f"ranker_global_{ts}.joblib"

    pack = {
        "model": ranker_model,
        "feature_columns": feature_cols,
        "trained_at": ts,
        "n_samples": int(len(df)),
        "buildings": sorted(df["building_id"].astype(str).unique()),
        "species": sorted(df["species"].astype(str).unique()),
        # tag stuff
        "tag_model": tag_model,
        "tag_classes": tag_classes,
        "tag_min_count": int(min_tag_count),
    }

    dump(pack, model_path)
    return model_path, len(df)


# -----------------------------
# Scoring: rank + predicted tags
# -----------------------------

def score_with_global_ranker_new(
    df_features: pd.DataFrame,
    model_path: Path,
    *,
    top_k_tags: int = 3,
    tag_min_p: float = 0.35,
) -> pd.DataFrame:
    """
    Adds:
      - ml_score (ranker probability)
      - predicted_tags (list[str]) if tag_model exists
      - predicted_tags_str (comma-separated)
    """
    pack = load(model_path)
    model = pack["model"]
    cols = pack["feature_columns"]

    X = df_features[cols].fillna(0.0)
    df_out = df_features.copy()

    # Ranker score
    df_out["ml_score"] = model.predict_proba(X)[:, 1]
    df_out = df_out.sort_values("ml_score", ascending=False)

    # Tag predictions (if present)
    tag_model = pack.get("tag_model", None)
    tag_classes = pack.get("tag_classes", []) or []

    if tag_model is not None and len(tag_classes) > 0:
        P = tag_model.predict_proba(X)  # (n, n_tags)

        def top_tags_row(p_row: np.ndarray) -> List[str]:
            idx = np.argsort(p_row)[::-1]
            picked = []
            for j in idx:
                if float(p_row[j]) < float(tag_min_p):
                    break
                picked.append(tag_classes[j])
                if len(picked) >= int(top_k_tags):
                    break
            return picked

        df_out["predicted_tags"] = [top_tags_row(p) for p in P]
        df_out["predicted_tags_str"] = df_out["predicted_tags"].apply(lambda arr: ", ".join(arr))
    else:
        df_out["predicted_tags"] = [[] for _ in range(len(df_out))]
        df_out["predicted_tags_str"] = ""

    return df_out

# OUTPUT_BASE = Path(
#     r"C:\Users\ILarikova\workspace\multispecies_facades_planner_AI\tests\data_training"
# )
# MODEL_OUT_DIR = OUTPUT_BASE / "models"

# model_path, n_samples = train_ranker_global(
#         output_base=OUTPUT_BASE,
#         model_out_dir=MODEL_OUT_DIR,
#         min_tag_count=3,        
#         train_tag_model=True,   
#     )

# print("Global model saved to:", model_path)
# print("Training samples used:", n_samples)

# # checking which buildings are evaluated
# df_fb = load_all_feedback_global(OUTPUT_BASE)
# print("Buildings in feedback:", df_fb["building_id"].unique())