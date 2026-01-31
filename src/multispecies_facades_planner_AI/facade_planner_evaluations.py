import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

#internal
from multispecies_facades_planner_AI import facade_planner as fp
from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI.facade_planner_trainer_global import load_all_feedback_global

 
def log_feedback(
    *,
    out_path: str | Path,
    iteration_id: str,
    building_id: str,
    species: str,
    candidate_id: str,
    decision: str,  # "accept" or "reject"
    tags: Optional[List[str]] = None,
    comment: Optional[str] = None,
    source: str = "manual",   # "manual" | "auto_unselected"
    strength: int = 2,        # 2=strong (explicit), 1=weak (unselected)
):
    if decision not in {"accept", "reject"}:
        raise ValueError("decision must be 'accept' or 'reject'")
    if source not in {"manual", "auto_unselected"}:
        raise ValueError("source must be 'manual' or 'auto_unselected'")
    if strength not in {1, 2}:
        raise ValueError("strength must be 1 or 2")

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration_id": iteration_id,
        "building_id": building_id,
        "species": species,
        "candidate_id": candidate_id,
        "decision": decision,
        "source": source,
        "strength": strength,
        "tags": tags or [],
        "comment": comment,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_accepts_and_rejects(
    *,
    df_iter: pd.DataFrame,
    accepted_candidate_ids: list[str],
    rejected_candidate_ids: Optional[list[str]] = None,
    feedback_path: str | Path,
    iteration_id: str,
    building_id: str,
    species: str,
    accept_tags=None,
    accept_comment=None,
    reject_tags=None,
    reject_comment=None,
    auto_reject_unselected: bool = False,  # if True: weak negatives
):
    accepted_set = set(accepted_candidate_ids or [])
    rejected_set = set(rejected_candidate_ids or [])

    overlap = accepted_set & rejected_set
    if overlap:
        raise ValueError(f"Candidate(s) appear in both accept and reject: {sorted(overlap)}")

    # 1) strong accepts (manual)
    for cid in accepted_set:
        log_feedback(
            out_path=feedback_path,
            iteration_id=iteration_id,
            building_id=building_id,
            species=species,
            candidate_id=cid,
            decision="accept",
            tags=accept_tags or [],
            comment=accept_comment,
            source="manual",
            strength=2,
        )

    # 2) strong rejects (manual)
    for cid in rejected_set:
        log_feedback(
            out_path=feedback_path,
            iteration_id=iteration_id,
            building_id=building_id,
            species=species,
            candidate_id=cid,
            decision="reject",
            tags=reject_tags or [],
            comment=reject_comment,
            source="manual",
            strength=2,
        )

    # 3) optional weak rejects for everything else
    if auto_reject_unselected:
        all_cids = df_iter["candidate_id"].tolist()
        for cid in all_cids:
            if cid in accepted_set or cid in rejected_set:
                continue
            log_feedback(
                out_path=feedback_path,
                iteration_id=iteration_id,
                building_id=building_id,
                species=species,
                candidate_id=cid,
                decision="reject",
                source="auto_unselected",
                strength=1,
            )

def load_all_feature_csvs(output_base: Path, building_id: str) -> pd.DataFrame:
    building_dir = output_base / building_id
    paths = sorted(building_dir.glob("iter_*/features.csv"))
    if not paths:
        raise FileNotFoundError(f"No features.csv found under {building_dir}")

    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def load_feedback_jsonl(feedback_path: Path) -> pd.DataFrame:
    if not feedback_path.exists():
        raise FileNotFoundError(f"Missing feedback file: {feedback_path}")

    rows = []
    with feedback_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Backward compatibility (older logs without these fields)
    if "source" not in df.columns:
        df["source"] = "manual"
    if "strength" not in df.columns:
        df["strength"] = 2

    df["label"] = (df["decision"] == "accept").astype(int)
    return df

def train_ranker_for_building(
    output_base: Path,
    building_id: str,
    model_out_dir: Path,
):
    features = load_all_feature_csvs(output_base, building_id)
    feedback = load_feedback_jsonl(output_base / building_id / "feedback.jsonl")

    # join labels to features
    df = features.merge(
        feedback[["iteration_id", "candidate_id", "accepted"]],
        on=["iteration_id", "candidate_id"],
        how="inner",
    )

    if df.empty:
        raise ValueError("No matched (features ↔ feedback) rows. Did you log feedback for those iterations?")

    # choose numeric feature columns
    drop_cols = {
        "timestamp", "species", "building_id", "wall_id", "iteration_id", "candidate_id"
    }
    X = df[[c for c in df.columns if c not in drop_cols]]

    # handle missing values (MVP)
    X = X.fillna(-999.0)

    y = df["accepted"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_out_dir / f"ranker_{building_id}.joblib"
    dump({"model": model, "feature_columns": list(X.columns)}, model_path)

    return model_path, df.shape[0]

def resolve_candidate_id(df_iter, short_id: str) -> str:
    short_id = short_id.strip().lower()
    matches = df_iter[df_iter["candidate_id"].str.lower().str.startswith(short_id)]
    if len(matches) == 0:
        raise ValueError(f"No candidate_id starts with '{short_id}' in this iteration.")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous short id '{short_id}' matches {len(matches)} candidates. "
            f"Use a longer prefix."
        )
    return matches.iloc[0]["candidate_id"]

#new 
def read_feedback_table(path: str | Path) -> pd.DataFrame:
    """Read your table as TSV (tab-separated)."""
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        engine="python",
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_tags_cell(x) -> List[str]:
    """
    Accepts cells like:
      '"cold_facade", "high_height", "garden"'
      'cold_facade, "high_height", "garden"'
      '["cold_facade","high_height"]'
      '' / NaN
    Returns a clean list of strings.
    """
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []

    # JSON-like list
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass

    # Strip outer quotes
    s = s.strip().strip('"').strip("'")

    # Prefer quoted tokens
    quoted = re.findall(r'"([^"]+)"', s)
    if quoted:
        return [t.strip() for t in quoted if t.strip()]

    # fallback: comma split
    return [t.strip().strip('"').strip("'") for t in s.split(",") if t.strip()]


def resolve_candidate_id_from_features(features_csv, short_id: str) -> str:
    short_id = str(short_id).strip().lower()
    if not short_id:
        raise ValueError("Empty short_id")

    df_iter = pd.read_csv(features_csv)
    if "candidate_id" not in df_iter.columns:
        raise ValueError(f"'candidate_id' column missing in {features_csv}")

    cand = df_iter["candidate_id"].astype(str)
    cand_lower = cand.str.lower()

    # 1) startswith (fast + safest)
    m = df_iter[cand_lower.str.startswith(short_id)]
    if len(m) == 1:
        return str(m.iloc[0]["candidate_id"])
    if len(m) > 1:
        raise ValueError(
            f"Ambiguous short id '{short_id}' matches {len(m)} candidate_id (startswith) in {features_csv}."
        )

    # 2) contains (more forgiving)
    m = df_iter[cand_lower.str.contains(short_id, regex=False)]
    if len(m) == 1:
        return str(m.iloc[0]["candidate_id"])
    if len(m) > 1:
        raise ValueError(
            f"Ambiguous short id '{short_id}' matches {len(m)} candidate_id (contains) in {features_csv}. "
            f"Use a longer prefix."
        )

    # 3) nothing found
    raise ValueError(f"No candidate_id contains '{short_id}' in {features_csv}")

# 2) CSV -> feedback.jsonl files

TAG_ALIASES = {
    "pedestrial_local": "pedestrian_local",
}

def normalize_tags(tags: list[str]) -> list[str]:
    out = []
    for t in (tags or []):
        t = str(t).strip().lower()
        if not t:
            continue
        t = TAG_ALIASES.get(t, t)
        # keep only real-ish tokens (letters/digits/_)
        if not re.search(r"[a-z0-9_]", t):
            continue
        out.append(t)

    # de-dup preserve order
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def parse_tags_cell(x) -> List[str]:
    """
    Accepts cells like:
      '"cold_facade", "high_height", "garden"'
      'cold_facade, "high_height", "garden"'
      '["cold_facade","high_height"]'
      '' / NaN
    Returns a clean list of strings (no ',' garbage).
    """
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []

    # JSON-like list
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return normalize_tags([str(t).strip() for t in arr])
        except Exception:
            pass

    # Prefer quoted tokens
    quoted = re.findall(r'"([^"]+)"', s)
    if quoted:
        return normalize_tags([t.strip() for t in quoted])

    # fallback: comma split
    raw = [t.strip().strip('"').strip("'") for t in s.split(",")]
    return normalize_tags(raw)


def import_feedback_csv_to_jsonl(
    *,
    csv_path: str | Path,
    output_base: str | Path,
    overwrite: bool = False,
    strict: bool = True,
) -> Tuple[int, int]:
    """
    Converts CSV feedback table into per-building feedback.jsonl.
    - best_pick_* -> accept (manual strong)
    - worst_pick_* -> reject (manual strong)
    - resolves short ids against OUTPUT_BASE/<building>/<iteration>/features.csv
    - attaches tags from the nearest tags column between this pick_* and the next pick_*

    Returns: (written_records, errors_count)
    """
    csv_path = Path(csv_path)
    output_base = Path(output_base)

    df = read_feedback_table(csv_path)

    required = {"building", "iteration", "species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV is missing required columns: {sorted(missing)}")

    cols_list = list(df.columns)
    cols_lower = [c.lower() for c in cols_list]

    # pick columns
    best_cols = [c for c in cols_list if c.lower().startswith("best_pick_")]
    worst_cols = [c for c in cols_list if c.lower().startswith("worst_pick_")]
    if not best_cols and not worst_cols:
        raise ValueError("No best_pick_* or worst_pick_* columns found.")

    pick_prefixes = ("best_pick_", "worst_pick_")

    def _is_pick_col(col_lower: str) -> bool:
        return col_lower.startswith(pick_prefixes)

    def find_tags_col_for(pick_col: str, decision: str) -> Optional[str]:
        """
        Search forward from pick_col until next pick column (or end),
        and return the first matching tags column.
        decision='accept' prefers accept_tags, then tags.
        decision='reject' prefers reject_tags, then tags.
        """
        i = cols_list.index(pick_col)

        preferred = ["accept_tags", "tags"] if decision == "accept" else ["reject_tags", "tags"]

        # scan until next pick column
        for j in range(i + 1, len(cols_list)):
            if _is_pick_col(cols_lower[j]):
                break
            name = cols_lower[j]
            if name in preferred:
                return cols_list[j]

        # fallback: try within the rest of the row segment again for any *_tags
        for j in range(i + 1, len(cols_list)):
            if _is_pick_col(cols_lower[j]):
                break
            if cols_lower[j].endswith("_tags"):
                return cols_list[j]

        return None

    # overwrite existing feedback files if requested
    if overwrite:
        for bdir in output_base.iterdir():
            if bdir.is_dir():
                fb = bdir / "feedback.jsonl"
                if fb.exists():
                    fb.unlink()

    written = 0
    errors = 0

    for idx, row in df.iterrows():
        building_id = str(row["building"]).strip()
        iteration_id = str(row["iteration"]).strip()
        species = str(row["species"]).strip()

        iter_dir = output_base / building_id / iteration_id
        features_csv = iter_dir / "features.csv"

        if not features_csv.exists():
            msg = f"[Row {idx}] Missing features.csv: {features_csv}"
            if strict:
                raise FileNotFoundError(msg)
            print("WARN:", msg)
            errors += 1
            continue

        building_dir = output_base / building_id
        building_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = building_dir / "feedback.jsonl"

        def add_record(short_candidate, decision: str, tags_cell):
            nonlocal written, errors
            if short_candidate is None:
                return
            short_candidate = str(short_candidate).strip()
            if not short_candidate:
                return

            try:
                full_cid = resolve_candidate_id_from_features(features_csv, short_candidate)
            except Exception as e:
                msg = f"[Row {idx}] Could not resolve '{short_candidate}' in {features_csv}: {e}"
                if strict:
                    raise
                print("WARN:", msg)
                errors += 1
                return

            tags = parse_tags_cell(tags_cell)

            rec = {
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_id": iteration_id,
                "building_id": building_id,
                "species": species,
                "candidate_id": full_cid,
                "decision": decision,     # accept / reject
                "source": "manual",
                "strength": 2,
                "tags": tags,
                "comment": None,
            }

            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            written += 1

        # best picks -> accept
        for c in best_cols:
            tags_col = find_tags_col_for(c, "accept")
            add_record(row.get(c), "accept", row.get(tags_col) if tags_col else None)

        # worst picks -> reject
        for c in worst_cols:
            tags_col = find_tags_col_for(c, "reject")
            add_record(row.get(c), "reject", row.get(tags_col) if tags_col else None)

    return written, errors

