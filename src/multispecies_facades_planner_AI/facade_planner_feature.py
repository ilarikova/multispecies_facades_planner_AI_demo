import json
import uuid
from pathlib import Path
from typing import List, Sequence, Dict, Any
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Circle
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import proj3d

#internal
from multispecies_facades_planner_AI import facade_planner as fp
from multispecies_facades_planner_AI import facade_planner_functions as fpf
from multispecies_facades_planner_AI.data_extraction import load_species_core_as_dict, load_species_training_as_dict



 
def make_iteration_id(prefix="iter") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def derive_building_id(candidates):
    """
    Extract building id from wall_id like '5128-wall-00'.
    Assumes all candidates belong to the same building.
    """
    if not candidates:
        raise ValueError("No candidates provided; cannot derive building_id.")

    wall_id = candidates[0]["wall_id"]

    #robust split
    building_id = wall_id.split("-")[0]

    return building_id

def encode_species_traits(needs: dict) -> dict:
    #noise (ordinal) 
    noise_raw = (needs.get("species_noise") or "").strip().lower()
    NOISE_MAP = {"low": 0, "medium": 1, "high": 2}
    noise_level = NOISE_MAP.get(noise_raw, 0)

    # colonial (binary)
    colonial_raw = (needs.get("colonie") or "").strip().lower().replace('"', "")
    colonial = int(colonial_raw == "yes")

    # time of activity (multi-label) 
    toa_raw = (needs.get("time_of_activity") or "").lower()
    toa_parts = [p.strip() for p in toa_raw.split(",")]

    is_day_active = int("day" in toa_parts)
    is_evening_active = int("evening" in toa_parts)
    is_dusk_active = int("dusk" in toa_parts)

    # height preferences 
    min_height_pref = fpf.parse_min_height_m(needs.get("nest_height"))
    max_height_pref = fpf.parse_max_height_m(needs.get("nest_height"))

    min_height_pref = np.nan if min_height_pref is None else min_height_pref
    max_height_pref = np.nan if max_height_pref is None else max_height_pref

    return {
        "noise_level": noise_level,
        "colonial": colonial,
        "is_day_active": is_day_active,
        "is_evening_active": is_evening_active,
        "is_dusk_active": is_dusk_active,
        "preferred_height_min_m": min_height_pref,
        "preferred_height_max_m": max_height_pref,
    }


def build_feature_table(
    candidates: List[Dict[str, Any]],
    building_dict: dict,
    species_dict: dict,
    iteration_id: str,
    building_id: str,
) -> pd.DataFrame:
    """
    Convert candidates into a flat, numeric feature table.
    One row = one candidate.
    Species traits are derived from species_dict.
    """

    FLOOR_PREF_MAP = {
        "garden": 2,
        "pedestrian": 1,
        "pedestrian_local": 0,
    }

    rows = []

    #cache traits per species 
    species_trait_cache: Dict[str, Dict[str, Any]] = {}

    #cache neighbor features per wall (important: many candidates share one wall)
    neighbor_feat_cache: Dict[str, Dict[str, Any]] = {}

    def _traits_for(species_name: str) -> Dict[str, Any]:
        if species_name in species_trait_cache:
            return species_trait_cache[species_name]

        needs = species_dict.get(species_name)
        if needs is None:
            raise KeyError(f"Species '{species_name}' not found in species_dict.")

        traits = encode_species_traits(needs)
        species_trait_cache[species_name] = traits
        return traits

    def _neighbor_feats_for(wall_id: str) -> Dict[str, Any]:
        """
        Uses fpf.neighbor_floor_functions(...) output:
          [{"neighbor_id":..., "floor_function":..., "matched_side": "umin"/"umax", ...}, ...]
        Ignores neighbors without floor_function.
        """
        if wall_id in neighbor_feat_cache:
            return neighbor_feat_cache[wall_id]

        # defaults
        feats = {
            "neighbor_best_floor_preference": 0,
            "neighbor_mean_floor_preference": 0.0,
            "neighbor_has_garden": 0,
            "neighbor_garden_on_umin": 0,
            "neighbor_garden_on_umax": 0,
            "neighbor_best_on_umin": 0,
            "neighbor_best_on_umax": 0,
            "neighbor_count_considered": 0,
        }

        try:
            nbs = fpf.neighbor_floor_functions(building_dict, wall_id)
        except Exception:
            nbs = []

        prefs_all = []
        best_umin = 0
        best_umax = 0
        garden_umin = 0
        garden_umax = 0
        count = 0

        for nb in (nbs or []):
            ff = nb.get("floor_function")
            if not ff:
                continue

            ff_norm = str(ff).strip().lower()
            pref = FLOOR_PREF_MAP.get(ff_norm, 0)
            prefs_all.append(pref)
            count += 1

            side = (nb.get("matched_side") or "").strip().lower()  # "umin" / "umax"

            if side == "umin":
                best_umin = max(best_umin, pref)
                if ff_norm == "garden":
                    garden_umin = 1
            elif side == "umax":
                best_umax = max(best_umax, pref)
                if ff_norm == "garden":
                    garden_umax = 1

        if prefs_all:
            feats["neighbor_best_floor_preference"] = int(max(prefs_all))
            feats["neighbor_mean_floor_preference"] = float(np.mean(prefs_all))
            feats["neighbor_has_garden"] = int(max(prefs_all) == 2)

        feats["neighbor_garden_on_umin"] = int(garden_umin)
        feats["neighbor_garden_on_umax"] = int(garden_umax)
        feats["neighbor_best_on_umin"] = int(best_umin)
        feats["neighbor_best_on_umax"] = int(best_umax)
        feats["neighbor_count_considered"] = int(count)

        neighbor_feat_cache[wall_id] = feats
        return feats

    for c in candidates:
        wall_id = c["wall_id"]
        wall = building_dict.get(wall_id, {})

        xyz = c.get("xyz") or []
        zs = [p[2] for p in xyz] if xyz else []

        #geometry features
        mean_height = float(np.mean(zs)) if zs else np.nan
        min_height = float(np.min(zs)) if zs else np.nan
        max_height = float(np.max(zs)) if zs else np.nan

        #wall context
        floor_fn = (wall.get("floor_function") or "").strip().lower()

        is_garden = int(floor_fn == "garden")
        is_pedestrian = int(floor_fn == "pedestrian")
        is_pedestrian_local = int(floor_fn == "pedestrian_local")
        floor_preference = int(FLOOR_PREF_MAP.get(floor_fn, 0))

        # climate 
        climate_vals = wall.get("climate") or []
        climate_median = float(np.median(climate_vals)) if climate_vals else np.nan

        # species traits
        species_name = c["species"]
        traits = _traits_for(species_name)

        # neighbor features (computed per wall_id) 
        nb_feats = _neighbor_feats_for(wall_id)

        # sector features (3x3) 
        sector_feats = fpf.sector_features_3x3(building_dict, c)

        rows.append({
            # identifiers
            "iteration_id": iteration_id,
            "candidate_id": c["candidate_id"],
            "building_id": building_id,
            "wall_id": wall_id,
            "species": species_name,
            "seed": c.get("seed", np.nan),

            # wall context
            "wall_free_area_m2": fpf.free_wall_area(wall),
            "door_count": len(wall.get("doors") or {}),
            "is_pedestrian": is_pedestrian,
            "is_garden": is_garden,
            "is_pedestrian_local": is_pedestrian_local,
            "floor_preference": floor_preference,

            # environment
            "climate_median": climate_median,
            "dist_to_street_m": wall.get("dist_to_street_m", np.nan),

            # candidate geometry
            "mean_height_m": mean_height,
            "min_height_m": min_height,
            "max_height_m": max_height,

            # neighbor context
            **nb_feats,

            # 3x3 sector context
            **sector_feats,

            # species traits (encoded)
            **traits,

            # bookkeeping
            "timestamp": datetime.utcnow().isoformat(),
        })

    return pd.DataFrame(rows)

