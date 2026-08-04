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

# NOTE: a module-level `building_dict = load_building_dict(<absolute path into
# the old multispecies_facades_planner repo>)` used to sit here. It was never
# read anywhere in this module and made importing it fail on any other machine.
# Removed 2026-08-04 — load_building_dict above is still available to callers.

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def empty_orientation_set() -> set:
    return set()


def clean_orientation_text(raw: Any) -> str:
    if raw is None:
        return ""

    s = str(raw).strip().lower()

    replacements = {
        "-": "-",
        "–": "-",
        "—": "-",
        "/": ",",
        ";": ",",
        "facing": "",
        "facing,": "",
        "north- to east": "north, east",
        "north to east": "north, east",
        "east-southeast": "east, southeast",
        "east southeast": "east, southeast",
        "southwest - southeast": "southwest, southeast",
        "northeast - southeast": "northeast, southeast",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    return s


def parse_orientation_set(raw: Any) -> set:
    """
    Parse messy orientation text into simplified direction labels:
    N, NE, E, SE, S, SW, W, NW
    """
    s = clean_orientation_text(raw)

    if not s:
        return set()

    mapping = {
        "northeast": "NE",
        "north-east": "NE",
        "southeast": "SE",
        "south-east": "SE",
        "southwest": "SW",
        "south-west": "SW",
        "northwest": "NW",
        "north-west": "NW",
        "north": "N",
        "east": "E",
        "south": "S",
        "west": "W",
    }

    found = set()

    # Important: longer words first, so "southeast" is not split into "south" + "east"
    for word, label in mapping.items():
        if word in s:
            found.add(label)

    return found


def has_no_orientation_preference(raw: Any) -> bool:
    s = clean_orientation_text(raw)

    if not s:
        return True

    no_pref_markers = [
        "no preference",
        "no clear preference",
        "no clear",
        "not clear",
        "any",
        "none",
        "unknown",
    ]

    return any(marker in s for marker in no_pref_markers)


def orientation_match(
    wall_orientation: Any,
    preferred_orientation_raw: Any,
) -> float:
    """
    Ternary match encoding:
    1 = wall orientation matches species preference
    0 = wall orientation conflicts with species preference
    NaN = species has no orientation preference
    """
    preferred = parse_orientation_set(preferred_orientation_raw)

    if has_no_orientation_preference(preferred_orientation_raw) or not preferred:
        return np.nan

    wall = str(wall_orientation).strip().upper() if wall_orientation is not None else ""

    if wall not in DIRECTIONS:
        return np.nan

    return int(wall in preferred)

def parse_avoided_orientation_set(raw: Any) -> set:
    """
    Parse avoided orientation from messy text.
    Example: '(west)' -> {'W'}
    """
    s = clean_orientation_text(raw)

    if not s:
        return set()

    # if avoided orientation is marked with brackets, extract only bracket content
    bracket_parts = re.findall(r"\((.*?)\)", s)
    if bracket_parts:
        s = ",".join(bracket_parts)

    return parse_orientation_set(s)


def parse_month_window(raw):
    if raw is None:
        return np.nan, np.nan

    s = str(raw).replace(".", "").strip().lower()
    if not s:
        return np.nan, np.nan

    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    found = [
        num
        for name, num in month_map.items()
        if name in s
    ]

    if not found:
        return np.nan, np.nan

    return min(found), max(found)


def encode_species_traits(needs: dict) -> dict:
    """
    Encode species-specific traits from the raw species-table row into the
    ML-friendly feature names used in the training CSVs.

    `needs` is the raw Excel row as returned by
    data_extraction.load_species_training_as_dict — its keys are sheet column
    names ("species_noise", "time_activity", "temperature_optimum_in_nest_box"),
    NOT the feature names ("noise_level", "is_day_active", "nest_temp_min_c").
    Anything reading needs.get("noise_level") directly gets None; that is what
    made 17 of the model's features dead at inference time.

    This lives here so the training generator and the inference pipeline can
    share one implementation. facade_planner_feature.encode_species_traits is
    an older copy of the same logic; it cannot be imported from the inference
    side because that module runs the full training generation at import time.
    The two are verified to agree against the exported CSVs.
    """
    # imported lazily: facade_planner imports this module, so a module-level
    # import here would be circular.
    from multispecies_facades_planner_AI import facade_planner_functions as fpf
    from multispecies_facades_planner_AI import facade_planner as fp

    LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}

    def _clean(key):
        return (
            str(needs.get(key) or "")
            .strip()
            .lower()
            .replace('"', "")
            .replace("\xa0", " ")
            .strip()
        )

    # taxa
    taxa = str(needs.get("taxa") or "").strip().lower().replace('"', "") or None
    is_bird = int(taxa == "bird")
    is_bat = int(taxa == "bat")

    # ordinals
    noise_level = LEVEL_MAP.get(_clean("species_noise"), np.nan)
    human_tolerance_level = LEVEL_MAP.get(_clean("tolerance_to_human"), np.nan)

    dirt_raw = _clean("dirt").replace("–", "-").replace("—", "-")
    dirt_level = {**LEVEL_MAP, "low-medium": 0.5, "medium-high": 1.5}.get(dirt_raw, np.nan)

    # colonial
    colonial = int(_clean("colonie") == "yes")

    if colonial == 1:
        colony_min, colony_max = fpf.parse_min_max_count(needs.get("colonie_size_local"))
        colony_min = np.nan if colony_min is None else colony_min
        colony_max = np.nan if colony_max is None else colony_max

        dist_raw = (
            needs.get("distance_to_next_nest")
            or needs.get("if_colonial_distance_to_next_nest")
            or needs.get("distance_between_nest_boxes")
        )
        nest_dist_min_m, nest_dist_max_m = fp.parse_next_nest_distance_m(
            dist_raw,
            default=(1.0, 1.5),
            close_default=(0.0, 0.5),
        )
    else:
        colony_min = colony_max = np.nan
        nest_dist_min_m = nest_dist_max_m = np.nan

    # time of activity multi-label
    toa_raw = str(needs.get("time_activity") or "").lower()
    is_day_active = int("day" in toa_raw)
    is_evening_active = int("evening" in toa_raw)
    is_dusk_active = int("dusk" in toa_raw)
    is_night_active = int("night" in toa_raw)

    # nest use window
    nest_use_start, nest_use_end = parse_month_window(needs.get("nest_use_window"))
    nest_use_duration = (
        nest_use_end - nest_use_start + 1
        if not np.isnan(nest_use_start) and not np.isnan(nest_use_end)
        else np.nan
    )

    # height preferences
    min_height_pref = fpf.parse_min_height_m(needs.get("nest_height"))
    max_height_pref = fpf.parse_max_height_m(needs.get("nest_height"))
    min_height_pref = np.nan if min_height_pref is None else min_height_pref
    max_height_pref = np.nan if max_height_pref is None else max_height_pref

    # edge / roof / window proximity preferences
    prefers_edges = int(_clean("distance_to_edges") == "as close as possible")
    prefers_roof = int(_clean("distance_to_roof") == "as close as possible")
    far_from_windows_important = int(_clean("far_from_windows") == "important")

    # orientation preferences
    preferred_set = parse_orientation_set(needs.get("preferred_orientation"))
    avoided_set = parse_avoided_orientation_set(needs.get("avoided_orientation"))

    # nest temperature optimum
    temp_min, temp_max = fpf.parse_min_max_numeric(
        needs.get("temperature_optimum_in_nest_box")
    )
    temp_min = np.nan if temp_min is None else temp_min
    temp_max = np.nan if temp_max is None else temp_max

    return {
        "taxa": taxa,
        "is_bird": is_bird,
        "is_bat": is_bat,

        "noise_level": noise_level,
        "human_tolerance_level": human_tolerance_level,
        "dirt": dirt_level,

        "colonial": colonial,
        "colony_size_local_min": colony_min,
        "colony_size_local_max": colony_max,

        "nest_distance_min_m": nest_dist_min_m,
        "nest_distance_max_m": nest_dist_max_m,

        "is_day_active": is_day_active,
        "is_evening_active": is_evening_active,
        "is_dusk_active": is_dusk_active,
        "is_night_active": is_night_active,

        "nest_use_start_month": nest_use_start,
        "nest_use_end_month": nest_use_end,
        "nest_use_duration": nest_use_duration,

        "preferred_height_min_m": min_height_pref,
        "preferred_height_max_m": max_height_pref,

        "prefers_edges_proximity": prefers_edges,
        "prefers_roof_proximity": prefers_roof,
        "far_from_windows_important": far_from_windows_important,

        "preferred_orientation": ",".join(sorted(preferred_set)) if preferred_set else None,
        "avoided_orientation": ",".join(sorted(avoided_set)) if avoided_set else None,

        "nest_temp_min_c": temp_min,
        "nest_temp_max_c": temp_max,
    }