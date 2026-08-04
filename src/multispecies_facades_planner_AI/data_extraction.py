

from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional, Union


def load_species_core_as_dict(
    excel_path: str,
    species_name: str,
    *,
    species_column: str = "specie_name_EN",
    suffix: str = "_core",
    fields: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads a single species core row (e.g. 'house_sparrow_core') from an Excel dataset
    and returns:
        { species_name: {field: value, ...} }

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    species_name : str
        Logical species name, e.g. 'house_sparrow'.
    species_column : str
        Column that contains identifiers like 'house_sparrow_core'.
    suffix : str
        Suffix appended to species_name to find the core row.
    fields : list[str] | None
        Which columns to extract. If None, uses a default set.

    Raises
    ------
    KeyError : if species_column doesn't exist
    ValueError : if the core row isn't found

    Returns
    -------
    dict
        {species_name: {field: value}}
    """

    if fields is None:
        fields = [
            "nest_height",
            "colonie",
            "if_colonial_distance_to_next_nest",
            "colonie_size",
            "diet_type",
            "distance_to_roof",
            "far_from_windows",
        ]

    df = pd.read_excel(excel_path)

    if species_column not in df.columns:
        raise KeyError(
            f"Column '{species_column}' not found. Available columns: {df.columns.tolist()}"
        )

    row_id = f"{species_name}{suffix}"

    filtered = df.loc[df[species_column].astype(str).str.strip() == row_id]

    if filtered.empty:
        raise ValueError(f"Species core row '{row_id}' not found in column '{species_column}'")

    row = filtered.iloc[0]

    # extract only fields that exist; set missing ones to None (optional behavior)
    out_fields: Dict[str, Any] = {}
    for f in fields:
        if f in df.columns:
            val = row[f]
            # normalize pandas NaN to None for JSON friendliness
            if pd.isna(val):
                val = None
            out_fields[f] = val
        else:
            out_fields[f] = None

    return {species_name: out_fields}


def load_species_training_as_dict(
    excel_path: str,
    species_name: str,
    *,
    species_column: str = "specie_name_EN",
    suffix: str = "_core",
    fields: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads a single species training row, e.g. 'house_sparrow_core',
    and returns:
 
        { species_name: {field: value, ...} }
    """
 
    if fields is None:
        fields = [
            "taxa",
 
            "nest_height",
            "colonie",
            "colonie_size_local",
            "colonie_size",
            "distance_to_next_nest",
            "ditance_to_next_nest",  # old typo fallback
            "if_colonial_distance_to_next_nest",
 
            "preferred_orientation",
            "avoided_orientation",
            "number_of_orientations",  # how many walls the species uses: 1, 2, 2-3, or 3-4
 
            "distance_to_edges",
            "distance_to_roof",
            "far_from_windows",

            "time_activity",
            "nest_use_window",
            "temperature_optimum_in_nest_box",
 
            "species_noise",
            "tolerance_to_human",
            "dirt",
 
            "diet_type",
            "number_of_individual_nest_boxes_on_building",
            "distance_between_nest_boxes",
            "distance_to_next_nest_of_other_species",
        ]
 
    COLUMN_ALIASES = {
        "colonie_size_local": [
            "colonie_size_local",
            'if colonie == "yes"\ncolonie_size_local',
            'if colonie == ""yes""\ncolonie_size_local',
            'if colonie == "yes" colonie_size_local',
            'if colonie == ""yes"" colonie_size_local',
        ],
 
        "distance_to_next_nest": [
            "distance_to_next_nest",
            'if colonie == "yes"\ndistance_to_next_nest',
            'if colonie == ""yes""\ndistance_to_next_nest',
            'if colonie == "yes" distance_to_next_nest',
            'if colonie == ""yes"" distance_to_next_nest',
 
            # fallback old typo names
            "ditance_to_next_nest",
            'if colonie == "yes"\nditance_to_next_nest',
            'if colonie == ""yes""\nditance_to_next_nest',
        ],
 
        "ditance_to_next_nest": [
            "ditance_to_next_nest",
            'if colonie == "yes"\nditance_to_next_nest',
            'if colonie == ""yes""\nditance_to_next_nest',
 
            # fallback corrected names
            "distance_to_next_nest",
            'if colonie == "yes"\ndistance_to_next_nest',
            'if colonie == ""yes""\ndistance_to_next_nest',
        ],
 
        "if_colonial_distance_to_next_nest": [
            "if_colonial_distance_to_next_nest",
            "distance_to_next_nest",
            'if colonie == "yes"\ndistance_to_next_nest',
            'if colonie == ""yes""\ndistance_to_next_nest',
            'if colonie == "yes" distance_to_next_nest',
            'if colonie == ""yes"" distance_to_next_nest',
 
            # fallback old typo
            "ditance_to_next_nest",
            'if colonie == "yes"\nditance_to_next_nest',
            'if colonie == ""yes""\nditance_to_next_nest',
        ],
 
        "number_of_individual_nest_boxes_on_building": [
            "number_of_individual_nest_boxes_on_building",
            'if colonie == "no"\nnumber_of_individual_nest_boxes_on_building',
            'if colonie == ""no""\nnumber_of_individual_nest_boxes_on_building',
            'if colonie == "no" number_of_individual_nest_boxes_on_building',
            'if colonie == ""no"" number_of_individual_nest_boxes_on_building',
        ],
 
        "distance_between_nest_boxes": [
            "distance_between_nest_boxes",
            'if colonie == "no"\ndistance_between_nest_boxes',
            'if colonie == ""no""\ndistance_between_nest_boxes',
            'if colonie == "no" distance_between_nest_boxes',
            'if colonie == ""no"" distance_between_nest_boxes',
        ],

        # interspecies nest spacing — colony variant ("if colonie == yes") and
        # solitary variant ("if colonie == no", stored under a typo'd column name
        # missing "to", same typo pattern as ditance_to_next_nest above).
        "distance_to_next_nest_of_other_species": [
            "distance_to_next_nest_of_other_species",
            'if colonie == "yes"\ndistance_to_next_nest_of_other_species',
            'if colonie == ""yes""\ndistance_to_next_nest_of_other_species',
            'if colonie == "yes" distance_to_next_nest_of_other_species',
            'if colonie == ""yes"" distance_to_next_nest_of_other_species',

            "distance_next_nest_of_other_species",
            'if colonie == "no"\ndistance_next_nest_of_other_species',
            'if colonie == ""no""\ndistance_next_nest_of_other_species',
            'if colonie == "no" distance_next_nest_of_other_species',
            'if colonie == ""no"" distance_next_nest_of_other_species',
        ],
    }
 
    df = pd.read_excel(excel_path)
 
    if species_column not in df.columns:
        raise KeyError(
            f"Column '{species_column}' not found. Available columns: {df.columns.tolist()}"
        )
 
    row_id = f"{species_name}{suffix}"
 
    filtered = df.loc[
        df[species_column].astype(str).str.strip() == row_id
    ]
 
    if filtered.empty:
        raise ValueError(
            f"Species core row '{row_id}' not found in column '{species_column}'"
        )
 
    row = filtered.iloc[0]
 
    out_fields: Dict[str, Any] = {}
 
    for f in fields:
        possible_cols = COLUMN_ALIASES.get(f, [f])
 
        found_col = None
        for col in possible_cols:
            if col in df.columns:
                found_col = col
                break
 
        if found_col is not None:
            val = row[found_col]
 
            if pd.isna(val):
                val = None
 
            out_fields[f] = val
        else:
            out_fields[f] = None
 
    return {species_name: out_fields}