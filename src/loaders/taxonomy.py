"""
Shared activity taxonomy across UCI-HAR, WISDM, PAMAP2, and MHEALTH.

See docs/activity_taxonomy.md for the full derivation and rationale.
This module is the single source of truth for native-label -> shared-label
mapping; loaders should NOT hardcode this mapping themselves.

Tier 1 (CORE4): WALKING, SITTING, STANDING, STAIRS -- all 4 datasets
Tier 2 (EXTENDED5): + LYING -- UCI-HAR, PAMAP2, MHEALTH only (no WISDM)
"""

CORE4 = ["WALKING", "SITTING", "STANDING", "STAIRS"]
EXTENDED5 = CORE4 + ["LYING"]

# Native activity_name (as produced by each dataset's own loader) -> shared label.
# Any native label not listed here maps to None and should be dropped.

UCI_HAR_TO_SHARED = {
    "WALKING": "WALKING",
    "WALKING_UPSTAIRS": "STAIRS",
    "WALKING_DOWNSTAIRS": "STAIRS",
    "SITTING": "SITTING",
    "STANDING": "STANDING",
    "LAYING": "LYING",
}

WISDM_TO_SHARED = {
    "Walking": "WALKING",
    "Jogging": None,          # no counterpart in UCI-HAR; excluded from both tiers
    "Upstairs": "STAIRS",
    "Downstairs": "STAIRS",
    "Sitting": "SITTING",
    "Standing": "STANDING",
}

PAMAP2_TO_SHARED = {
    "walking": "WALKING",
    "sitting": "SITTING",
    "standing": "STANDING",
    "ascending_stairs": "STAIRS",
    "descending_stairs": "STAIRS",
    "lying": "LYING",
    # everything else (running, cycling, nordic_walking, watching_tv,
    # computer_work, car_driving, vacuum_cleaning, ironing, folding_laundry,
    # house_cleaning, playing_soccer, rope_jumping) -> None, dropped
}

MHEALTH_TO_SHARED = {
    "walking": "WALKING",
    "sitting_relaxing": "SITTING",
    "standing_still": "STANDING",
    "climbing_stairs": "STAIRS",   # direction unknown, merged regardless
    "lying_down": "LYING",
    # everything else (waist_bends_forward, frontal_elevation_arms,
    # knees_bending, cycling, jogging, running, jump_front_back) -> None
}

LOADER_MAPS = {
    "UCI_HAR": UCI_HAR_TO_SHARED,
    "WISDM": WISDM_TO_SHARED,
    "PAMAP2": PAMAP2_TO_SHARED,
    "MHEALTH": MHEALTH_TO_SHARED,
}


def apply_shared_taxonomy(df, dataset_name: str, tier: str = "core4",
                           native_label_col: str = "activity_name"):
    """
    Add a 'shared_activity' column to df, mapping native labels to the
    shared taxonomy, and drop rows whose native label has no counterpart
    in the chosen tier.

    Parameters
    ----------
    df : DataFrame with a native activity-name column (as produced by the
         dataset's own loader, e.g. load_uci_har / load_wisdm / ...)
    dataset_name : one of "UCI_HAR", "WISDM", "PAMAP2", "MHEALTH"
    tier : "core4" or "extended5"
    native_label_col : name of the column holding native activity labels

    Returns
    -------
    filtered df with an added 'shared_activity' column, rows with no
    mapping (or outside the chosen tier) removed.
    """
    if dataset_name not in LOADER_MAPS:
        raise ValueError(f"Unknown dataset_name '{dataset_name}', expected one of {list(LOADER_MAPS)}")
    if tier not in ("core4", "extended5"):
        raise ValueError("tier must be 'core4' or 'extended5'")

    if tier == "extended5" and dataset_name == "WISDM":
        raise ValueError(
            "WISDM has no LYING class and is excluded from the extended5 tier "
            "by design -- see docs/activity_taxonomy.md section 3."
        )

    allowed = set(CORE4) if tier == "core4" else set(EXTENDED5)
    mapping = LOADER_MAPS[dataset_name]

    out = df.copy()
    out["shared_activity"] = out[native_label_col].map(mapping)
    out = out[out["shared_activity"].isin(allowed)]
    return out


if __name__ == "__main__":
    # quick self-check: print how many native classes survive per tier
    for name, mapping in LOADER_MAPS.items():
        core4_kept = sorted({v for v in mapping.values() if v in CORE4})
        ext5_kept = sorted({v for v in mapping.values() if v in EXTENDED5})
        print(f"{name}: core4 -> {core4_kept} | extended5 -> {ext5_kept}")
