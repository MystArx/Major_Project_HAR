"""
Loader for the PAMAP2 Physical Activity Monitoring dataset.

Source: Reiss & Stricker, 2012, "Introducing a New Benchmarked Dataset
for Activity Monitoring", ISWC.
Official host (UCI ML Repository): https://doi.org/10.24432/C5NW2H
Direct zip (as of writing): https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip



Expected folder layout under data/raw/PAMAP2/ once you extract the zip:
    Protocol/subject101.dat ... subject109.dat   (9 subjects)
    Optional/subject101.dat ... (a few subjects have a second session)

File format: each .dat file is whitespace-separated, NO header row.
Columns (54 total), per the dataset's official readme:
    0  : timestamp (s)
    1  : activityID (0 = transient/other, else 1-24, see ACTIVITY_MAP)
    2  : heart rate (bpm)
    3-19  : IMU hand   (17 cols: temp, 3-axis accel x2 scales, 3-axis gyro,
                          3-axis magnetometer, 4 orientation [invalid,
                          ignore per dataset readme])
    20-36 : IMU chest  (same 17-column layout)
    37-53 : IMU ankle  (same 17-column layout)

Sampling rate: 100 Hz (raw), heart rate ~9 Hz (forward-filled by device).
Sensor placement: wrist (hand), chest, ankle -- NOTE this is a genuinely
different sensing modality from UCI-HAR/WISDM (single phone) -- multiple
body-worn IMUs -- which is exactly why this dataset is useful for testing
generalization beyond simple phone-placement shifts.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ACTIVITY_MAP = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking", 9: "watching_tv",
    10: "computer_work", 11: "car_driving", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer",
    24: "rope_jumping",
}

# Column names for one 17-column IMU block
_IMU_COLS = [
    "temp",
    "acc16_x", "acc16_y", "acc16_z",
    "acc6_x", "acc6_y", "acc6_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "orient1", "orient2", "orient3", "orient4",  # invalid per PAMAP2 readme, dropped later
]

COLS = (
    ["timestamp", "activity_id", "heart_rate"]
    + [f"hand_{c}" for c in _IMU_COLS]
    + [f"chest_{c}" for c in _IMU_COLS]
    + [f"ankle_{c}" for c in _IMU_COLS]
)


def load_pamap2(root: str | Path, subjects=None, drop_transient=True):
    """
    Load and concatenate PAMAP2 subject files.

    Parameters
    ----------
    root : path to data/raw/PAMAP2 (containing a 'Protocol' subfolder)
    subjects : list of ints (e.g. [101, 102, ...]) or None for all found
    drop_transient : drop activity_id == 0 rows (no activity label)

    Returns
    -------
    df : pd.DataFrame with one row per 100Hz sample, columns as in COLS
         plus 'subject' and 'activity_name'.
    """
    root = Path(root)
    protocol_dir = root / "Protocol"
    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"{protocol_dir} not found. Download PAMAP2 from "
            "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring "
            "and extract so that data/raw/PAMAP2/Protocol/subject101.dat etc. exist."
        )

    files = sorted(protocol_dir.glob("subject1*.dat"))
    if subjects is not None:
        files = [f for f in files if int(f.stem.replace("subject", "")) in subjects]

    frames = []
    for f in files:
        subj_id = int(f.stem.replace("subject", ""))
        df = pd.read_csv(f, sep=r"\s+", header=None, names=COLS, na_values="NaN")
        df["subject"] = subj_id
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    if drop_transient:
        full = full[full["activity_id"] != 0]
    full["activity_name"] = full["activity_id"].map(ACTIVITY_MAP)

    # orientation columns are explicitly marked invalid in the PAMAP2 readme
    drop_cols = [c for c in full.columns if "orient" in c]
    full = full.drop(columns=drop_cols)
    return full


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/PAMAP2"
    df = load_pamap2(root)
    print(f"Loaded PAMAP2: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Subjects: {df['subject'].nunique()}  Activities: {df['activity_name'].nunique()}")
    print(df["activity_name"].value_counts())


