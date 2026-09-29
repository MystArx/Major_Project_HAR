"""
Loader for the MHEALTH (Mobile Health) dataset.

Source: Banos et al., 2014, "mHealthDroid: A Novel Framework for Agile
Development of Mobile Health Applications", IWAAL.
Official host (UCI ML Repository): https://doi.org/10.24432/C5TW22

Expected folder layout under data/raw/MHEALTH/ once extracted:
    mHealth_subject1.log ... mHealth_subject10.log   (10 subjects)

File format: tab-separated, NO header row, one row per 50 Hz sample.
Columns (24 total), per the dataset's official readme:
    0-2   : chest accelerometer (x, y, z), m/s^2
    3-4   : ECG lead 1, lead 2, mV
    5-7   : left-ankle accelerometer (x, y, z), m/s^2
    8-10  : left-ankle gyroscope (x, y, z), deg/s
    11-13 : left-ankle magnetometer (x, y, z), local
    14-16 : right-lower-arm accelerometer (x, y, z), m/s^2
    17-19 : right-lower-arm gyroscope (x, y, z), deg/s
    20-22 : right-lower-arm magnetometer (x, y, z), local
    23    : activity label (0 = null/unlabelled, else 1-12)

Sampling rate: 50 Hz.
Sensor placement: chest + left ankle + right lower arm (3 body-worn
sensor units) -- like PAMAP2, this is a multi-IMU wearable setup, NOT a
single smartphone, which is the key modality contrast we want against
UCI-HAR/WISDM.
"""

from pathlib import Path
import pandas as pd

ACTIVITY_MAP = {
    0: "null", 1: "standing_still", 2: "sitting_relaxing", 3: "lying_down",
    4: "walking", 5: "climbing_stairs", 6: "waist_bends_forward",
    7: "frontal_elevation_arms", 8: "knees_bending", 9: "cycling",
    10: "jogging", 11: "running", 12: "jump_front_back",
}

COLS = (
    ["chest_acc_x", "chest_acc_y", "chest_acc_z",
     "ecg_lead1", "ecg_lead2",
     "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
     "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
     "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
     "arm_acc_x", "arm_acc_y", "arm_acc_z",
     "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
     "arm_mag_x", "arm_mag_y", "arm_mag_z",
     "activity_id"]
)


def load_mhealth(root: str | Path, subjects=None, drop_null=True):
    """
    Load and concatenate MHEALTH subject log files.

    Parameters
    ----------
    root : path to data/raw/MHEALTH (containing mHealth_subjectN.log files)
    subjects : list of ints (1-10) or None for all found
    drop_null : drop activity_id == 0 rows (unlabelled transition periods)

    Returns
    -------
    df : pd.DataFrame with one row per 50Hz sample, columns as in COLS
         (minus activity_id, replaced by activity_name) plus 'subject'.
    """
    root = Path(root)
    files = sorted(root.glob("mHealth_subject*.log"))
    if not files:
        raise FileNotFoundError(
            f"No mHealth_subject*.log files found under {root}. Download from "
            "https://archive.ics.uci.edu/dataset/319/mhealth+dataset "
            "and extract so that data/raw/MHEALTH/mHealth_subject1.log etc. exist."
        )

    if subjects is not None:
        files = [f for f in files if int(f.stem.split("subject")[-1]) in subjects]

    frames = []
    for f in files:
        subj_id = int(f.stem.split("subject")[-1])
        df = pd.read_csv(f, sep=r"\s+", header=None, names=COLS)
        df["subject"] = subj_id
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    if drop_null:
        full = full[full["activity_id"] != 0]
    full["activity_name"] = full["activity_id"].map(ACTIVITY_MAP)
    full = full.drop(columns=["activity_id"])
    return full


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/MHEALTH"
    df = load_mhealth(root)
    print(f"Loaded MHEALTH: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Subjects: {df['subject'].nunique()}  Activities: {df['activity_name'].nunique()}")
    print(df["activity_name"].value_counts())


