"""
Loader for the UCI-HAR (Human Activity Recognition Using Smartphones) dataset.

Source: Anguita et al., 2013, "A Public Domain Dataset for Human Activity
Recognition Using Smartphones", ESANN.
Official host: https://archive.ics.uci.edu/dataset/240

Expected folder layout under data/raw/UCI_HAR/:
    train/X_train.txt   (7352 x 561 pre-extracted features, space-separated)
    train/y_train.txt   (7352 x 1  activity label, 1-6)
    train/subject_train.txt (7352 x 1 subject id, 1-30)
    test/X_test.txt, test/y_test.txt, test/subject_test.txt (same format)
    activity_labels.txt (maps 1-6 -> activity name)
    features.txt         (561 feature names)

Notes on this dataset:
    - Sensor: smartphone accelerometer + gyroscope, waist-mounted
    - Sampling rate: 50 Hz
    - 30 subjects, 6 activities
    - Data here is ALREADY windowed and feature-extracted by the original
      authors (2.56s windows, 50% overlap, 561 time+frequency domain
      features per window) -- there is no raw time-series in this
      particular distribution, which matters when we later try to build
      a *common* raw-signal pipeline across all four datasets (WISDM,
      PAMAP2, MHEALTH give raw time series; UCI-HAR here does not).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def load_uci_har(root: str | Path):
    """
    Load UCI-HAR pre-extracted feature data.

    Returns
    -------
    df : pd.DataFrame
        One row per window. Columns: 561 features (f0..f560), 'subject',
        'activity_id', 'activity_name', 'split' (train/test).
    """
    root = Path(root)

    frames = []
    for split in ("train", "test"):
        X = pd.read_csv(root / split / f"X_{split}.txt", sep=r"\s+", header=None)
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = pd.read_csv(root / split / f"y_{split}.txt", header=None, names=["activity_id"])
        subj = pd.read_csv(root / split / f"subject_{split}.txt", header=None, names=["subject"])

        activity_name = y["activity_id"].map(ACTIVITY_MAP).rename("activity_name")
        split_col = pd.Series(split, index=X.index, name="split")
        df = pd.concat([X, y, subj, activity_name, split_col], axis=1).copy()
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    return full


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/UCI_HAR"
    df = load_uci_har(root)
    print(f"Loaded UCI-HAR: {df.shape[0]} windows, {df.shape[1]} columns")
    print(f"Subjects: {df['subject'].nunique()}  Activities: {df['activity_name'].nunique()}")
    print(df["activity_name"].value_counts())
