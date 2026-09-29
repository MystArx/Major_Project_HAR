"""
Loader for the WISDM (Wireless Sensor Data Mining) Actitracker dataset,
transformed/feature-extracted release (WISDM_ar_v1.1_transformed.arff).

Source: Kwapisz, Weiss & Moore, 2011, "Activity Recognition Using Cell
Phone Accelerometers", ACM SIGKDD Explorations.
Official host: https://www.cis.fordham.edu/wisdm/dataset.php

Expected file under data/raw/WISDM/:
    WISDM_ar_v1.1_transformed.arff

Notes on this dataset:
    - Sensor: smartphone accelerometer only (no gyroscope), front pocket
    - Nominal sampling rate: 20 Hz (actual rate varies -- see DAGHAR
      benchmark paper's discussion of WISDM's trimodal sampling-rate
      distribution; something worth citing when we discuss cross-dataset
      heterogeneity)
    - ~36 users, 6 activities: Walking, Jogging, Sitting, Standing,
      Upstairs, Downstairs
    - This is the TRANSFORMED (feature-extracted) release: each row is a
      10-second window summarized into 43 statistical features (average,
      peak, distribution bins, etc per axis) rather than raw samples --
      same situation as UCI-HAR's pre-extracted release.
"""

from pathlib import Path
import re
import pandas as pd


def load_wisdm(root: str | Path):
    """
    Parse the WISDM transformed ARFF file into a DataFrame.

    Returns
    -------
    df : pd.DataFrame
        One row per 10s window. Columns include UNIQUE_ID, user,
        43 statistical features, and 'activity_name' (the ARFF 'class'
        attribute).
    """
    root = Path(root)
    arff_path = root / "WISDM_ar_v1.1_transformed.arff"

    attr_names = []
    data_lines = []
    in_data = False

    with open(arff_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip("\r\n")
            if not line.strip():
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if in_data:
                data_lines.append(line)
            elif line.lower().startswith("@attribute"):
                # Most lines:   @attribute "XAVG" numeric
                # Class line:   @attribute class{ "Walking" , ... }   (no quotes, no space before {)
                m = re.match(r'@attribute\s+"([^"]+)"', line, re.IGNORECASE)
                if not m:
                    m = re.match(r"@attribute\s+(\S+?)[\s{]", line, re.IGNORECASE)
                if m:
                    attr_names.append(m.group(1))

    # ARFF class attribute is literally named "class" in this file
    if attr_names[-1].lower() != "class":
        # fall back: last attribute is still assumed to be the label
        pass

    rows = [r.split(",") for r in data_lines]
    df = pd.DataFrame(rows, columns=attr_names)

    # numeric coercion for everything except identifiers/labels
    non_numeric = {"user", "class"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"class": "activity_name", "user": "subject"})
    df["subject"] = df["subject"].astype(str)
    return df


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/WISDM"
    df = load_wisdm(root)
    print(f"Loaded WISDM: {df.shape[0]} windows, {df.shape[1]} columns")
    print(f"Subjects: {df['subject'].nunique()}  Activities: {df['activity_name'].nunique()}")
    print(df["activity_name"].value_counts())
