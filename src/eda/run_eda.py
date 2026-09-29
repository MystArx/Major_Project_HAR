"""
Exploratory data analysis across the HAR datasets.

Generates, per dataset (where data is available):
    - class distribution bar chart
    - subject count summary
    - for datasets with raw time series (WISDM has none in the
      transformed release we have; add PAMAP2/MHEALTH once downloaded),
      a sample sensor trace plot

Run from the project root:
    python src/eda/run_eda.py

Outputs PNGs to notebooks/eda_outputs/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "loaders"))

OUT_DIR = ROOT / "notebooks" / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def class_distribution_plot(df, label_col, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[label_col].value_counts().sort_values(ascending=True)
    ax.barh(counts.index.astype(str), counts.values, color="#1F4E79")
    ax.set_xlabel("Number of windows / samples")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"  saved {fname}")


def subject_summary(df, subject_col, label_col, name):
    n_subj = df[subject_col].nunique()
    per_subj = df.groupby(subject_col).size()
    print(f"  {name}: {n_subj} subjects | windows/samples per subject: "
          f"min={per_subj.min()}, median={int(per_subj.median())}, max={per_subj.max()}")


def summarize_dataset(name, df, label_col, subject_col):
    print(f"\n=== {name} ===")
    print(f"  Shape: {df.shape}")
    subject_summary(df, subject_col, label_col, name)
    class_distribution_plot(df, label_col, f"{name}: class distribution", f"{name.lower()}_class_dist.png")


def main():
    # ---- UCI-HAR ----
    try:
        from uci_har import load_uci_har
        uci = load_uci_har(ROOT / "data" / "raw" / "UCI_HAR")
        summarize_dataset("UCI-HAR", uci, "activity_name", "subject")
    except Exception as e:
        print(f"UCI-HAR skipped: {e}")
        uci = None

    # ---- WISDM ----
    try:
        from wisdm import load_wisdm
        wisdm = load_wisdm(ROOT / "data" / "raw" / "WISDM")
        summarize_dataset("WISDM", wisdm, "activity_name", "subject")
    except Exception as e:
        print(f"WISDM skipped: {e}")
        wisdm = None

    # ---- PAMAP2 (will only run once you've downloaded it) ----
    try:
        from pamap2 import load_pamap2
        pamap2 = load_pamap2(ROOT / "data" / "raw" / "PAMAP2")
        summarize_dataset("PAMAP2", pamap2, "activity_name", "subject")
    except FileNotFoundError as e:
        print(f"\nPAMAP2 skipped (not downloaded yet): {e}")
        pamap2 = None

    # ---- MHEALTH (will only run once you've downloaded it) ----
    try:
        from mhealth import load_mhealth
        mhealth = load_mhealth(ROOT / "data" / "raw" / "MHEALTH")
        summarize_dataset("MHEALTH", mhealth, "activity_name", "subject")
    except FileNotFoundError as e:
        print(f"\nMHEALTH skipped (not downloaded yet): {e}")
        mhealth = None

    # ---- Cross-dataset comparison table (the dataset-heterogeneity table
    #      that goes straight into the report / mid-sem slides) ----
    rows = []
    if uci is not None:
        rows.append(["UCI-HAR", "Smartphone (waist)", uci["subject"].nunique(),
                     uci["activity_name"].nunique(), "50 Hz", "pre-extracted features (561)"])
    if wisdm is not None:
        rows.append(["WISDM", "Smartphone (pocket)", wisdm["subject"].nunique(),
                     wisdm["activity_name"].nunique(), "~20 Hz (nominal)", "pre-extracted features (43)"])
    if pamap2 is not None:
        rows.append(["PAMAP2", "3x IMU (hand/chest/ankle) + HR", pamap2["subject"].nunique(),
                     pamap2["activity_name"].nunique(), "100 Hz", "raw signal"])
    if mhealth is not None:
        rows.append(["MHEALTH", "3x IMU (chest/ankle/arm) + ECG", mhealth["subject"].nunique(),
                     mhealth["activity_name"].nunique(), "50 Hz", "raw signal"])

    summary_df = pd.DataFrame(
        rows, columns=["Dataset", "Sensors/Placement", "Subjects", "Activities", "Sampling Rate", "Data Form"]
    )
    summary_df.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    print("\n=== Cross-dataset summary (saved to dataset_summary.csv) ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
