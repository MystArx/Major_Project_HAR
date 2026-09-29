# Cross-Domain Generalization in Wearable Human Activity Recognition

B.Tech ECE Major Project — code and data pipeline.
See `docs/` for the full proposal document and this checkpoint's working
notes.

## Status (Checkpoint 1 — Mid-Sem 1)

| Dataset  | Status | Notes |
|----------|--------|-------|
| UCI-HAR  | ✅ Loaded & verified | 10,299 windows, 30 subjects, 6 activities. Pre-extracted 561-feature release (no raw signal). |
| WISDM    | ✅ Loaded & verified | 5,418 windows, 36 subjects, 6 activities. Pre-extracted 43-feature (ARFF) release. Real class imbalance present (Standing=246 vs Walking=2081). |
| PAMAP2   | ⏳ Loader ready, data not yet downloaded | ~700MB, raw 100Hz signal, 3 IMUs + HR. See `src/loaders/pamap2.py` docstring for download link. |
| MHEALTH  | ⏳ Loader ready, data not yet downloaded | ~40MB, raw 50Hz signal, 3 IMUs + ECG. See `src/loaders/mhealth.py` docstring for download link. |

## Project structure

```
data/
  raw/            <- dataset files go here (UCI_HAR/, WISDM/ already populated)
  processed/       <- windowed/normalized data will go here in Phase 2
src/
  loaders/          <- one loader module per dataset + shared taxonomy.py
  eda/              <- exploratory analysis script
notebooks/
  eda_outputs/      <- generated plots + dataset_summary.csv
docs/
  activity_taxonomy.md   <- how native labels map onto the shared taxonomy, and why
  HAR_Major_Project_Proposal.docx  <- full proposal document
```

## Setup

```bash
pip install -r requirements.txt
```

## Getting the two missing datasets

**PAMAP2** (~700MB):
1. Download from https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
2. Extract so that `data/raw/PAMAP2/Protocol/subject101.dat` ... `subject109.dat` exist

**MHEALTH** (~40MB):
1. Download from https://archive.ics.uci.edu/dataset/319/mhealth+dataset
2. Extract so that `data/raw/MHEALTH/mHealth_subject1.log` ... `mHealth_subject10.log` exist

Both loaders (`src/loaders/pamap2.py`, `src/loaders/mhealth.py`) are
written directly from each dataset's documented file format and are
ready to run as soon as the files are in place — no code changes needed.

## Running the EDA

```bash
python src/eda/run_eda.py
```

Produces class-distribution plots and a cross-dataset comparison table
in `notebooks/eda_outputs/`. Currently runs on UCI-HAR and WISDM; will
automatically pick up PAMAP2/MHEALTH once downloaded.

## Shared activity taxonomy

Cross-dataset experiments need one common label set. See
`docs/activity_taxonomy.md` for the full reasoning; short version:

- **Core-4** (`WALKING, SITTING, STANDING, STAIRS`) — works across all 4 datasets, used for the main cross-dataset evaluation matrix
- **Extended-5** (+ `LYING`) — works across UCI-HAR/PAMAP2/MHEALTH only (WISDM has no lying/resting class), used as a secondary experiment

Apply it via:

```python
from src.loaders.taxonomy import apply_shared_taxonomy
df_core4 = apply_shared_taxonomy(df, "UCI_HAR", tier="core4")
```

## Team

- **Member A** — data pipeline, preprocessing, baseline models
- **Member B** — cross-dataset evaluation, domain adaptation, explainability

## Next (Phase 2, before End-Sem 1 Viva)

- [ ] Subject-disjoint train/val/test splitting per dataset
- [ ] Fixed-size sliding window segmentation + per-channel normalization
- [ ] 1D-CNN and LSTM baseline architectures
- [ ] Within-dataset training + evaluation on all 4 datasets
