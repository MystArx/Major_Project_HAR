# Cross-Domain Generalization in Wearable Human Activity Recognition

B.Tech ECE Major Project — code and data pipeline.
See `docs/` for the full proposal document and this checkpoint's working
notes.

## Status (Checkpoint 1 — Mid-Sem 1)

| Dataset  | Status | Notes |
|----------|--------|-------|
| **UCI-HAR**  | ✅ Loaded & verified | 10,299 windows, 30 subjects, 6 activities. Pre-extracted 561-feature release (no raw signal). |
| **WISDM**    | ✅ Loaded & verified | 5,418 windows, 36 subjects, 6 activities. Pre-extracted 43-feature (ARFF) release. Real class imbalance present (Standing=246 vs Walking=2081). |
| **PAMAP2** | ✅ Loaded & analyzed | 9 | 12 in current processed view | 100 Hz | Raw multichannel signal |
| **MHEALTH** | ✅ Loaded & analyzed | 10 | 12 | 50 Hz | Raw multichannel signal |

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

