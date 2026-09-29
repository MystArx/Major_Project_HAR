# Cross-Domain Generalization in Wearable Human Activity Recognition

**B.Tech ECE Major Project**

This project investigates **cross-domain generalization in wearable Human Activity Recognition (HAR)**.

Human Activity Recognition models can achieve high accuracy when training and testing data come from the same dataset. However, their performance may degrade significantly when deployed in a different sensing environment due to differences in:

- Sensor placement
- Device hardware
- Sampling frequency
- Number and type of sensors
- Participant population
- Activity definitions
- Data collection protocols

The objective of this project is to systematically study this **cross-dataset generalization problem** using multiple public wearable HAR datasets.

The project will evaluate standard deep-learning HAR models, measure their performance across different sensing domains, investigate **unsupervised domain adaptation using Deep CORAL**, and analyze model behaviour using **SHAP-based explainability**.

---

## Current Status

### Checkpoint 1 — Dataset Acquisition, Harmonization and Exploratory Data Analysis

The initial data preparation and exploratory analysis stage has been completed for all four benchmark datasets selected for the project.

All four datasets have:

- Been acquired and organized within the project
- Been processed using dataset-specific loaders
- Undergone initial validation
- Had their activity distributions analyzed
- Been incorporated into a common cross-dataset summary
- Been mapped toward a shared activity taxonomy for future cross-dataset experiments

| Dataset | Status | Subjects | Activities | Sampling Rate | Data Representation |
|---|---|---:|---:|---:|---|
| **UCI-HAR** | ✅ Loaded & analyzed | 30 | 6 | 50 Hz | Pre-extracted features (561) |
| **WISDM** | ✅ Loaded & analyzed | 36 | 6 | ~20 Hz nominal | Pre-extracted features (43) |
| **PAMAP2** | ✅ Loaded & analyzed | 9 | 12 in current processed view | 100 Hz | Raw multichannel signal |
| **MHEALTH** | ✅ Loaded & analyzed | 10 | 12 | 50 Hz | Raw multichannel signal |

---

## Work Completed

### 1. Dataset Acquisition

The four HAR datasets selected in the project proposal have been obtained and incorporated into the project pipeline:

- **UCI Human Activity Recognition (UCI-HAR)**
- **WISDM**
- **PAMAP2 Physical Activity Monitoring**
- **MHEALTH**

These datasets intentionally represent different sensing environments so that the effect of **domain shift** can later be studied.

---

### 2. Dataset Loaders

Separate Python loaders have been implemented for each dataset:

```text
src/loaders/
├── uci_har.py
├── wisdm.py
├── pamap2.py
├── mhealth.py
└── taxonomy.py
```

The loaders convert the different source formats into representations that can be inspected and processed by the common project pipeline.

This is necessary because the datasets differ substantially in their original formats, sensor configurations, sampling rates, and activity labels.

---

### 3. Exploratory Data Analysis

Initial exploratory analysis has been performed for all four datasets.

The EDA currently includes:

- Dataset loading and validation
- Number of subjects
- Number of activities
- Activity/class distributions
- Comparison of dataset characteristics
- Generation of dataset summary information
- Generation of class-distribution plots

The generated outputs are stored in:

```text
notebooks/eda_outputs/
```

Current outputs include:

```text
dataset_summary.csv

uci-har_class_dist.png
wisdm_class_dist.png
pamap2_class_dist.png
mhealth_class_dist.png
```

The dataset summary provides a compact comparison of the sensing characteristics of the four datasets.

---

## Dataset Characteristics

The selected datasets intentionally differ in several important ways.

### UCI-HAR

- Smartphone-based HAR dataset
- Smartphone positioned at the waist
- 30 subjects
- 6 activity classes
- 50 Hz sampling frequency
- Current project version uses the **561-dimensional pre-extracted feature representation**

### WISDM

- Smartphone-based activity dataset
- Smartphone carried by participants
- 36 subjects
- 6 activity classes
- Approximately 20 Hz nominal sampling rate
- Current project version uses a **43-feature pre-extracted representation**

### PAMAP2

- Multi-sensor wearable activity dataset
- Sensors placed at multiple body locations
- IMUs positioned at the hand, chest, and ankle
- Heart-rate information is also available
- 9 subjects
- 100 Hz sampling frequency
- Raw multichannel sensor data
- 12 activities are present in the current processed project view

### MHEALTH

- Multi-sensor wearable activity dataset
- Sensors positioned at multiple body locations
- Includes chest, ankle, and arm measurements
- ECG information is also available
- 10 subjects
- 12 activity classes
- 50 Hz sampling frequency
- Raw multichannel sensor data

---

## Shared Activity Taxonomy

A major problem in cross-dataset HAR is that different datasets do not use identical activity labels.

For example, one dataset may distinguish between walking upstairs and walking downstairs while another may provide a more general stair-related activity representation.

To support cross-dataset experiments, a shared activity taxonomy has therefore been defined.

The mapping logic is implemented in:

```text
src/loaders/taxonomy.py
```

Additional documentation is available in:

```text
docs/activity_taxonomy.md
```

### Core-4 Taxonomy

The primary cross-dataset experiments will use four activities that can be represented across all four datasets:

```text
WALKING
SITTING
STANDING
STAIRS
```

This will be used for the main cross-dataset generalization experiments.

### Extended-5 Taxonomy

A secondary taxonomy additionally includes:

```text
LYING
```

giving:

```text
WALKING
SITTING
STANDING
STAIRS
LYING
```

The Extended-5 taxonomy can be used with:

- UCI-HAR
- PAMAP2
- MHEALTH

WISDM does not provide a directly compatible lying/resting activity class, so it is excluded from this secondary experiment.

The shared taxonomy can be applied through:

```python
from src.loaders.taxonomy import apply_shared_taxonomy

df_core4 = apply_shared_taxonomy(
    df,
    "UCI_HAR",
    tier="core4"
)
```

---

## Important Data Representation Difference

One of the main technical challenges identified during the first stage of the project is that the four datasets do **not** currently have identical input representations.

### Feature-based datasets

UCI-HAR and WISDM currently use pre-extracted feature representations:

```text
UCI-HAR → 561 features
WISDM   → 43 features
```

### Raw-signal datasets

PAMAP2 and MHEALTH contain raw multichannel time-series sensor measurements.

```text
PAMAP2  → Raw IMU + heart-rate signals
MHEALTH → Raw wearable sensor + ECG signals
```

Therefore, a common representation must be established before direct cross-dataset CNN/LSTM experiments can be performed.

Handling this representation mismatch is part of the upcoming preprocessing and experimental stage rather than assuming that all four datasets are directly interchangeable.

---

## Project Structure

```text
Major_Project_HAR/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── loaders/
│   │   ├── uci_har.py
│   │   ├── wisdm.py
│   │   ├── pamap2.py
│   │   ├── mhealth.py
│   │   └── taxonomy.py
│   │
│   └── eda/
│       └── run_eda.py
│
├── notebooks/
│   └── eda_outputs/
│       ├── dataset_summary.csv
│       ├── uci-har_class_dist.png
│       ├── wisdm_class_dist.png
│       ├── pamap2_class_dist.png
│       └── mhealth_class_dist.png
│
├── docs/
│   └── activity_taxonomy.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the EDA

Run:

```bash
python src/eda/run_eda.py
```

The script analyzes the available datasets and generates the corresponding outputs inside:

```text
notebooks/eda_outputs/
```

---

## Planned Research Pipeline

The overall experimental pipeline of the project is:

```text
HAR Datasets
     │
     ▼
Dataset-Specific Loading
     │
     ▼
Data Validation & EDA
     │
     ▼
Activity Taxonomy Harmonization
     │
     ▼
Common Preprocessing / Representation
     │
     ▼
Subject-Disjoint Data Splitting
     │
     ▼
Baseline HAR Models
 ┌──────────┴──────────┐
 ▼                     ▼
1D-CNN                LSTM
 │                     │
 └──────────┬──────────┘
            ▼
Within-Dataset Evaluation
            │
            ▼
Cross-Dataset Evaluation
            │
            ▼
Generalization Gap Analysis
            │
            ▼
Deep CORAL
Domain Adaptation
            │
            ▼
Cross-Dataset Re-evaluation
            │
            ▼
SHAP / Error Analysis
```

---

## Planned Cross-Dataset Evaluation

The eventual objective is to construct a cross-dataset evaluation matrix.

Conceptually:

| Train ↓ / Test → | UCI-HAR | WISDM | PAMAP2 | MHEALTH |
|---|---:|---:|---:|---:|
| **UCI-HAR** | Within-domain | Cross-domain | Cross-domain | Cross-domain |
| **WISDM** | Cross-domain | Within-domain | Cross-domain | Cross-domain |
| **PAMAP2** | Cross-domain | Cross-domain | Within-domain | Cross-domain |
| **MHEALTH** | Cross-domain | Cross-domain | Cross-domain | Within-domain |

The diagonal results will represent **within-dataset performance**.

The off-diagonal results will measure **cross-dataset generalization**.

This will allow the project to quantify how much HAR performance changes when a trained model encounters a different sensing domain.

---
