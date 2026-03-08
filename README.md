# SRIP 2026 — AI for Health: Sleep Breathing Irregularity Detection

**Author:** Raj Kumar Gupta

---

## Overview

This project detects abnormal breathing patterns during sleep using overnight physiological recordings from 5 participants. The signals are processed, segmented into windows, labeled using breathing event annotations, and classified using a 1D CNN.

---

## Dataset

Each participant folder (AP01–AP05) contains:

| File | Description | Sampling Rate |
|------|-------------|---------------|
| nasal_airflow.txt | Nasal airflow signal | 32 Hz |
| thoracic_movement.txt | Thoracic movement signal | 32 Hz |
| spo2.txt | Oxygen saturation | 4 Hz |
| flow_events.csv | Breathing event annotations | — |
| sleep_profile.csv | Sleep stage annotations | — |

> The `Data/` folder is included in this repo. `Dataset/breathing_dataset.csv` is tracked via Git LFS.

---

## How to Run

**Step 1 — Visualize signals**
```bash
python scripts/vis.py -name "Data/AP01"
```
Plots Nasal Airflow, Thoracic Movement and SpO2 for a participant with breathing events overlaid. Saves PDF to `Visualizations/`.

**Step 2 — Create dataset**
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Filters signals, splits into 30-second windows with 50% overlap, labels each window and saves to `Dataset/breathing_dataset.csv`.

**Step 3 — Train model**

Open `notebooks/train_model.ipynb` and run all cells.
Trains a 1D CNN with Leave-One-Participant-Out Cross Validation.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 67.2% |
| Precision | 36.0% |
| Recall | 40.7% |

Low precision and recall is expected — 91% of windows are Normal with very few Apnea/Hypopnea examples.

Confusion matrix: `Visualizations/confusion_matrix.png`

---

## Setup
```bash
pip install -r requirements.txt
```

---

## AI Disclosure

This project was completed with assistance from Claude (Anthropic) for code writing and debugging. All code has been reviewed and understood by the author.