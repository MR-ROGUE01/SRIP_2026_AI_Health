# SRIP 2026 — AI for Health: Sleep Breathing Irregularity Detection

**Author:** Raj Kumar Gupta  
**Task:** Health Sensing — Detecting breathing irregularities during sleep using overnight PSG data

---

## Problem Statement

Detecting abnormal breathing patterns (Apnea, Hypopnea) during sleep using physiological signals collected from 5 participants over 8-hour overnight sessions.

## Dataset

The dataset contains overnight sleep recordings for 5 participants (AP01–AP05). Each participant folder contains:

- `nasal_airflow.txt` — Nasal airflow signal at 32 Hz
- `thoracic_movement.txt` — Thoracic movement signal at 32 Hz
- `spo2.txt` — Oxygen saturation signal at 4 Hz
- `flow_events.csv` — Annotated breathing events (Apnea, Hypopnea)
- `sleep_profile.csv` — Sleep stage annotations

> **Note:** The `Data/` folder is included in this repository. The generated dataset `Dataset/breathing_dataset.csv` is tracked via Git LFS.

---

## Project Structure
```
SRIP_2026_AI_Health/
├── Data/
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Dataset/
│   └── breathing_dataset.csv
├── Visualizations/
│   ├── AP01_visualization.pdf
│   ├── AP02_visualization.pdf
│   ├── AP03_visualization.pdf
│   ├── AP04_visualization.pdf
│   ├── AP05_visualization.pdf
│   └── confusion_matrix.png
├── models/
│   └── cnn_model.py
├── notebooks/
│   └── train_model.ipynb
├── scripts/
│   ├── vis.py
│   └── create_dataset.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Part 1 — Visualize signals
```bash
python scripts/vis.py -name "Data/AP01"
```
Generates a PDF with Nasal Airflow, Thoracic Movement and SpO2 plots with breathing events overlaid. Output saved to `Visualizations/`.

### Part 2 — Create dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Applies bandpass filtering, splits signals into 30-second windows with 50% overlap, labels each window, and saves to `Dataset/breathing_dataset.csv`.

### Part 3 — Train model
Open `notebooks/train_model.ipynb` in Jupyter or VS Code and run all cells.  
Trains a 1D CNN using Leave-One-Participant-Out Cross Validation across all 5 participants.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 67.2% |
| Precision | 36.0% |
| Recall | 40.7% |

Confusion matrix saved to `Visualizations/confusion_matrix.png`.

The relatively low precision and recall is expected due to heavy class imbalance — 91% of windows are Normal, with very few Apnea examples.

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## AI Disclosure

This project was completed with assistance from Claude (Anthropic) for code writing and debugging. All code has been reviewed, understood, and can be explained by the author.