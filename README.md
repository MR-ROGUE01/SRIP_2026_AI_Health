# SRIP 2026 — AI for Health: Sleep Breathing Irregularity Detection

## About

This project is part of the SRIP 2026 Health Sensing assignment. The goal is to detect breathing irregularities like Apnea and Hypopnea during sleep using overnight physiological signals collected from 5 participants.

Sleep breathing disorders are serious medical conditions. During an Apnea event, breathing completely stops. During Hypopnea, breathing becomes shallow. Both reduce oxygen levels in the blood and disrupt sleep quality. Early detection using machine learning can help in diagnosis.

---

## Dataset

The dataset contains ~8 hours of overnight sleep recordings for 5 participants (AP01 to AP05).

Each participant folder contains:

| File | Signal | Sampling Rate |
|------|--------|---------------|
| nasal_airflow.txt | Nasal airflow | 32 Hz |
| thoracic_movement.txt | Thoracic movement | 32 Hz |
| spo2.txt | Oxygen saturation | 4 Hz |
| flow_events.csv | Breathing event annotations | — |
| sleep_profile.csv | Sleep stage annotations | — |

> **Note:** Data is included in this repository. `Dataset/breathing_dataset.csv` is stored via Git LFS due to its size.

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

## Pipeline
```
Raw Signals (nasal airflow, thoracic movement, SpO2)
        ↓
Bandpass Filtering (0.17 Hz – 0.4 Hz)
        ↓
Sliding Window Segmentation (30 sec, 50% overlap)
        ↓
Event Labeling (Normal / Hypopnea / Obstructive Apnea)
        ↓
1D CNN Training with LOPO Cross Validation
        ↓
Evaluation (Accuracy, Precision, Recall, Confusion Matrix)
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Part 1 — Visualize signals
```bash
python scripts/vis.py -name "Data/AP01"
```

Generates a PDF with all 3 signals plotted over the full 8-hour session with breathing events overlaid as colored shading. Output saved to `Visualizations/`.

### Part 2 — Create dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

- Applies a Butterworth bandpass filter (0.17–0.4 Hz) to remove noise
- Splits each signal into 30-second windows with 50% overlap
- Labels each window as Normal, Hypopnea, or Obstructive Apnea based on event overlap
- Saves final dataset to `Dataset/breathing_dataset.csv`

Dataset summary:
- Total windows: ~8800
- Columns per window: 2045 (960 flow + 960 thoracic + 120 SpO2 + metadata)

### Part 3 — Train and evaluate model

Open `notebooks/train_model.ipynb` in Jupyter or VS Code and run all cells.

- Trains a 1D CNN on the labeled windows
- Uses Leave-One-Participant-Out Cross Validation (5 folds)
- Reports Accuracy, Precision, Recall and Confusion Matrix

---

## Model Architecture
```
Input (960 timesteps, 3 channels)
        ↓
Conv1D (32 filters, kernel=5, ReLU)
        ↓
MaxPooling1D (pool=2)
        ↓
Conv1D (64 filters, kernel=5, ReLU)
        ↓
MaxPooling1D (pool=2)
        ↓
Flatten
        ↓
Dense (64, ReLU) + Dropout (0.3)
        ↓
Dense (3, Softmax)
        ↓
Output: Normal / Hypopnea / Obstructive Apnea
```

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 67.2% |
| Precision | 36.0% |
| Recall | 40.7% |

The dataset is heavily imbalanced — 91% of windows are Normal while Apnea events make up only 2%. This explains the lower precision and recall on minority classes. Class weights were used during training to address this.

Confusion matrix is saved at `Visualizations/confusion_matrix.png`.

---

## Label Distribution

| Label | Count |
|-------|-------|
| Normal | 8038 |
| Hypopnea | 593 |
| Obstructive Apnea | 164 |

---

## AI Disclosure

This project was completed with assistance from Claude (Anthropic) for code writing and debugging. All code has been reviewed, understood, and can be explained by the author.

---

## Author

Raj Kumar Gupta