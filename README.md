# XAI Medical Diagnosis System

> Multi-disease prediction with explainable AI — transparent, trustworthy, clinically oriented.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-purple?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

This project predicts disease risk across **four clinical domains** — heart disease, diabetes, lung cancer, and breast cancer — using Logistic Regression and Neural Networks, then uses **SHAP** to explain every prediction in human-readable terms.

---

## What it does

- Trains disease-specific models on large public health datasets (up to ~319K records)
- Compares an interpretable baseline (Logistic Regression) against a neural network per disease
- Generates **SHAP summary plots and local explanations** for each prediction
- Outputs a CLI report: predicted risk + top contributing features + direction of influence
- Persists fitted scalers and column schemas so inference always matches the training distribution

---

## Results at a glance

| Dataset | Records | LR ROC-AUC | LR Recall+ | NN ROC-AUC | NN Recall+ |
|---|---|---|---|---|---|
| Heart Disease | ~319K | **0.837** | 0.86 | 0.802 | 0.77 |
| Diabetes | ~250K | **0.864** | 0.86 | 0.825 | 0.83 |
| Lung Cancer | ~20K | ⚠️ 0.485 | 1.00* | ⚠️ 0.496 | 0.86 |
| Breast Cancer | 569 | **0.997** | 0.98 | 0.998 | 0.98 |

> **Recall+** = recall on the positive (disease) class.
> Models are optimised for **recall over accuracy** — missing a true positive is far costlier than a false positive in screening.
> *Lung cancer LR degenerated to majority-class prediction. High recall is an artefact of class imbalance, not genuine discriminative ability (ROC-AUC ≈ 0.5 confirms this).

---

## Project structure

```
XAI_Medical_System/
├── src/
│   ├── data_loader.py       # CSV ingestion → DataFrames
│   ├── preprocessing.py     # encode · scale · align · split
│   ├── models.py            # LR + MLP training, class weighting
│   └── evaluation.py        # classification report, ROC-AUC
├── plots/
│   ├── heart/               # SHAP summary + feature importance plots
│   ├── diabetes/
│   ├── lung/
│   └── cancer/
├── models/                  # saved artefacts (git-ignored)
│   ├── heart_logistic.pkl
│   ├── heart_nn.keras
│   ├── heart_columns.pkl    # ← critical: post-encoding feature schema
│   └── heart_scaler.pkl
├── pipeline.py              # end-to-end orchestration
├── main.py                  # entry point — runs all 4 pipelines
├── demo.py                  # CLI inference + explanation
├── requirements.txt
└── README.md
```

---

## Saved artefacts explained

Four files are saved per disease. **All four are required** for consistent inference.

| File | What it stores | Why it matters |
|---|---|---|
| `*_logistic.pkl` | Trained Logistic Regression | Stable, interpretable model used in demo |
| `*_nn.keras` | Trained MLP (TensorFlow/Keras) | Primary high-capacity model for benchmarking |
| `*_columns.pkl` | Post-encoding column list | Prevents dimensionality mismatch — training expands to 50 columns via one-hot encoding, but demo input has only ~17 raw features. Missing columns are zero-padded at inference. |
| `*_scaler.pkl` | Fitted StandardScaler | Applies identical normalisation to demo input as was applied during training |

---

## Requirements

```
tensorflow>=2.12
scikit-learn>=1.3
shap>=0.44
pandas>=2.0
numpy>=1.24
joblib>=1.3
matplotlib>=3.7
```

---

## Setup

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/rawcdr/XAI_Medical_System.git
cd XAI_Medical_System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add datasets

Place the following CSV files inside a `data/` directory at the project root. They are not included in this repo due to size constraints.

| File | Source | Records |
|---|---|---|
| `heart_disease.csv` | [CDC BRFSS — Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) | ~319K |
| `diabetes.csv` | [CDC BRFSS Diabetes — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | ~250K |
| `lung_cancer.csv` | [Lung Cancer Survey — Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) | ~20K |
| `breast_cancer.csv` | [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 |

---

## Running the system

### Train all four models

```bash
python main.py
```

Runs the full pipeline for every disease: load → preprocess → train LR + NN → evaluate → generate SHAP plots → save artefacts.
Expect **5–10 minutes on CPU**. Output models are saved to `models/` and SHAP plots to `plots/`.

### Run the interactive demo

```bash
python demo.py
```

You will be prompted to select a disease and enter patient feature values. The system outputs:

- Predicted risk (Low / High)
- Model confidence score
- Top contributing features and their direction (increases / decreases risk)
- SHAP-based explanation in plain language

### Run a single disease pipeline

```bash
python pipeline.py --disease heart
```

Valid options: `heart` | `diabetes` | `lung` | `cancer`

---

## How explainability works

**Step 1 — Compute SHAP values**
SHAP values are calculated on the test set using `shap.LinearExplainer` for Logistic Regression and `shap.DeepExplainer` for the Neural Network.

**Step 2 — Global summary plots**
Features are ranked by mean absolute SHAP value, showing which inputs matter most across the entire patient population. Saved to `plots/<disease>/`.

**Step 3 — Local explanations**
For a single patient, SHAP values identify which features pushed the prediction above or below the baseline risk probability.

**Step 4 — Human-readable output**
The top-N SHAP drivers are formatted into plain language, for example:
> *"Elevated BMI and active smoking status were the primary factors increasing this patient's predicted heart disease risk."*

---

## Design decisions

| Decision | Rationale |
|---|---|
| Optimised for recall, not accuracy | Missing a true positive (undetected disease) is far costlier than a false positive in a screening context. Class weights are set inversely proportional to class frequency. |
| LR used as demo fallback, not NN | Neural network validation accuracy was unstable across epochs on smaller datasets. LR provides consistent, calibrated outputs for real-time inference. |
| `*_columns.pkl` is mandatory at inference | One-hot encoding expands raw input from ~17 columns to up to 50. Without the saved schema, inference crashes with a dimensionality mismatch error. |
| Separate scaler per dataset | Each disease dataset has a different feature distribution. A shared scaler would introduce subtle data leakage between domains. |

---

## Known limitations

- **Lung cancer dataset**: binary symptom features provide near-random discriminative signal (ROC-AUC ≈ 0.49). A richer clinical dataset with imaging or biomarker data is needed before this model is meaningful.
- **No external validation**: all evaluation is performed on held-out splits from the same data source. Do not use in any production or clinical setting without prospective validation.
- **Breast cancer dataset**: only 569 records. Neural network results may not generalise to unseen populations.

---

## Future work

- Replace MLP with gradient-boosted trees (XGBoost / LightGBM) — typically stronger baselines for structured tabular data
- Add a **Streamlit web interface** for clinician-friendly input and SHAP visualisation
- Integrate real-time data sources (lab result APIs, wearable sensor streams)
- Implement **federated learning** for privacy-preserving multi-site model training
- Expand coverage to additional diseases (chronic kidney disease, stroke, hypertension)
- Prospective validation against independent clinical datasets

---

*Built as a research project. Not validated for clinical use. All datasets are sourced from public repositories — see links in the Setup section above.*
