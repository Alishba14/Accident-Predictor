# Accident Risk Predictor

A machine-learning system that predicts road accident risk in **Karachi, Pakistan** using real-time weather and traffic data, backed by a fully automated GitHub Actions CI/CD pipeline.

---

## What It Does

The system ingests three environmental signals and classifies the current hour as **Low / Medium / High Risk**:

| Feature | Source | Description |
|---------|--------|-------------|
| `precipitation_mm` | Open-Meteo Archive / Forecast API | Hourly rainfall |
| `visibility_km` | Open-Meteo Archive / Forecast API | Visibility clamped to 0.3–15 km |
| `traffic_density_index` | Rule-based (local Karachi time) | 0.85 during rush hours, 0.40 otherwise |

Five classifiers are benchmarked (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVC). The best model by **5-fold cross-validated F1** is serialized to `accident_model.pkl` and used for live inference.

---

## Project Structure

```
accident-predictor-cicd/
│
├── .github/
│   └── workflows/
│       └── daily_prediction.yml    # GitHub Actions: test → predict → log
│
├── fetch_data.py                   # Downloads 1-year hourly weather from Open-Meteo Archive
├── generate_labels.py              # Assigns synthetic accident labels from a risk scoring formula
├── train_model.py                  # 5-model benchmarking + best model export
├── predict_daily_risk.py           # Live inference using current weather + time
├── run_pipeline.py                 # Orchestrator: runs all steps in the correct order
├── test_accident_systems.py        # pytest suite (12 test cases)
├── requirements.txt                # Pinned Python dependencies
├── .gitignore                      # Excludes caches, backups, and .venv
├── historical_accidents.csv        # Training data (8,760 hourly rows, 1 year)
├── daily_risk_log.csv              # Append-only prediction log
├── accident_model.pkl              # Serialized best model binary
└── accident_model.pkl.sha256       # SHA-256 integrity hash of the model
```

---

## How to Run Locally

```bash
# 1. Clone and enter the repository
git clone https://github.com/Alishba14/accident-predictor-cicd.git
cd accident-predictor-cicd

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (fetch → label → train → predict)
python run_pipeline.py

# 5. Run tests
pytest test_accident_systems.py -v

# 6. Prediction only (skips training, uses existing model)
python run_pipeline.py --predict
```

**Manual step-by-step** (if you prefer):
```bash
python fetch_data.py          # Download weather data
python generate_labels.py     # Assign accident labels
python train_model.py         # Train and export best model
python predict_daily_risk.py  # Run live prediction
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/daily_prediction.yml`) runs automatically:

- **On schedule** — every day at 06:00 UTC (11:00 AM Karachi time)
- **On `workflow_dispatch`** — manually triggered from GitHub UI

```
Scheduled / Manual trigger
          │
          ▼
  [Install dependencies from requirements.txt]
          │
          ▼
  [Run pytest suite — 12 tests]
          │  all pass?
          ▼
  [python predict_daily_risk.py]
          │
          ▼
  [git commit & push updated daily_risk_log.csv]
```

---

## Test Suite (12 tests)

| Test | What it covers |
|------|----------------|
| `test_synthetic_data_generation` | DataFrame shape, column names, feature bounds |
| `test_map_risk_logic` | High / Medium / Low classification |
| `test_map_risk_exact_boundaries` | Exact boundary values (0.70, 0.40) |
| `test_traffic_density_rush_hour` | 8 AM Karachi → 0.85 (UTC-corrected) |
| `test_traffic_density_off_peak` | 12 PM Karachi → 0.40 |
| `test_traffic_density_evening_rush` | 5 PM Karachi → 0.85 |
| `test_load_real_data_missing_required_column` | Raises on missing column |
| `test_load_real_data_all_zero_labels_raises` | Raises `ValueError` on all-zero labels |
| `test_load_real_data_clips_negative_precipitation` | Negative precip clipped to 0 |
| `test_load_real_data_removes_duplicates` | Duplicate rows dropped silently |
| `test_model_file_exists_and_loads` | Model binary is a valid joblib classifier *(skipped on fresh clone)* |
| `test_model_prediction_schema` | Feature shape, output bounds, valid probability *(skipped on fresh clone)* |

---

## Security & Quality Measures

| Area | Measure |
|------|---------|
| Model integrity | SHA-256 hash written on save; verified before every `joblib.load` |
| API failure | Fail-closed — raises `RuntimeError` instead of silently returning Low Risk |
| Model output | `predict_proba` result bounds-checked to `[0.0, 1.0]` |
| Log file | Atomic creation with `touch(exist_ok=False)` to fix TOCTOU race condition |
| Input validation | Date format and calendar validity checked before URL construction |
| Data validation | Negative precip clipped; duplicate rows dropped; all-zero labels raise error |
| Feature scaling | SVC and Logistic Regression wrapped in `StandardScaler` Pipeline |
| Model selection | 5-fold stratified CV (not a single split) to avoid lucky/unlucky splits |
| Null guard | Raises if all models score CV F1 = 0 instead of saving a null model |
| Model versioning | Timestamped `.pkl` backup saved on every training run |
| Data backup | `historical_accidents.csv.bak` written before any label overwrite |
| Dependencies | Pinned to exact major versions in `requirements.txt` |

---

## Technology Stack

| Tool | Role |
|------|------|
| **Python 3.10+** | Core language |
| **scikit-learn** | 5-classifier benchmarking, Pipelines, StandardScaler, StratifiedKFold |
| **pandas / numpy** | Data loading, feature engineering, label generation |
| **joblib** | Model serialization |
| **Open-Meteo API** | Free, no-auth weather data (archive + forecast endpoints) |
| **GitHub Actions** | CI/CD scheduling, test gate, automated log commits |
| **pytest** | 12-test suite with `skipif` markers for model-dependent tests |

---

## Known Limitations

- **Labels are synthetic** — `generate_labels.py` computes `accident_occurred` from a risk-scoring formula applied to the same features used for training. Model metrics reflect how well it learned the formula, not real-world accident prediction accuracy.
- **Only 3 features** — adding road type, day-of-week, wind speed, and historical hotspot density would meaningfully improve signal.
- **Binary traffic density** — the `traffic_density_index` is effectively a rush-hour flag (0.85 / 0.40). A real traffic feed would improve this.

---

## Potential Improvements

- **Real accident data** — integrate Pakistan's NHA or Karachi traffic police open datasets to replace synthetic labels.
- **Hyperparameter tuning** — add `GridSearchCV` or `Optuna` instead of default classifier parameters.
- **SMOTE** — handle class imbalance in real accident data with oversampling.
- **MLflow tracking** — log training runs and compare model versions over time.
- **Docker + FastAPI** — containerize and expose `predict_daily_risk.py` as a REST endpoint.
- **Matrix testing** — test against Python 3.10, 3.11, and 3.12 in CI.
- **Coverage reports** — integrate `pytest-cov` and publish HTML coverage to GitHub Pages.

---

## AI Tools Used

This project was developed with **GitHub Copilot** as a primary development assistant:

---

## Dependencies

```
numpy>=2.4,<3
pandas>=3.0,<4
scikit-learn>=1.9,<2
joblib>=1.5,<2
pytest>=8.0,<9
```
