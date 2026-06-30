# 🚦Accident Risk Predictor (CI/CD Pipeline Project)

A machine-learning system that predicts road accident risk in **Karachi, Pakistan** using real-time weather and traffic data, backed by a fully automated GitHub Actions CI/CD pipeline.

---

This project delivers:

### 1. Machine Learning System
An accident risk classifier that ingests three environmental signals:

| `precipitation_mm` | Open-Meteo Archive/Forecast API | Hourly rainfall in Karachi |
| `visibility_km` | Open-Meteo Archive/Forecast API | Visibility clamped to 0.3–15 km |
| `traffic_density_index` | Rule-based (local time) | 0.85 during rush hours, 0.40 otherwise |

The pipeline compares **5 classifiers** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVC), selects the best by F1-Score, serializes it to `accident_model.pkl`, and uses it to output a live **Low / Medium / High Risk** verdict.

### 2. CI/CD Automation via GitHub Actions
A three-job automated pipeline (`ci.yml`) that runs on every push and pull request:

```
Push / PR
    │
    ▼
[Job 1] Lint (flake8)
    │  passes?
    ▼
[Job 2] Unit Tests (pytest)
    │  passes? + push to main only?
    ▼
[Job 3] Daily Risk Prediction
```

Branch protection rules enforce that **no PR can be merged into `main` unless both lint and tests pass**.

---

## Project Structure

```
accident-predictor-cicd/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions pipeline definition
│
├── fetch_data.py                   # Downloads 1-year hourly weather from Open-Meteo Archive
├── generate_labels.py              # Score-based label generation (deterministic)
├── inject_label.py                 # Probabilistic label injection (binomial sampling)
├── train_model.py                  # Multi-model benchmarking + best model export
├── predict_daily_risk.py           # Live inference using current weather + time
├── test_accident_systems.py        # pytest suite (4 test cases)
├── requirements.txt                # Python dependencies
├── historical_accidents.csv        # Training data (8,760 hourly rows, 1 year)
├── daily_risk_log.csv              # Append-only prediction log
└── accident_model.pkl              # Serialized best model binary
```

---

## Technology Stack

### Python
Python is the industry standard for machine learning. Its ecosystem (scikit-learn, pandas, numpy, joblib) covers every stage of the pipeline from data loading to model serialization in a consistent, readable syntax that matches how the system's logic is described.

### scikit-learn
Provides a unified API for 5 different classifier families under one `fit/predict/predict_proba` interface, making multi-model comparison trivial. `RandomForestClassifier` and `GradientBoostingClassifier` are well-suited for small tabular datasets where deep learning would overfit.

### Open-Meteo API
Free, no-auth-required, high-quality meteorological data with both archive (historical) and forecast (real-time) endpoints - ideal for a self-contained project that needs real environmental signals without API key management complexity.

### GitHub Actions
Native to GitHub, zero infrastructure cost, and YAML-based - meaning the pipeline is version-controlled alongside the code it tests. The `needs:` keyword enforces a strict job dependency chain so tests cannot run on unlinted code.

### flake8
Lightweight, fast, and widely adopted Python linter. Catches syntax errors, undefined names, and PEP 8 style violations in under 2 seconds - a valuable first gate before running the heavier test suite.

### pytest
A Python testing framework. Clean fixture model, readable assertion output, and integrates with GitHub Actions via standard exit codes (0 = pass, non-zero = fail).

---

## CI/CD Pipeline Explanation

### Trigger Events
The pipeline fires on:
- `push` to `main` or any `feature/**` branch
- `pull_request` targeting `main`

### Job 1 - Lint (`flake8`)
Installs flake8 and scans all `.py` files for style violations and syntax errors.
**Max line length:** 120 characters (accommodates pandas/sklearn method chains).
If this job fails, Job 2 is skipped entirely (`needs: lint`).

### Job 2 - Unit Tests (`pytest`)
Installs all dependencies from `requirements.txt`, trains the model using `train_model.py` (so `accident_model.pkl` exists for tests), then runs the 4-test suite:

| `test_synthetic_data_generation` | DataFrame shape, column names, feature bounds |
| `test_map_risk_logic` | Risk classification thresholds (High/Medium/Low) |
| `test_model_file_exists_and_loads` | Model binary exists and is a valid joblib classifier |
| `test_model_prediction_schema` | Model accepts correct feature shape, returns valid probabilities |

### Job 3 - Daily Prediction (main-only)
Runs `predict_daily_risk.py` to fetch live weather from Open-Meteo and log the current risk level. This only triggers on `push` to `main` (not on PRs) to avoid unnecessary API calls.

### Branch Protection
Configured via GitHub Settings ----> Branches ----> Protection Rules:
- PRs require `Lint Check (flake8)` ✅ and `Unit Tests (pytest)` ✅ to pass
- Direct pushes to `main` are blocked
- See `.github/BRANCH_PROTECTION.md` for setup steps

---

##  AI Tools Used During Development

This project was developed with **Github Copilot** and **gemini** as a primary development assistant. how i used it:


| `fetch_data.py` | copilot identified and fixed a timezone bug - raw UTC timestamps were being used to assign rush-hour traffic density, which shifted all hour classifications by 5 hours. The fix wraps UTC datetimes in `ZoneInfo("UTC")` before converting to `Asia/Karachi`. |
| `generate_labels.py` | copilot suggested lowering the accident threshold from 0.5 ---> 0.3 and adding a safety check that forces the top 15% of risk-scored hours to be labeled `1` when the dataset produces too few positive samples. |
| `train_model.py` | copilot expanded a single-model script into a 5-model benchmarking loop with a `results_df` comparison table, keeping the best-F1 model selection logic clean. |
| `test_accident_systems.py` | All 4 test cases were written with copilot, covering data generation bounds, classification logic, model file existence, and prediction schema. |
| `ci.yml` | copilot authored the full workflow, including the `needs:` chain, pip caching with `hashFiles`, and the `if: github.event_name == 'push'` condition for Job 3. |
| README | Drafted with copilot, then reviewed and edited for accuracy against the actual codebase. |

**Human judgment was applied to:** data source selection (Open-Meteo vs alternatives), feature engineering decisions (rush hour windows, visibility clipping bounds), threshold choices, and all architectural decisions about file structure. Also the model selection code provided by me for selecting best model that performs well among various models.

---

## How to Run Locally

```bash
# 1. Clone and enter the repository
git clone [https://github.com/Alishba14/accident-predictor-cicd.git](https://github.com/Al-Rafay-Consulting/Accident-Predictor.git)
cd accident-predictor-cicd

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch fresh historical training data from Open-Meteo
python fetch_data.py

# 4. Generate realistic accident labels
python inject_label.py

# 5. Train models and export the best one
python train_model.py

# 6. Run live risk prediction
python predict_daily_risk.py

# 7. Run the test suite
pytest test_accident_systems.py -v
```

---

## Improvements I can Make With More Time

### Data Quality
- **Replace synthetic labels with real accident reports** integration of Pakistan's National Highway Authority or Karachi traffic police open datasets to replace the probabilistically-generated `accident_occurred` column with verified incidents.
- **Add more features** - road type, time of day (categorical), day of week, fog index, wind speed, and historical accident density per road segment would all improve model signal.

### Model Improvements
- **Hyperparameter tuning** ---> i can add `GridSearchCV` or `Optuna` optimization in `train_model.py` instead of using default parameters for all classifiers.
- **Class imbalance handling** - implement SMOTE (Synthetic Minority Oversampling) since real accident data will be heavily imbalanced (far more 0s than 1s).
- **Model versioning** - i can use MLflow to track model versions alongside their training metrics instead of overwriting a single `accident_model.pkl`.

### Pipeline Enhancements
- **Scheduled runs** - add a `schedule: cron: '0 6 * * *'` trigger to run the daily prediction every morning automatically.
- **Artifact upload** - use `actions/upload-artifact` to persist `daily_risk_log.csv` across workflow runs.
- **Coverage reports** - integrate `pytest-cov` and upload HTML coverage reports to GitHub Pages.
- **Slack/email notifications** ----> notify the team when the pipeline fails or when a High Risk prediction is logged.
- **Matrix testing** ----> test against Python 3.10, 3.11, and 3.12 using `strategy: matrix`.

### Production Readiness
- **Containerize with Docker** ----> add a `Dockerfile` so the prediction script runs identically in all environments.
- **Deploy as an API** ----> wrap `predict_daily_risk.py` in a FastAPI endpoint and deploy to Azure Container Apps (fitting the ARC Microsoft Solutions Partner context).
- **Database logging** ----> replace the flat CSV log with a PostgreSQL table for queryable, persistent prediction history.

---

## 📋 Git Workflow Followed

Each feature branch was submitted as a **Pull Request** with a description of changes. No direct commits to `main`.

---

## Dependencies

```
numpy
pandas
scikit-learn
joblib
```

---
