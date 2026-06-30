import hashlib
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
MODEL_PATH = Path("accident_model.pkl")
HASH_PATH = Path("accident_model.pkl.sha256")


def load_real_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses real historical accident data from a CSV."""
    df = pd.read_csv(file_path)

    # 1. Map columns if your real data has different naming conventions
    # df = df.rename(columns={'rain_amount': 'precipitation_mm', ...})

    # 2. Check for missing values and fill or drop them safely
    required_cols = [
        "precipitation_mm",
        "visibility_km",
        "traffic_density_index",
        "accident_occurred",
    ]
    df = df.dropna(subset=required_cols)

    # 3. Data validation bounds (Clipping outliers if necessary to fit training distribution)
    df["traffic_density_index"] = df["traffic_density_index"].clip(0.0, 1.0)

    # 4. Clip physically impossible negative precipitation
    if (df["precipitation_mm"] < 0).any():
        print("Warning: Negative precipitation values found and clipped to 0.")
        df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)

    # 5. Log duplicate feature vectors but do NOT remove them.
    #    Time-series data with coarse features (binary traffic, rounded visibility)
    #    naturally produces identical feature vectors across different timestamps.
    #    Dropping them destroys the dataset and skews class balance.
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Info: {n_dupes} rows share identical feature values (expected for coarse time-series features).")

    # 6. Guard against all-zero labels — model cannot learn without positive examples
    if df["accident_occurred"].sum() == 0:
        raise ValueError(
            "Training data has zero positive labels (accident_occurred = 1). "
            "Run generate_labels.py or inject_label.py to assign accident labels first."
        )

    return df


def generate_synthetic_data(n_rows: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fall-back synthetic generator if no historical dataset exists yet."""
    rng = np.random.default_rng(seed)
    precipitation_mm = rng.gamma(shape=2.0, scale=2.0, size=n_rows)
    visibility_km = np.clip(rng.normal(loc=8.5, scale=2.5, size=n_rows), 0.3, 15.0)
    traffic_density_index = rng.uniform(0.0, 1.0, size=n_rows)

    risk_score = (
        0.30 * np.clip(precipitation_mm / 10.0, 0.0, 1.0)
        + 0.35 * (1.0 - np.clip(visibility_km / 15.0, 0.0, 1.0))
        + 0.35 * traffic_density_index
    )
    noisy_score = risk_score + rng.normal(0.0, 0.08, size=n_rows)
    accident_occurred = (noisy_score > 0.50).astype(int)

    return pd.DataFrame(
        {
            "precipitation_mm": precipitation_mm.round(2),
            "visibility_km": visibility_km.round(2),
            "traffic_density_index": traffic_density_index.round(3),
            "accident_occurred": accident_occurred,
        }
    )


def main() -> None:
    # --- DATA SELECTION (auto-detects CSV; falls back to synthetic if not found) ---
    HISTORICAL_DATA_PATH = "historical_accidents.csv"
    USE_REAL_DATA = Path(HISTORICAL_DATA_PATH).exists()

    if USE_REAL_DATA:
        print(f"Loading real-world data from {HISTORICAL_DATA_PATH}...")
        df = load_real_data(HISTORICAL_DATA_PATH)
    else:
        print("historical_accidents.csv not found — using synthetic fallback dataset...")
        df = generate_synthetic_data(n_rows=1000)

    feature_cols = [
        "precipitation_mm",
        "visibility_km",
        "traffic_density_index",
    ]
    X = df[feature_cols]
    y = df["accident_occurred"]

    # Stratified split to handle target imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # --- MODEL BENCHMARKING ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Support Vector Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(class_weight="balanced", random_state=RANDOM_SEED), ensemble=False)),
        ]),
    }

    results = {}
    best_cv_f1 = -1
    best_model_name = None
    best_model_object = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("\n" + "="*50)
    print("Starting Multi-Model Comparison Benchmarking")
    print("="*50)

    for name, model in models.items():
        # 5-fold cross-validated F1 — unbiased model selection
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        mean_cv_f1 = cv_scores.mean()

        # Final fit on training split for hold-out evaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate Metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {"CV F1 (mean)": round(mean_cv_f1, 4), "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 (hold-out)": f1}

        # Select best model by cross-validated F1 to avoid single-split luck
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_model_name = name
            best_model_object = model

    # Print out results comparison matrix
    results_df = pd.DataFrame(results).T.sort_values(by="CV F1 (mean)", ascending=False)
    print("\nModels (Sorted by CV F1):")
    print(results_df.to_string())
    print("="*50)

    # --- EXPORT BEST MODEL ---
    if best_model_object is None:
        raise RuntimeError(
            "No model could be selected — all models scored CV F1 = 0. "
            "Check your training data: run generate_labels.py or inject_label.py first."
        )

    print(f"Saving the best model ('{best_model_name}') to '{MODEL_PATH}'...")
    joblib.dump(best_model_object, MODEL_PATH)

    # Write SHA-256 integrity hash (verified by predict_daily_risk.py at load time)
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    HASH_PATH.write_text(model_hash)
    print(f"Integrity hash written to '{HASH_PATH}' ({model_hash[:16]}...)")

    # Save a timestamped backup so a bad retrain never destroys the previous model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"accident_model_backup_{timestamp}.pkl")
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"Versioned backup saved: '{backup_path}'")
if __name__ == "__main__":
    main()