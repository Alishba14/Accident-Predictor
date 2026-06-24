import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
MODEL_PATH = "accident_model.pkl"


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
    # --- DATA SELECTION ---
    USE_REAL_DATA = True  # Set to True since you created your CSV
    HISTORICAL_DATA_PATH = "historical_accidents.csv"

    if USE_REAL_DATA:
        print(f"Loading real-world data from {HISTORICAL_DATA_PATH}...")
        df = load_real_data(HISTORICAL_DATA_PATH)
    else:
        print("Using synthetic fallback dataset...")
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
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Support Vector Classifier": SVC(class_weight="balanced", probability=True, random_state=RANDOM_SEED)
    }

    results = {}
    best_f1 = -1
    best_model_name = None
    best_model_object = None

    print("\n" + "="*50)
    print("Starting Multi-Model Comparison Benchmarking")
    print("="*50)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate Metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
        
        # Track the best performing model based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_object = model

    # Print out results comparison matrix
    results_df = pd.DataFrame(results).T.sort_values(by="F1-Score", ascending=False)
    print("\nmodels (Sorted by F1-Score):")
    print(results_df.to_string())
    print("="*50)

    # --- EXPORT BEST MODEL ---
    print(f"Saving the best model ('{best_model_name}') to '{MODEL_PATH}'...")
    joblib.dump(best_model_object, MODEL_PATH)
    
if __name__ == "__main__":
    main()