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
    # Change USE_REAL_DATA to True once you have a real historical dataset file
    USE_REAL_DATA = False
    HISTORICAL_DATA_PATH = "historical_accidents.csv"

    if USE_REAL_DATA:
        print(f"Loading real-world data from {HISTORICAL_DATA_PATH}...")
        df = load_real_data(HISTORICAL_DATA_PATH)
    else:
        print("Using synthetic fallback dataset...")
        df = generate_synthetic_data(n_rows=1000)  # Upgraded size for better variance

    feature_cols = [
        "precipitation_mm",
        "visibility_km",
        "traffic_density_index",
    ]
    X = df[feature_cols]
    y = df["accident_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()