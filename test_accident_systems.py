from pathlib import Path
from datetime import datetime
import io

import numpy as np
import pandas as pd
import pytest
import joblib

from train_model import generate_synthetic_data, load_real_data
from predict_daily_risk import map_risk
from fetch_data import get_traffic_density

MODEL_PATH = Path("accident_model.pkl")

# Decorator: skip model-dependent tests gracefully on a fresh clone
model_required = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="accident_model.pkl not present — run train_model.py first",
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def test_synthetic_data_generation():
    """Test that the synthetic data generator outputs data in the expected shape and bounds."""
    n_rows = 50
    df = generate_synthetic_data(n_rows=n_rows)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_rows

    required_cols = ["precipitation_mm", "visibility_km", "traffic_density_index", "accident_occurred"]
    for col in required_cols:
        assert col in df.columns

    assert df["traffic_density_index"].min() >= 0.0
    assert df["traffic_density_index"].max() <= 1.0
    assert df["visibility_km"].min() >= 0.3
    assert df["visibility_km"].max() <= 15.0
    assert set(df["accident_occurred"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# map_risk thresholds — including exact boundary values
# ---------------------------------------------------------------------------

def test_map_risk_logic():
    """Test that risk mapping classifications respond correctly to probability thresholds."""
    assert map_risk(0.85) == "High Risk"
    assert map_risk(0.55) == "Medium Risk"
    assert map_risk(0.20) == "Low Risk"


def test_map_risk_exact_boundaries():
    """Exact boundary values must fall into the correct (lower) bucket."""
    # 0.70 is NOT > 0.70, so it maps to Medium Risk
    assert map_risk(0.70) == "Medium Risk"
    assert map_risk(0.71) == "High Risk"
    # 0.40 is NOT > 0.40, so it maps to Low Risk
    assert map_risk(0.40) == "Low Risk"
    assert map_risk(0.41) == "Medium Risk"


# ---------------------------------------------------------------------------
# Traffic density index (Karachi rush-hour logic)
# ---------------------------------------------------------------------------

def test_traffic_density_rush_hour():
    """8 AM Karachi (UTC+5) = 03:00 UTC should return rush-hour density."""
    dt_utc = datetime(2025, 6, 1, 3, 0, 0)  # 03:00 UTC = 08:00 Karachi
    assert get_traffic_density(dt_utc) == 0.85


def test_traffic_density_off_peak():
    """12 PM Karachi = 07:00 UTC is not a rush hour."""
    dt_utc = datetime(2025, 6, 1, 7, 0, 0)  # 07:00 UTC = 12:00 Karachi
    assert get_traffic_density(dt_utc) == 0.40


def test_traffic_density_evening_rush():
    """5 PM Karachi (17:00) = 12:00 UTC should return rush-hour density."""
    dt_utc = datetime(2025, 6, 1, 12, 0, 0)  # 12:00 UTC = 17:00 Karachi
    assert get_traffic_density(dt_utc) == 0.85


# ---------------------------------------------------------------------------
# load_real_data — edge cases
# ---------------------------------------------------------------------------

def test_load_real_data_missing_required_column(tmp_path):
    """load_real_data must raise when a required column is absent."""
    csv = "precipitation_mm,visibility_km\n1.0,10.0\n2.0,8.0\n"
    p = tmp_path / "bad.csv"
    p.write_text(csv)
    with pytest.raises((KeyError, ValueError)):
        load_real_data(str(p))


def test_load_real_data_all_zero_labels_raises(tmp_path):
    """load_real_data must raise ValueError when every label is 0."""
    csv = (
        "precipitation_mm,visibility_km,traffic_density_index,accident_occurred\n"
        "1.0,10.0,0.40,0\n"
        "2.0,8.0,0.85,0\n"
    )
    p = tmp_path / "zero_labels.csv"
    p.write_text(csv)
    with pytest.raises(ValueError, match="zero positive labels"):
        load_real_data(str(p))


def test_load_real_data_clips_negative_precipitation(tmp_path):
    """Negative precipitation must be clipped to 0 without raising."""
    csv = (
        "precipitation_mm,visibility_km,traffic_density_index,accident_occurred\n"
        "-3.0,10.0,0.40,1\n"
        "1.0,8.0,0.85,0\n"
    )
    p = tmp_path / "neg_precip.csv"
    p.write_text(csv)
    df = load_real_data(str(p))
    assert (df["precipitation_mm"] >= 0).all()


def test_load_real_data_keeps_duplicates(tmp_path):
    """Duplicate rows are expected in coarse time-series features and must be preserved."""
    csv = (
        "precipitation_mm,visibility_km,traffic_density_index,accident_occurred\n"
        "1.0,10.0,0.40,1\n"
        "1.0,10.0,0.40,1\n"
    )
    p = tmp_path / "dupes.csv"
    p.write_text(csv)
    df = load_real_data(str(p))
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Model binary — skipped gracefully if not present
# ---------------------------------------------------------------------------

@model_required
def test_model_file_exists_and_loads():
    """Ensure the trained model binary exists and can be parsed by joblib."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Loaded object is not a valid classifier."


@model_required
def test_model_prediction_schema():
    """Verify the model accepts the correct feature shape and predicts binary targets."""
    model = joblib.load(MODEL_PATH)
    mock_input = pd.DataFrame([{
        "precipitation_mm": 5.0,
        "visibility_km": 10.0,
        "traffic_density_index": 0.5,
    }])
    prediction = model.predict(mock_input)
    probabilities = model.predict_proba(mock_input)

    assert prediction[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert np.isclose(np.sum(probabilities[0]), 1.0)
    # Probability must be in valid range
    assert 0.0 <= probabilities[0][1] <= 1.0
