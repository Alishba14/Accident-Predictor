from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

# Import functions from your project scripts
from train_model import generate_synthetic_data
from predict_daily_risk import map_risk

MODEL_PATH = Path("accident_model.pkl")

def test_synthetic_data_generation():
    """Test that the synthetic data generator outputs data in the expected shape and bounds."""
    n_rows = 50
    df = generate_synthetic_data(n_rows=n_rows)
    
    # Check if output is a DataFrame and has correct number of rows
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_rows
    
    # Check that required features exist
    required_cols = ["precipitation_mm", "visibility_km", "traffic_density_index", "accident_occurred"]
    for col in required_cols:
        assert col in df.columns

    # Check bounds logic from train_model.py
    assert df["traffic_density_index"].min() >= 0.0
    assert df["traffic_density_index"].max() <= 1.0
    assert df["visibility_km"].min() >= 0.3
    assert df["visibility_km"].max() <= 15.0


def test_map_risk_logic():
    """Test that risk mapping classifications respond correctly to probability thresholds."""
    assert map_risk(0.85) == "Low Risk"
    assert map_risk(0.55) == "Medium Risk"
    assert map_risk(0.20) == "Low Risk"


def test_model_file_exists_and_loads():
    """Ensure the trained model binary exists and can be parsed by joblib."""
    assert MODEL_PATH.exists(), "Model binary is missing! Run train_model.py first."
    
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Loaded object is not a valid classifier."


def test_model_prediction_schema():
    """Verify the model accepts the correct feature shape and predicts binary targets."""
    model = joblib.load(MODEL_PATH)
    
    # Mock dataframe mimicking exactly what the model expects
    mock_input = pd.DataFrame([{
        "precipitation_mm": 5.0,
        "visibility_km": 10.0,
        "traffic_density_index": 0.5
    }])
    
    prediction = model.predict(mock_input)
    probabilities = model.predict_proba(mock_input)
    
    # Assert output values and formats conform to expectations
    assert prediction[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert np.isclose(np.sum(probabilities[0]), 1.0)