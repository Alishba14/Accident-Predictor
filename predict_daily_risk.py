import hashlib
import urllib.request
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd

MODEL_PATH = Path("accident_model.pkl")
HASH_PATH = Path("accident_model.pkl.sha256")
LOG_PATH = Path("daily_risk_log.csv")

# Set the coordinates for the location you want to track (e.g., Karachi, Pakistan)
LATITUDE = 24.8607
LONGITUDE = 67.0011

def get_live_weather(lat: float, lon: float) -> tuple:
    """Fetches real-time precipitation and visibility from Open-Meteo free API.

    Raises RuntimeError if data cannot be fetched — fail-closed by design to
    prevent a silent false Low Risk output when the data source is unavailable.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=precipitation,visibility&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode())
    current = data.get("current")
    if current is None:
        raise RuntimeError("API response is missing the 'current' field.")
    precipitation = float(current.get("precipitation", 0.0))
    visibility_m = float(current.get("visibility", 10000.0))
    visibility_km = min(max(visibility_m / 1000.0, 0.3), 15.0)
    return precipitation, visibility_km


def _verify_model_integrity() -> None:
    """Verifies the model file has not been tampered with using its saved SHA-256 hash."""
    if not HASH_PATH.exists():
        print("Warning: No integrity hash found for model file. Skipping verification.")
        return
    expected = HASH_PATH.read_text().strip()
    actual = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            "Model integrity check FAILED — the .pkl file may have been tampered with.\n"
            f"Expected: {expected}\n"
            f"Got:      {actual}"
        )

def map_risk(probability: float) -> str:
    if probability > 0.70:
        return "High Risk"
    if probability > 0.40:
        return "Medium Risk"
    return "Low Risk"

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train_model.py first."
        )

    _verify_model_integrity()

    # 1. Fetch real weather data (fail-closed — raises if data is unavailable)
    print("Fetching live environmental metrics...")
    try:
        precipitation, visibility = get_live_weather(LATITUDE, LONGITUDE)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot run prediction: live weather data is unavailable ({exc}). "
            "Resolve the connection issue before predicting to avoid a false Low Risk result."
        ) from exc
    
    # 2. Estimate live traffic density index based on rush hours (06:00 UTC = 11:00 AM PKT)
    karachi_tz = ZoneInfo("Asia/Karachi")
    current_local_hour = datetime.now(karachi_tz).hour

    if current_local_hour in [8, 9, 10, 11, 16, 17, 18, 19]: 
        traffic_density = 0.85
    else:
        traffic_density = 0.40

    todays_conditions = {
        "precipitation_mm": precipitation,
        "visibility_km": visibility,
        "traffic_density_index": traffic_density,
    }

    print("\n" + "="*50)
    print("LIVE ACCIDENT RISK INFERENCE MONITOR")
    print("="*50)
    print(f"Current Local Karachi Time: {datetime.now(karachi_tz).strftime('%Y-%m-%d %I:%M %p')}")
    print(f"Fetched Precipitation  : {precipitation} mm")
    print(f"Fetched Visibility     : {visibility} km")
    print(f"Calculated Traffic Index: {traffic_density} ({'RUSH HOUR' if traffic_density == 0.85 else 'Normal Flow'})")
    print("-" * 50)

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([todays_conditions])
    raw_proba = model.predict_proba(input_df)[0][1]
    # Bounds-validate model output — guards against a tampered or corrupt model file
    if not (0.0 <= raw_proba <= 1.0):
        raise RuntimeError(
            f"Model returned an invalid probability ({raw_proba:.6f}). "
            "The model file may be corrupt or tampered with."
        )
    accident_probability = float(raw_proba)
    risk_level = map_risk(accident_probability)

    if precipitation == 0.0 and visibility >= 10.0:
        if traffic_density == 0.85:
            # Overwrite High Risk to Medium Risk because it's just heavy traffic, no hazardous weather
            if risk_level == "High Risk":
                risk_level = "Medium Risk"
        else:
            # Overwrite everything to Low Risk because weather is perfect and traffic is flowing smoothly
            risk_level = "Low Risk"

    # Atomic log file initialization — fixes TOCTOU race condition
    try:
        LOG_PATH.touch(exist_ok=False)
        pd.DataFrame(
            columns=["Date", "Precipitation", "Visibility", "Traffic_Density", "Probability", "Risk_Level"]
        ).to_csv(LOG_PATH, index=False)
    except FileExistsError:
        pass  # File already exists — safe to append

    new_row = pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Precipitation": todays_conditions["precipitation_mm"],
        "Visibility": todays_conditions["visibility_km"],
        "Traffic_Density": todays_conditions["traffic_density_index"],
        "Probability": round(accident_probability, 4),
        "Risk_Level": risk_level,
    }])

    new_row.to_csv(LOG_PATH, mode="a", index=False, header=False)
    print(f"Logged successfully: Prob={accident_probability:.2f}, Risk={risk_level}")

if __name__ == "__main__":
    main()