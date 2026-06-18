import urllib.request
import json
from datetime import date, datetime
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("accident_model.pkl")
LOG_PATH = Path("daily_risk_log.csv")

# Set the coordinates for the location you want to track (e.g., Karachi, Pakistan)
LATITUDE = 24.8607
LONGITUDE = 67.0011

def get_live_weather(lat, lon):
    """Fetches real-time precipitation and visibility data from Open-Meteo free API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,visibility&timezone=auto"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            current = data["current"]
            
            precipitation = float(current.get("precipitation", 0.0))
            # Convert visibility from meters to kilometers safely
            visibility_m = float(current.get("visibility", 10000.0))
            visibility_km = min(max(visibility_m / 1000.0, 0.3), 15.0)
            
            return precipitation, visibility_km
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch live weather ({e}). Using baseline safe conditions.")
        return 0.0, 10.0

def map_risk(probability: float) -> str:
    if probability > 0.70:
        return "High Risk"
    if probability > 0.40:
        return "Medium Risk"
    return "Low Risk"

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Make sure it's committed to the repository."
        )

    # 1. Fetch real weather data
    print("🌐 Fetching live environmental metrics...")
    precipitation, visibility = get_live_weather(LATITUDE, LONGITUDE)
    
    # 2. Estimate live traffic density index based on rush hours (06:00 UTC = 11:00 AM PKT)
    current_hour = datetime.utcnow().hour
    if current_hour in [3, 4, 5, 6, 11, 12, 13]: # Rush hours
        traffic_density = 0.85
    else:
        traffic_density = 0.40

    todays_conditions = {
        "precipitation_mm": precipitation,
        "visibility_km": visibility,
        "traffic_density_index": traffic_density,
    }

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([todays_conditions])
    accident_probability = float(model.predict_proba(input_df)[0][1])
    risk_level = map_risk(accident_probability)

    if not LOG_PATH.exists():
        pd.DataFrame(
            columns=["Date", "Precipitation", "Visibility", "Traffic_Density", "Probability", "Risk_Level"]
        ).to_csv(LOG_PATH, index=False)

    new_row = pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Precipitation": todays_conditions["precipitation_mm"],
        "Visibility": todays_conditions["visibility_km"],
        "Traffic_Density": todays_conditions["traffic_density_index"],
        "Probability": round(accident_probability, 4),
        "Risk_Level": risk_level,
    }])

    new_row.to_csv(LOG_PATH, mode="a", index=False, header=False)
    print(f"✅ Logged successfully: Prob={accident_probability:.2f}, Risk={risk_level}")

if __name__ == "__main__":
    main()