import urllib.request
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

# --- PROJECT CONFIGURATION PARAMETERS ---
LATITUDE = 24.8607
LONGITUDE = 67.0011
OUTPUT_CSV_PATH = "historical_accidents.csv"
KARACHI_TZ = ZoneInfo("Asia/Karachi")

def get_traffic_density(dt_utc: datetime) -> float:
    """
    Computes traffic density based on local Karachi rush hours.
    Converts the UTC timestamp to Asia/Karachi timezone to avoid the hour-shifting bug.
    """
    # Localize UTC time and convert to Karachi local time
    dt_local = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(KARACHI_TZ)
    local_hour = dt_local.hour
    
    # Standard local traffic rush hours in Karachi:
    # Morning rush (8 AM - 11 AM) and Evening rush (4 PM - 7 PM)
    if local_hour in [8, 9, 10, 11, 16, 17, 18, 19]:
        return 0.85
    else:
        return 0.40

def fetch_historical_training_data(start_date: str, end_date: str):
    """
    Fetches hourly precipitation and visibility from Open-Meteo Archive API,
    applies the project's data engineering scaling rules, merges local traffic density,
    and saves the output as a training-ready CSV dataset.
    
    Parameters:
        start_date (str): format "YYYY-MM-DD" (e.g., "2025-01-01")
        end_date (str): format "YYYY-MM-DD" (e.g., "2025-12-31")
    """
    # Open-Meteo Archive API URL constructed with project parameters
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=precipitation,visibility&timezone=auto"
    )
    
    print(f"Fetching timeline: {start_date} to {end_date}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            hourly = data["hourly"]
            
            # 1. Parse raw lists into a DataFrame
            df = pd.DataFrame({
                "timestamp_utc": pd.to_datetime(hourly["time"]),
                "precipitation_mm": hourly["precipitation"],
                "visibility_m": hourly["visibility"]
            })
            
            print(f"Downloaded {len(df)} hourly rows. Beginning data conversion...")
            
            # 2. Match exact system preprocessing rule for precipitation
            df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0).astype(float)
            
            # 3. Match exact system preprocessing rule for visibility:
            # Convert meters to kilometers and clamp strictly between 0.3km and 15.0km
            df["visibility_km"] = df["visibility_m"] / 1000.0
            df["visibility_km"] = df["visibility_km"].fillna(10.0) # default baseline
            df["visibility_km"] = df["visibility_km"].clip(0.3, 15.0).round(1)
            
            # 4. Generate Traffic Density Index mapping using the local Karachi time correction
            print("Calculating timezone-adjusted Traffic Density Index metrics...")
            df["traffic_density_index"] = df["timestamp_utc"].apply(get_traffic_density)
            
            # 5. Initialize the ground truth target column. 
            # (Set to 0 by default; you will replace these 0s with 1s using actual accident reports)
            df["accident_occurred"] = 0
            
            # 6. Final cleanup to match what train_model.py expects
            df = df.drop(columns=["visibility_m", "timestamp_utc"])
            
            # Reorder columns explicitly to fit load_real_data() schema
            final_columns = ["precipitation_mm", "visibility_km", "traffic_density_index", "accident_occurred"]
            df = df[final_columns]
            
            # Save out
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Success! Training-ready file saved to: '{OUTPUT_CSV_PATH}' ({len(df)} rows)")
            print("\nNext Step: Open the CSV and flip 'accident_occurred' from 0 to 1 for hours where a real accident happened, then set USE_REAL_DATA = True in train_model.py!")
            
    except Exception as e:
        print(f"Failed to fetch archive data from API: {e}")

if __name__ == "__main__":
    # Adjust your historical training window range here
    START_WINDOW = "2025-01-01"
    END_WINDOW = "2025-12-31"
    
    fetch_historical_training_data(start_date=START_WINDOW, end_date=END_WINDOW)