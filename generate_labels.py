import shutil
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("historical_accidents.csv")

print(
    "NOTE: This script generates SYNTHETIC labels from a scoring formula. "
    "It does NOT use real accident reports. Metrics from a model trained on "
    "this data reflect how well it learned the formula, not real-world accuracy."
)

# 1. Load the original API data
df = pd.read_csv(DATA_PATH)

# Backup before overwriting so a bad run can be recovered
if DATA_PATH.exists():
    backup = DATA_PATH.with_suffix(".csv.bak")
    shutil.copy2(DATA_PATH, backup)
    print(f"Backup saved to '{backup}'")

# 2. Calculate a realistic base risk score 
rain_effect = df["precipitation_mm"] * 0.40
visibility_effect = (15.0 - df["visibility_km"]) * 0.05
traffic_effect = df["traffic_density_index"] * 0.35
calculated_risk = rain_effect + visibility_effect + traffic_effect

# 3. Add random variation so it's a realistic machine learning problem
rng = np.random.default_rng(seed=42)
random_noise = rng.uniform(-0.1, 0.1, size=len(df))
final_score = calculated_risk + random_noise

# 4. Threshold of 0.50 produces a realistic ~25-30% positive rate.
#    (0.30 was too low: Karachi's typical visibility alone pushes the base
#    score to ~0.39, labelling nearly every row as an accident.)
df["accident_occurred"] = (final_score > 0.50).astype(int)

# SAFETY CHECK: If it still generated too few accidents, force the top 15% highest risk hours to be 1s
if df["accident_occurred"].sum() < 20:
    print("⚠️ Risk scores were low. Forcing the top 15% most dangerous hours to be marked as accidents...")
    threshold_value = final_score.quantile(0.85)
    df["accident_occurred"] = (final_score >= threshold_value).astype(int)

# 5. Save it back to the CSV file
df.to_csv(DATA_PATH, index=False)

total_accidents = df["accident_occurred"].sum()
total_rows = len(df)
print(f"Data Fixed successfully!")
print(f"Total Rows: {total_rows}")
print(f"Total Accidents (1s): {total_accidents} ({ (total_accidents/total_rows)*100 :.1f}%)")
print("Now you can safely run 'python train_model.py' without crashes!")