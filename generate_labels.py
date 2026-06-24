import pandas as pd
import numpy as np

# 1. Load the original API data
df = pd.read_csv("historical_accidents.csv")

# 2. Calculate a realistic base risk score 
rain_effect = df["precipitation_mm"] * 0.40
visibility_effect = (15.0 - df["visibility_km"]) * 0.05
traffic_effect = df["traffic_density_index"] * 0.35
calculated_risk = rain_effect + visibility_effect + traffic_effect

# 3. Add random variation so it's a realistic machine learning problem
rng = np.random.default_rng(seed=42)
random_noise = rng.uniform(-0.1, 0.1, size=len(df))
final_score = calculated_risk + random_noise

# 4. LOWER THE THRESHOLD AND GUARANTEE AT LEAST SOMe 1s
# If the score is higher than 0.30, mark it as an accident
df["accident_occurred"] = (final_score > 0.30).astype(int)

# SAFETY CHECK: If it still generated too few accidents, force the top 15% highest risk hours to be 1s
if df["accident_occurred"].sum() < 20:
    print("⚠️ Risk scores were low. Forcing the top 15% most dangerous hours to be marked as accidents...")
    threshold_value = final_score.quantile(0.85)
    df["accident_occurred"] = (final_score >= threshold_value).astype(int)

# 5. Save it back to the CSV file
df.to_csv("historical_accidents.csv", index=False)

total_accidents = df["accident_occurred"].sum()
total_rows = len(df)
print(f"Data Fixed successfully!")
print(f"Total Rows: {total_rows}")
print(f"Total Accidents (1s): {total_accidents} ({ (total_accidents/total_rows)*100 :.1f}%)")
print("Now you can safely run 'python train_model.py' without crashes!")