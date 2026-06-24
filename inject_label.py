import numpy as np
import pandas as pd

# 1. Load your API-fetched data
df = pd.read_csv("historical_accidents.csv")

# 2. Define a baseline probability (accidents are generally rare)
base_risk = 0.02 

# 3. Increase risk dynamically based on environmental conditions
rain_risk = df["precipitation_mm"] * 0.15
fog_risk = (15.0 - df["visibility_km"]) * 0.03
traffic_risk = df["traffic_density_index"] * 0.25

# Combine factors into a cumulative probability score
total_probability = base_risk + rain_risk + fog_risk + traffic_risk
total_probability = total_probability.clip(0.01, 0.85) # Clamp boundaries

# 4. Use a random choice generator to apply 0 or 1 based on that calculated probability
rng = np.random.default_rng(seed=42)
df["accident_occurred"] = rng.binomial(1, total_probability)

# 5. Save it back to the CSV
df.to_csv("historical_accidents.csv", index=False)

total_accidents = df["accident_occurred"].sum()
print(f"Processed with realistic noise. Generated {total_accidents} accidents out of {len(df)} total hours.")