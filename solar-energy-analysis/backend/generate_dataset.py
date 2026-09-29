"""
generate_dataset.py
Generates a realistic synthetic solar energy dataset (sample data).
Run this once to create backend/data/solar_energy.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

INSTALLATIONS = ["INS_A", "INS_B", "INS_C", "INS_D", "INS_E"]
CAPACITIES = {"INS_A": 1.0, "INS_B": 0.85, "INS_C": 1.15, "INS_D": 0.70, "INS_E": 1.05}
PANEL_PROBS = {"INS_A": [0.7, 0.2, 0.1], "INS_B": [0.5, 0.3, 0.2],
               "INS_C": [0.8, 0.15, 0.05], "INS_D": [0.3, 0.4, 0.3], "INS_E": [0.6, 0.25, 0.15]}

start = pd.Timestamp("2023-01-01")
end   = pd.Timestamp("2023-12-31 23:00:00")
timestamps = pd.date_range(start, end, freq="h")

records = []
for ts in timestamps:
    hour  = ts.hour
    month = ts.month
    # Sunlight: bell curve peaking at noon, seasonal amplitude
    seasonal = 1.0 + 0.3 * np.sin((month - 3) * np.pi / 6)
    sun_peak = 900 * seasonal
    if 6 <= hour <= 18:
        sun_factor = np.sin((hour - 6) * np.pi / 12)
        sunlight = max(0, sun_peak * sun_factor + np.random.normal(0, 40))
    else:
        sunlight = max(0, np.random.normal(0, 5))

    temp = 15 + 10 * np.sin((month - 1) * np.pi / 6) + (hour - 12) * 0.3 + np.random.normal(0, 3)
    cloud = np.clip(np.random.beta(2, 5) * 100, 0, 100)

    for ins in INSTALLATIONS:
        panel = np.random.choice(["Good", "Moderate", "Poor"], p=PANEL_PROBS[ins])
        panel_eff = {"Good": 1.0, "Moderate": 0.85, "Poor": 0.65}[panel]
        cloud_eff = 1 - (cloud / 100) * 0.75
        temp_eff  = 1 - max(0, (temp - 25) * 0.004)
        cap = CAPACITIES[ins]

        if sunlight < 10:
            gen = max(0, np.random.normal(0, 0.02))
        else:
            base = (sunlight / 1000) * cap * panel_eff * cloud_eff * temp_eff
            gen  = max(0, base + np.random.normal(0, 0.03))

        records.append({
            "Timestamp": ts,
            "Installation_ID": ins,
            "Solar_Generation": round(gen, 4),
            "Sunlight": round(sunlight, 2),
            "Temperature": round(temp, 2),
            "Cloud_Cover": round(cloud, 2),
            "Panel_Condition": panel,
        })

df = pd.DataFrame(records)

# --- Inject anomalies (sudden drops) ---
anomaly_idx = np.random.choice(df[df["Solar_Generation"] > 0.1].index, size=80, replace=False)
df.loc[anomaly_idx, "Solar_Generation"] *= np.random.uniform(0.05, 0.25, size=80)

# --- Inject missing values ---
for col in ["Solar_Generation", "Sunlight", "Temperature", "Cloud_Cover"]:
    miss_idx = np.random.choice(df.index, size=60, replace=False)
    df.loc[miss_idx, col] = np.nan

# --- Inject duplicates ---
dup_rows = df.sample(30, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# --- Inject invalid values ---
inv_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[inv_idx, "Solar_Generation"] = -np.random.uniform(0.1, 0.5, size=20)
inv_cloud = np.random.choice(df.index, size=10, replace=False)
df.loc[inv_cloud, "Cloud_Cover"] = np.random.uniform(101, 150, size=10)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = Path(__file__).parent / "data" / "solar_energy.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Dataset saved. Rows: {len(df)}")
