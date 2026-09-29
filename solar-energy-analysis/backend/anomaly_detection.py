"""
anomaly_detection.py
Detects abnormal drops in solar generation using rolling averages and model residuals.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from feature_engineering import get_X_y

CLEANED_PATH = Path(__file__).parent / "data" / "cleaned_solar_energy.csv"
MODEL_PATH   = Path(__file__).parent / "models" / "xgboost_model.json"

SUNLIGHT_THRESHOLD = 50   # W/m² — ignore near-dark periods
DROP_THRESHOLD     = 0.45  # actual < 45% of expected → anomaly


def _reason(row: pd.Series) -> str:
    if row["Panel_Condition"] == "Poor":
        return "Poor panel condition detected"
    if row["Cloud_Cover"] > 70:
        return "High cloud cover reducing output"
    if row["drop_pct"] > 0.80:
        return "Possible equipment malfunction — sudden output reduction"
    if row["Cloud_Cover"] > 40:
        return "Unexpected weather change"
    return "Unexplained output reduction — investigate further"


def _severity(drop_pct: float) -> str:
    if drop_pct > 0.80:
        return "High"
    if drop_pct > 0.60:
        return "Medium"
    return "Low"


def detect_anomalies() -> list[dict]:
    df = pd.read_csv(CLEANED_PATH, parse_dates=["Timestamp"])
    df = df.sort_values(["Installation_ID", "Timestamp"]).reset_index(drop=True)

    # Load model and get predictions
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    X, y, df_feat, _, _ = get_X_y(df)
    df_feat["predicted"] = np.clip(model.predict(X), 0, None)

    # Only consider daytime records with meaningful sunlight
    mask = df_feat["Sunlight"] >= SUNLIGHT_THRESHOLD
    day_df = df_feat[mask].copy()

    day_df["drop_pct"] = 1 - (day_df["Solar_Generation"] /
                               day_df["predicted"].replace(0, np.nan))
    day_df["drop_pct"] = day_df["drop_pct"].clip(0, 1)

    anomalies = day_df[day_df["drop_pct"] >= DROP_THRESHOLD].copy()

    results = []
    for _, row in anomalies.iterrows():
        results.append({
            "timestamp":          str(row["Timestamp"]),
            "installation_id":    row["Installation_ID"],
            "actual_generation":  round(float(row["Solar_Generation"]), 4),
            "expected_generation":round(float(row["predicted"]), 4),
            "difference":         round(float(row["predicted"] - row["Solar_Generation"]), 4),
            "severity":           _severity(row["drop_pct"]),
            "reason":             _reason(row),
        })

    # Sort by severity then timestamp
    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    results.sort(key=lambda x: (sev_order[x["severity"]], x["timestamp"]))
    return results


if __name__ == "__main__":
    anomalies = detect_anomalies()
    print(f"Detected {len(anomalies)} anomalies")
    for a in anomalies[:5]:
        print(a)
