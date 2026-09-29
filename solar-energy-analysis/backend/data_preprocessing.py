"""
data_preprocessing.py
Loads, validates, and cleans the solar energy dataset.
Saves cleaned data to backend/data/cleaned_solar_energy.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH     = DATA_DIR / "solar_energy.csv"
CLEANED_PATH = DATA_DIR / "cleaned_solar_energy.csv"


def load_and_clean(raw_path=RAW_PATH) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(raw_path)
    summary = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
    }

    # --- Missing values ---
    summary["missing_before"] = int(df.isnull().sum().sum())

    # --- Duplicates ---
    dup_count = int(df.duplicated().sum())
    summary["duplicates_removed"] = dup_count
    df = df.drop_duplicates()

    # --- Timestamp ---
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    ts_invalid = int(df["Timestamp"].isnull().sum())
    summary["invalid_timestamps"] = ts_invalid
    df = df.dropna(subset=["Timestamp"])

    # --- Negative / invalid Solar_Generation ---
    neg_gen = int((df["Solar_Generation"] < 0).sum())
    summary["negative_generation_fixed"] = neg_gen
    df.loc[df["Solar_Generation"] < 0, "Solar_Generation"] = np.nan

    # --- Cloud_Cover out of range ---
    bad_cloud = int(((df["Cloud_Cover"] < 0) | (df["Cloud_Cover"] > 100)).sum())
    summary["invalid_cloud_cover_fixed"] = bad_cloud
    df.loc[(df["Cloud_Cover"] < 0) | (df["Cloud_Cover"] > 100), "Cloud_Cover"] = np.nan

    # --- Negative Sunlight ---
    bad_sun = int((df["Sunlight"] < 0).sum())
    summary["invalid_sunlight_fixed"] = bad_sun
    df.loc[df["Sunlight"] < 0, "Sunlight"] = np.nan

    # --- Fill missing numeric values ---
    # Use median per Installation_ID to preserve site-level patterns
    num_cols = ["Solar_Generation", "Sunlight", "Temperature", "Cloud_Cover"]
    for col in num_cols:
        df[col] = df.groupby("Installation_ID")[col].transform(
            lambda x: x.fillna(x.median())
        )
    # Fallback global median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # --- Panel_Condition missing ---
    df["Panel_Condition"] = df["Panel_Condition"].fillna("Good")

    # --- Sort chronologically ---
    df = df.sort_values(["Timestamp", "Installation_ID"]).reset_index(drop=True)

    summary["missing_after"] = int(df.isnull().sum().sum())
    summary["cleaned_rows"] = len(df)
    summary["invalid_values_fixed"] = neg_gen + bad_cloud + bad_sun

    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned dataset saved. Rows: {len(df)}")
    return df, summary


if __name__ == "__main__":
    _, s = load_and_clean()
    for k, v in s.items():
        print(f"  {k}: {v}")
