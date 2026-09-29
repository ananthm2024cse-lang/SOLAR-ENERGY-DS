"""
feature_engineering.py
Creates model-ready features from the cleaned dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

CLEANED_PATH = Path(__file__).parent / "data" / "cleaned_solar_energy.csv"

SEASON_MAP = {12: 0, 1: 0, 2: 0,   # Winter
              3: 1, 4: 1, 5: 1,    # Spring
              6: 2, 7: 2, 8: 2,    # Summer
              9: 3, 10: 3, 11: 3}  # Autumn

FEATURES = ["Installation_ID_enc", "Sunlight", "Temperature", "Cloud_Cover",
            "Panel_Condition_enc", "Hour", "Day_of_Week", "Month", "Season"]
TARGET   = "Solar_Generation"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    df = df.copy()
    df["Timestamp"]    = pd.to_datetime(df["Timestamp"])
    df["Hour"]         = df["Timestamp"].dt.hour
    df["Day"]          = df["Timestamp"].dt.day
    df["Day_of_Week"]  = df["Timestamp"].dt.dayofweek
    df["Month"]        = df["Timestamp"].dt.month
    df["Season"]       = df["Month"].map(SEASON_MAP)
    df["Is_Weekend"]   = (df["Day_of_Week"] >= 5).astype(int)

    le_ins   = LabelEncoder()
    le_panel = LabelEncoder()
    df["Installation_ID_enc"]  = le_ins.fit_transform(df["Installation_ID"])
    df["Panel_Condition_enc"]  = le_panel.fit_transform(df["Panel_Condition"])

    return df, le_ins, le_panel


def get_X_y(df: pd.DataFrame):
    df_feat, le_ins, le_panel = build_features(df)
    X = df_feat[FEATURES]
    y = df_feat[TARGET]
    return X, y, df_feat, le_ins, le_panel


if __name__ == "__main__":
    df = pd.read_csv(CLEANED_PATH)
    X, y, df_feat, _, _ = get_X_y(df)
    print("Feature matrix shape:", X.shape)
    print("Features:", list(X.columns))
