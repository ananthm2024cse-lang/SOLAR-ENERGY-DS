"""
train_model.py
Trains an XGBoost regression model on the cleaned solar energy dataset.
Saves the model to backend/models/xgboost_model.json
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from xgboost import XGBRegressor
from feature_engineering import get_X_y, FEATURES

CLEANED_PATH  = Path(__file__).parent / "data" / "cleaned_solar_energy.csv"
MODEL_PATH    = Path(__file__).parent / "models" / "xgboost_model.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def train():
    df = pd.read_csv(CLEANED_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    X, y, df_feat, le_ins, le_panel = get_X_y(df)

    # Chronological split — last 20% as test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    model.save_model(str(MODEL_PATH))
    print("Model saved.")

    # Save label encoder mappings as JSON for the API
    enc_path = Path(__file__).parent / "models" / "encoders.json"
    enc_data = {
        "installation_classes": list(le_ins.classes_),
        "panel_classes":        list(le_panel.classes_),
    }
    enc_path.write_text(json.dumps(enc_data, indent=2))
    print("Encoders saved.")

    return model, X_test, y_test, df_feat


if __name__ == "__main__":
    train()
