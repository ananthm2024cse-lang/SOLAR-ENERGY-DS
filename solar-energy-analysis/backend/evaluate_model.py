"""
evaluate_model.py
Evaluates the trained XGBoost model and saves metrics + feature importance.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import get_X_y, FEATURES

CLEANED_PATH   = Path(__file__).parent / "data" / "cleaned_solar_energy.csv"
MODEL_PATH     = Path(__file__).parent / "models" / "xgboost_model.json"
METRICS_PATH   = Path(__file__).parent / "outputs" / "model_metrics.json"
FI_PATH        = Path(__file__).parent / "outputs" / "feature_importance.png"
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def evaluate():
    df = pd.read_csv(CLEANED_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    X, y, df_feat, _, _ = get_X_y(df)

    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    ts_test = df_feat["Timestamp"].iloc[split:].reset_index(drop=True)

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {"MAE": round(mae, 4), "MSE": round(mse, 4),
               "RMSE": round(rmse, 4), "R2": round(r2, 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Metrics:", metrics)

    # Actual vs Predicted sample (500 points)
    sample_n = min(500, len(y_test))
    actual_vs_pred = pd.DataFrame({
        "timestamp": ts_test[:sample_n].astype(str).tolist(),
        "actual":    np.round(y_test.values[:sample_n], 4).tolist(),
        "predicted": np.round(y_pred[:sample_n], 4).tolist(),
    })
    avp_path = Path(__file__).parent / "outputs" / "actual_vs_predicted.json"
    avp_path.write_text(actual_vs_pred.to_json(orient="records"))

    # Feature importance plot
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    fi.plot(kind="barh", color="#2ecc71", ax=ax)
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(str(FI_PATH), dpi=100)
    plt.close(fig)
    print("Feature importance plot saved.")

    # Save feature importance as JSON
    fi_json_path = Path(__file__).parent / "outputs" / "feature_importance.json"
    fi_dict = {k: round(float(v), 6) for k, v in
               sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)}
    fi_json_path.write_text(json.dumps(fi_dict, indent=2))

    return metrics


if __name__ == "__main__":
    evaluate()
