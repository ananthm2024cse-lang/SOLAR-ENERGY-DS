"""
app.py
Flask REST API for the Solar Energy Analysis Dashboard.
Run: python backend/app.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from xgboost import XGBRegressor

from data_preprocessing import load_and_clean
from eda_analysis import (load_df, descriptive_stats, installation_stats,
                           daily_trend, monthly_trend, hourly_pattern,
                           seasonal_stats, weather_correlation, panel_condition_stats)
from feature_engineering import get_X_y, FEATURES, SEASON_MAP
from anomaly_detection import detect_anomalies

BASE      = Path(__file__).parent
MODEL_PATH   = BASE / "models" / "xgboost_model.json"
METRICS_PATH = BASE / "outputs" / "model_metrics.json"
FI_PATH      = BASE / "outputs" / "feature_importance.json"
AVP_PATH     = BASE / "outputs" / "actual_vs_predicted.json"
ENC_PATH     = BASE / "models" / "encoders.json"

app = Flask(__name__)
CORS(app)

# ── Cache heavy objects at startup ──────────────────────────────────────────
_df        = None
_model     = None
_encoders  = None
_clean_sum = None
_anomalies = None


def get_df():
    global _df
    if _df is None:
        _df = load_df()
    return _df


def get_model():
    global _model
    if _model is None:
        _model = XGBRegressor()
        _model.load_model(str(MODEL_PATH))
    return _model


def get_encoders():
    global _encoders
    if _encoders is None:
        _encoders = json.loads(ENC_PATH.read_text())
    return _encoders


def get_clean_summary():
    global _clean_sum
    if _clean_sum is None:
        _, _clean_sum = load_and_clean()
    return _clean_sum


def get_anomalies():
    global _anomalies
    if _anomalies is None:
        _anomalies = detect_anomalies()
    return _anomalies


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_json(obj):
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/api/summary")
def summary():
    df   = get_df()
    anoms = get_anomalies()
    inst  = installation_stats(df)
    inst_sorted = sorted(inst, key=lambda x: x["mean"])
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return jsonify({
        "total_energy_kwh":          round(float(df["Solar_Generation"].sum()), 2),
        "avg_hourly_generation":     round(float(df["Solar_Generation"].mean()), 4),
        "num_installations":         int(df["Installation_ID"].nunique()),
        "highest_installation":      inst_sorted[-1]["Installation_ID"],
        "lowest_installation":       inst_sorted[0]["Installation_ID"],
        "num_anomalies":             len(anoms),
        "r2_score":                  metrics.get("R2", "N/A"),
    })


@app.route("/api/generation-trend")
def generation_trend():
    df = get_df()
    return jsonify({
        "daily":   daily_trend(df),
        "monthly": monthly_trend(df),
        "hourly":  hourly_pattern(df),
        "seasonal":seasonal_stats(df),
    })


@app.route("/api/installation-performance")
def installation_performance():
    df   = get_df()
    inst = installation_stats(df)
    inst_sorted = sorted(inst, key=lambda x: x["mean"])
    return jsonify({
        "installations":  inst,
        "highest":        inst_sorted[-1]["Installation_ID"],
        "lowest":         inst_sorted[0]["Installation_ID"],
    })


@app.route("/api/weather-analysis")
def weather_analysis():
    df = get_df()
    corr = weather_correlation(df)
    # Sample scatter data (max 800 daytime points)
    day_df = df[df["Sunlight"] > 10].sample(min(800, len(df[df["Sunlight"] > 10])),
                                             random_state=42)
    scatter = day_df[["Sunlight", "Temperature", "Cloud_Cover",
                       "Solar_Generation"]].round(4).to_dict(orient="records")
    return jsonify({"correlations": corr, "scatter_data": scatter})


@app.route("/api/panel-condition")
def panel_condition():
    df = get_df()
    return jsonify(panel_condition_stats(df))


@app.route("/api/anomalies")
def anomalies():
    return jsonify(get_anomalies())


@app.route("/api/model-metrics")
def model_metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Model not trained yet"}), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))


@app.route("/api/feature-importance")
def feature_importance():
    if not FI_PATH.exists():
        return jsonify({"error": "Run evaluate_model.py first"}), 404
    return jsonify(json.loads(FI_PATH.read_text()))


@app.route("/api/actual-vs-predicted")
def actual_vs_predicted():
    if not AVP_PATH.exists():
        return jsonify({"error": "Run evaluate_model.py first"}), 404
    return jsonify(json.loads(AVP_PATH.read_text()))


@app.route("/api/cleaning-summary")
def cleaning_summary():
    return jsonify(get_clean_summary())


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["Installation_ID", "Sunlight", "Temperature",
                "Cloud_Cover", "Panel_Condition", "Hour",
                "Day_of_Week", "Month", "Season"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    enc = get_encoders()
    ins_classes   = enc["installation_classes"]
    panel_classes = enc["panel_classes"]

    ins_id = data["Installation_ID"]
    panel  = data["Panel_Condition"]
    if ins_id not in ins_classes:
        return jsonify({"error": f"Unknown Installation_ID: {ins_id}"}), 400
    if panel not in panel_classes:
        return jsonify({"error": f"Unknown Panel_Condition: {panel}"}), 400

    ins_enc   = ins_classes.index(ins_id)
    panel_enc = panel_classes.index(panel)

    try:
        row = pd.DataFrame([{
            "Installation_ID_enc": ins_enc,
            "Sunlight":            float(data["Sunlight"]),
            "Temperature":         float(data["Temperature"]),
            "Cloud_Cover":         float(data["Cloud_Cover"]),
            "Panel_Condition_enc": panel_enc,
            "Hour":                int(data["Hour"]),
            "Day_of_Week":         int(data["Day_of_Week"]),
            "Month":               int(data["Month"]),
            "Season":              int(data["Season"]),
        }])
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    model = get_model()
    pred  = float(np.clip(model.predict(row)[0], 0, None))
    return jsonify({"predicted_generation_kwh": round(pred, 4)})


@app.route("/api/insights")
def insights():
    df   = get_df()
    inst = installation_stats(df)
    inst_sorted = sorted(inst, key=lambda x: x["mean"])
    corr = weather_correlation(df)
    seasonal = seasonal_stats(df)
    seasonal_sorted = sorted(seasonal, key=lambda x: x["avg"])
    anoms = get_anomalies()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    best_season  = seasonal_sorted[-1]["Season"]
    worst_season = seasonal_sorted[0]["Season"]
    high_anoms   = [a for a in anoms if a["severity"] == "High"]

    insights_list = [
        f"Total energy generated: {round(float(df['Solar_Generation'].sum()), 1)} kWh across "
        f"{int(df['Installation_ID'].nunique())} installations.",

        f"{best_season} shows the highest average generation "
        f"({round(float(seasonal_sorted[-1]['avg']), 4)} kWh/hr), while "
        f"{worst_season} shows the lowest "
        f"({round(float(seasonal_sorted[0]['avg']), 4)} kWh/hr).",

        f"Sunlight intensity has the strongest positive correlation with generation "
        f"(r = {corr.get('Sunlight', 'N/A')}), confirming it as the primary driver.",

        f"Cloud cover shows a negative correlation with generation "
        f"(r = {corr.get('Cloud_Cover', 'N/A')}); high cloud cover significantly reduces output.",

        f"Installation {inst_sorted[-1]['Installation_ID']} is the top performer "
        f"(avg {round(float(inst_sorted[-1]['mean']), 4)} kWh/hr), while "
        f"{inst_sorted[0]['Installation_ID']} is the lowest "
        f"(avg {round(float(inst_sorted[0]['mean']), 4)} kWh/hr).",

        f"{len(anoms)} anomalous generation drops detected, including "
        f"{len(high_anoms)} high-severity events requiring immediate investigation.",

        f"XGBoost model achieved R2 = {metrics.get('R2', 'N/A')} with "
        f"RMSE = {metrics.get('RMSE', 'N/A')} kWh on the held-out test set.",
    ]
    return jsonify(insights_list)


@app.route("/api/action-plan")
def action_plan():
    df   = get_df()
    inst = installation_stats(df)
    inst_sorted = sorted(inst, key=lambda x: x["mean"])
    anoms = get_anomalies()
    high_anoms_ins = list({a["installation_id"] for a in anoms if a["severity"] == "High"})

    plan = [
        f"Inspect installation {inst_sorted[0]['Installation_ID']} — consistently lowest average "
        f"generation ({round(float(inst_sorted[0]['mean']), 4)} kWh/hr). Check for shading, soiling, or wiring issues.",
        "Schedule panel cleaning and maintenance for all sites showing 'Poor' panel condition.",
        f"Investigate {len([a for a in anoms if a['severity'] == 'High'])} high-severity anomalies "
        f"at installations: {', '.join(high_anoms_ins) if high_anoms_ins else 'none detected'}.",
        "Cross-reference low-output periods with weather data before concluding equipment failure.",
        "Compare installations under similar sunlight and temperature conditions to isolate site-specific issues.",
        "Use the XGBoost prediction tool to forecast generation and plan maintenance windows during low-output periods.",
        "Retrain the model quarterly as new operational data becomes available to maintain prediction accuracy.",
    ]
    return jsonify(plan)


if __name__ == "__main__":
    print("Starting Solar Energy API...")
    app.run(debug=True, port=5000)
