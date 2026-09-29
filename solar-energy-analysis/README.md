# ☀️ Solar Energy — Which Days Will Produce Less Power?

A complete Data Science and Machine Learning project that analyzes solar power generation across multiple installations, detects anomalies, and predicts hourly generation using XGBoost.

---

## Problem Statement

Solar energy companies need to monitor power generation continuously, identify underperforming installations, detect abnormal output drops, and plan maintenance proactively. This project provides a data-driven dashboard to support those goals.

## Objectives

1. Analyze daily and seasonal solar generation patterns.
2. Compare performance across multiple installations.
3. Investigate the relationship between weather conditions and generation.
4. Detect abnormal generation drops automatically.
5. Predict hourly solar generation using XGBoost.
6. Present all findings in an interactive web dashboard.

---

## Technologies Used

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Frontend    | HTML5, CSS3, JavaScript, Chart.js        |
| Backend     | Python, Flask, Flask-CORS                |
| ML          | XGBoost, Scikit-learn                    |
| Data        | Pandas, NumPy                            |
| Viz (EDA)   | Matplotlib, Seaborn                      |
| Storage     | CSV, JSON                                |

---

## Dataset Description

**Note: This project uses synthetic sample data generated to simulate realistic solar energy operations.**

| Column           | Description                                      |
|------------------|--------------------------------------------------|
| Timestamp        | Hourly datetime of the record                    |
| Installation_ID  | Unique site identifier (INS_A … INS_E)           |
| Solar_Generation | Power generated in kWh                           |
| Sunlight         | Sunlight intensity in W/m²                       |
| Temperature      | Ambient temperature in °C                        |
| Cloud_Cover      | Cloud coverage percentage (0–100)                |
| Panel_Condition  | Panel state: Good / Moderate / Poor              |

The dataset covers 5 installations over a full year (2023) at hourly resolution (~43,800 base records). Synthetic anomalies, missing values, duplicates, and invalid entries are injected for realistic data-cleaning exercises.

---

## Project Architecture

```
solar-energy-analysis/
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── generate_dataset.py     # Synthetic data generator
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── eda_analysis.py         # EDA & statistics
│   ├── feature_engineering.py  # Feature creation & encoding
│   ├── train_model.py          # XGBoost training
│   ├── evaluate_model.py       # Metrics & feature importance
│   ├── anomaly_detection.py    # Anomaly detection
│   ├── requirements.txt
│   ├── models/
│   │   ├── xgboost_model.json
│   │   └── encoders.json
│   ├── outputs/
│   │   ├── model_metrics.json
│   │   ├── feature_importance.json
│   │   ├── actual_vs_predicted.json
│   │   └── visualizations/
│   └── data/
│       ├── solar_energy.csv
│       └── cleaned_solar_energy.csv
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Data Cleaning Process

1. Load raw CSV with Pandas.
2. Remove duplicate rows.
3. Parse Timestamp to datetime; drop unparseable rows.
4. Set negative Solar_Generation values to NaN.
5. Set Cloud_Cover values outside 0–100 to NaN.
6. Set negative Sunlight values to NaN.
7. Fill missing numeric values using per-installation median (global median as fallback).
8. Fill missing Panel_Condition with "Good".
9. Sort chronologically.
10. Save cleaned dataset.

---

## Exploratory Data Analysis

- Descriptive statistics: mean, median, min, max, std, total energy.
- Per-installation generation statistics.
- Daily, monthly, and seasonal generation trends.
- Hourly generation pattern (bell curve peaking at noon).
- Correlation analysis: Sunlight, Temperature, Cloud Cover vs Generation.
- Panel condition impact on generation.

---

## XGBoost Algorithm

XGBoost (Extreme Gradient Boosting) is an ensemble tree-based algorithm that builds decision trees sequentially, each correcting the errors of the previous one. It is well-suited for tabular regression tasks and handles mixed feature types efficiently.

**Input features:** Installation_ID, Sunlight, Temperature, Cloud_Cover, Panel_Condition, Hour, Day_of_Week, Month, Season

**Target:** Solar_Generation (kWh)

**Split strategy:** Chronological 80/20 split to simulate real-world forecasting (no data leakage).

---

## Model Evaluation Metrics

| Metric | Description                                      |
|--------|--------------------------------------------------|
| MAE    | Mean Absolute Error — average prediction error  |
| MSE    | Mean Squared Error — penalises large errors      |
| RMSE   | Root MSE — same unit as target (kWh)             |
| R²     | Proportion of variance explained (1.0 = perfect) |

---

## Anomaly Detection Methodology

1. Load cleaned data and generate model predictions for all daytime records (Sunlight ≥ 50 W/m²).
2. Compute drop percentage: `1 - (actual / predicted)`.
3. Flag records where drop ≥ 45% as anomalies.
4. Classify severity: High (>80% drop), Medium (>60%), Low (>45%).
5. Assign a probable reason based on Panel_Condition and Cloud_Cover.

---

## Dashboard Features

- Summary cards: total energy, avg generation, installations, anomalies, R².
- Daily, monthly, hourly, and seasonal generation charts.
- Installation performance bar chart and table.
- Weather impact scatter plots (Sunlight, Temperature, Cloud Cover).
- Panel condition comparison chart.
- Anomaly monitoring table with severity badges.
- XGBoost metrics cards and actual vs predicted chart.
- Feature importance horizontal bar chart.
- Interactive prediction form.
- Data quality report.
- Key insights and action plan.

---

## Installation & Execution

### Prerequisites
- Python 3.10+
- A modern web browser

### Step 1 — Install dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2 — Generate the dataset
```bash
python backend/generate_dataset.py
```

### Step 3 — Clean the data
```bash
python backend/data_preprocessing.py
```

### Step 4 — Train the model
```bash
python backend/train_model.py
```

### Step 5 — Evaluate the model
```bash
python backend/evaluate_model.py
```

### Step 6 — Start the backend
```bash
python backend/app.py
```

### Step 7 — Open the dashboard
Open `frontend/index.html` in your browser.
*(Or serve with `python -m http.server 8080` inside the `frontend/` folder.)*

---

## Expected Output

- Cleaned dataset with ~43,000+ records.
- XGBoost model with R² > 0.90 on the test set.
- Anomaly report with severity classification.
- Interactive dashboard with 10+ charts.
- Prediction tool returning kWh estimates in real time.

---

## Key Insights (generated from data)

Insights are dynamically generated from actual dataset values and displayed on the dashboard under the Insights section.

---

## Practical Action Plan

Recommendations are generated from the analysis and displayed on the dashboard under the Action Plan section.

---

*This project uses synthetic sample data and is intended for educational and demonstration purposes.*
