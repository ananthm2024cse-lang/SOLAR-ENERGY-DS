"""
eda_analysis.py
Performs exploratory data analysis on the cleaned solar energy dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLEANED_PATH = Path(__file__).parent / "data" / "cleaned_solar_energy.csv"
VIZ_DIR      = Path(__file__).parent / "outputs" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Autumn", 10: "Autumn", 11: "Autumn"}


def load_df():
    df = pd.read_csv(CLEANED_PATH, parse_dates=["Timestamp"])
    df["Hour"]        = df["Timestamp"].dt.hour
    df["Day"]         = df["Timestamp"].dt.day
    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
    df["Month"]       = df["Timestamp"].dt.month
    df["Season"]      = df["Month"].map(SEASON_MAP)
    df["Date"]        = df["Timestamp"].dt.date
    return df


def descriptive_stats(df: pd.DataFrame) -> dict:
    gen = df["Solar_Generation"]
    stats = {
        "mean_generation":    round(float(gen.mean()), 4),
        "median_generation":  round(float(gen.median()), 4),
        "min_generation":     round(float(gen.min()), 4),
        "max_generation":     round(float(gen.max()), 4),
        "std_generation":     round(float(gen.std()), 4),
        "total_energy_kwh":   round(float(gen.sum()), 2),
        "avg_sunlight":       round(float(df["Sunlight"].mean()), 2),
        "avg_temperature":    round(float(df["Temperature"].mean()), 2),
        "avg_cloud_cover":    round(float(df["Cloud_Cover"].mean()), 2),
    }
    return stats


def installation_stats(df: pd.DataFrame) -> list[dict]:
    grp = df.groupby("Installation_ID")["Solar_Generation"].agg(
        mean="mean", total="sum", max="max", min="min"
    ).round(4).reset_index()
    return grp.to_dict(orient="records")


def daily_trend(df: pd.DataFrame) -> list[dict]:
    daily = df.groupby("Date")["Solar_Generation"].sum().reset_index()
    daily.columns = ["date", "total_generation"]
    daily["date"] = daily["date"].astype(str)
    return daily.to_dict(orient="records")


def monthly_trend(df: pd.DataFrame) -> list[dict]:
    monthly = df.groupby("Month")["Solar_Generation"].sum().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["Month"].map(month_names)
    monthly.columns = ["month_num", "total_generation", "month"]
    return monthly[["month", "total_generation"]].to_dict(orient="records")


def hourly_pattern(df: pd.DataFrame) -> list[dict]:
    hourly = df.groupby("Hour")["Solar_Generation"].mean().reset_index()
    hourly.columns = ["hour", "avg_generation"]
    return hourly.to_dict(orient="records")


def seasonal_stats(df: pd.DataFrame) -> list[dict]:
    seasonal = df.groupby("Season")["Solar_Generation"].agg(
        avg="mean", total="sum"
    ).round(4).reset_index()
    return seasonal.to_dict(orient="records")


def weather_correlation(df: pd.DataFrame) -> dict:
    cols = ["Solar_Generation", "Sunlight", "Temperature", "Cloud_Cover"]
    corr = df[cols].corr()["Solar_Generation"].drop("Solar_Generation").round(4).to_dict()
    return corr


def panel_condition_stats(df: pd.DataFrame) -> list[dict]:
    grp = df.groupby("Panel_Condition")["Solar_Generation"].agg(
        mean="mean", total="sum", count="count"
    ).round(4).reset_index()
    return grp.to_dict(orient="records")


def save_visualizations(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")

    # 1. Daily generation trend
    daily = df.groupby("Date")["Solar_Generation"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(daily["Date"]), daily["Solar_Generation"], color="#2ecc71", linewidth=1)
    ax.set_title("Daily Solar Generation Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Generation (kWh)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "daily_trend.png", dpi=100)
    plt.close(fig)

    # 2. Installation performance
    inst = df.groupby("Installation_ID")["Solar_Generation"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=inst, x="Installation_ID", y="Solar_Generation",
                palette="Greens_d", ax=ax)
    ax.set_title("Average Generation by Installation")
    ax.set_ylabel("Avg Generation (kWh)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "installation_performance.png", dpi=100)
    plt.close(fig)

    # 3. Sunlight vs Generation scatter
    sample = df[df["Sunlight"] > 0].sample(min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sample["Sunlight"], sample["Solar_Generation"],
               alpha=0.3, color="#3498db", s=10)
    ax.set_title("Sunlight vs Solar Generation")
    ax.set_xlabel("Sunlight (W/m²)"); ax.set_ylabel("Generation (kWh)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "sunlight_vs_generation.png", dpi=100)
    plt.close(fig)

    # 4. Monthly bar chart
    monthly = df.groupby("Month")["Solar_Generation"].sum().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["Month"].apply(lambda x: month_names[x-1])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=monthly, x="month_name", y="Solar_Generation",
                palette="YlOrRd", ax=ax)
    ax.set_title("Monthly Total Solar Generation")
    ax.set_ylabel("Total Generation (kWh)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "monthly_generation.png", dpi=100)
    plt.close(fig)

    print("Visualizations saved.")


def run_eda():
    df = load_df()
    stats = {
        "descriptive":    descriptive_stats(df),
        "installations":  installation_stats(df),
        "daily_trend":    daily_trend(df),
        "monthly_trend":  monthly_trend(df),
        "hourly_pattern": hourly_pattern(df),
        "seasonal":       seasonal_stats(df),
        "weather_corr":   weather_correlation(df),
        "panel_condition":panel_condition_stats(df),
    }
    save_visualizations(df)
    return stats


if __name__ == "__main__":
    results = run_eda()
    print("Descriptive stats:", results["descriptive"])
