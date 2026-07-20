import argparse
import os
import sqlite3
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
SENTIMENT_SIGN = {"positive": 1, "neutral": 0, "negative": -1}

def load_daily_series(conn, appid: int) -> pd.DataFrame:
    query = """
        SELECT
            r.timestamp_created,
            s.sentiment_label,
            s.sentiment_score
        FROM reviews r
        LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
        WHERE r.appid = ?
    """
    df = pd.read_sql_query(query, conn, params=(appid,), parse_dates=["timestamp_created"])
    df["day"] = df["timestamp_created"].dt.date
 
    df["sentiment_signed"] = df["sentiment_label"].map(SENTIMENT_SIGN) * df["sentiment_score"]
 
    daily = df.groupby("day").agg(
        review_count=("timestamp_created", "count"),
        avg_sentiment=("sentiment_signed", "mean"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D")
    daily["review_count"] = daily["review_count"].fillna(0)
    return daily


def print_spike_days(daily: pd.DataFrame, top_n: int = 5):
    top = daily["review_count"].sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} highest-volume days (check patch notes / news around these dates to double check the unusual spike of reviews):")
    for date, count in top.items():
        print(f"  {date.date()}: {int(count)} reviews")
 
 
def check_stationarity(series: pd.Series):
    result = adfuller(series.dropna())
    print(f"\nADF statistic: {result[0]:.3f}, p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("Series is likely stationary.")
    else:
        print("Series is likely non-stationary. ARIMA differencing (d>0) will handle this.")


def decompose_and_plot(series: pd.Series, label: str, filename: str, log_transform: bool = False):
    clean = series.fillna(0)
    if log_transform:
        # Review volume for this dataset has an extreme outlier period (a major game update would case a spike ~1000x the daily baseline).
        # We decompose raw counts. LEtting that handful of days dominate the trend/seasonal components. log1p compresses the scale so the decomposition reflects the whole series, not just the spike.
        clean = np.log1p(clean)
        label = f"log({label})"
 
    if len(clean.dropna()) < 14:
        print(f"Not enough days of data to decompose {label} (need 2+ weeks). Skipping.")
        return
 
    decomposition = seasonal_decompose(clean, model="additive", period=7)
 
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title=f"Observed daily {label}")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Weekly seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
 
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved decomposition plot to {out_path}")
 
 
def fit_forecast(series: pd.Series, label: str, filename: str, horizon: int = 7, log_transform: bool = False):
    clean = series.fillna(0)
    if log_transform:
        clean = np.log1p(clean)
 
    if len(clean) < 21:
        print(f"Not enough history to forecast {label} (need 3+ weeks). Skipping.")
        return
 
    model = ARIMA(clean, order=(2, 1, 2))
    fit = model.fit()
    print(f"\n=== ARIMA summary: {label}{' (fit on log1p scale)' if log_transform else ''} ===")
    print(fit.summary())
 
    forecast = fit.get_forecast(steps=horizon)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
 
    if log_transform:
        # Convert back to the original review-count scale for the plot. The ARIMA math happens in log space, but "log(reviews)" isn't a useful thing to show on a chart. expm1 undoes log1p; clip at 0 since a raw count can't go negative (the lower confidence bound can dip below 0 in log space for a low-volume forecast).
        plot_observed = np.expm1(clean)
        plot_forecast = np.expm1(mean_forecast)
        plot_conf_int = np.expm1(conf_int).clip(lower=0)
    else:
        plot_observed = clean
        plot_forecast = mean_forecast
        plot_conf_int = conf_int
 
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_observed.plot(ax=ax, label="Observed")
    plot_forecast.plot(ax=ax, label="Forecast", style="--")
    ax.fill_between(plot_conf_int.index, plot_conf_int.iloc[:, 0], plot_conf_int.iloc[:, 1], alpha=0.2)
    ax.set_title(f"{horizon}-day {label} forecast")
    ax.legend()
    plt.tight_layout()
 
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved forecast plot to {out_path}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appid", type=int, default=1623730, help="Steam appid to analyze")
    args = parser.parse_args()
 
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    daily = load_daily_series(conn, args.appid)
    conn.close()
 
    print(f"Loaded {len(daily)} days of activity (date range: "
          f"{daily.index.min().date()} to {daily.index.max().date()}).")
 
    print_spike_days(daily)
 
    print("\n--- Review volume ---")
    print("(stationarity check and ARIMA fit use log1p-transformed volume — "
          "see comments in fit_forecast for why)")
    check_stationarity(np.log1p(daily["review_count"]))
    decompose_and_plot(daily["review_count"], "review volume", "volume_decomposition.png", log_transform=True)
    fit_forecast(daily["review_count"], "review volume", "volume_forecast.png", log_transform=True)
 
    print("\n--- Sentiment ---")
    check_stationarity(daily["avg_sentiment"].fillna(0))
    decompose_and_plot(daily["avg_sentiment"], "sentiment", "sentiment_decomposition.png")
    fit_forecast(daily["avg_sentiment"], "sentiment", "sentiment_forecast.png")
 
 
if __name__ == "__main__":
    main()