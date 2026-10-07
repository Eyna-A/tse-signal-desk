
import os
import glob
import logging

import pandas as pd
import numpy as np
import lightgbm as lgb

from train_model import FEATURE_COLS, apply_diagnostic_corrections  # Single source of truth to avoid train/serve skew
from gdelt_geo_features import record_daily_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("live_predictor")

FEATURES_DIR = "ai_features_outputs"
MODEL_DIR = "ai_models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgb_robo_advisor.txt")
OUTPUT_PATH = "excel_outputs/live_market_predictions.xlsx"

STALE_DATA_THRESHOLD_DAYS = 15
MIN_HISTORY_ROWS_FOR_STABLE_FEATURES = 20


def _jalali_approx_ordinal(date_int) -> float:
    """Approximate conversion of a numeric Jalali date (YYYYMMDD) into an ascending, comparable scalar."""
    date_int = int(date_int)
    year, month, day = date_int // 10000, (date_int // 100) % 100, date_int % 100
    return year * 360 + month * 30 + day


def load_latest_live_data():
    """
    Loads the latest unlabeled rows for each ticker (live data for tomorrow).
    Prefers CSV files, falling back to Parquet for legacy files.
    """
    csv_files = glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.csv"))
    parquet_files = glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.parquet"))
    all_files = csv_files or parquet_files

    if not all_files:
        raise FileNotFoundError(" No feature files found. Please run feature_engineering.py first.")

    live_rows = []
    for f in all_files:
        df = pd.read_csv(f) if f.endswith('.csv') else pd.read_parquet(f)
        ticker = os.path.basename(f).split('_')[0]
        df['ticker_code'] = ticker

        if len(df) < MIN_HISTORY_ROWS_FOR_STABLE_FEATURES:
            logger.warning(
                f" {ticker}: Only {len(df)} historical rows available (< "
                f"{MIN_HISTORY_ROWS_FOR_STABLE_FEATURES}); stationary rolling features "
                "will be filled with neutral values."
            )
        df = apply_diagnostic_corrections(df)

        live_mask = df['label'].isna()
        live_df = df[live_mask]

        if live_df.empty:
            continue

        latest_day = live_df.sort_values(by='jalali_date').iloc[[-1]].copy()
        latest_day['ticker_code'] = ticker

        for col in FEATURE_COLS:
            if col not in latest_day.columns or latest_day[col].isna().any():
                latest_day[col] = 0.0

        live_rows.append(latest_day)

    if not live_rows:
        raise ValueError(" No valid live data found for prediction!")

    return pd.concat(live_rows, ignore_index=True)


def generate_live_predictions(output_path=OUTPUT_PATH):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f" Model not found at '{MODEL_PATH}'! Run train_model.py first.")

    record_daily_snapshot()

    logger.info(" Loading LightGBM model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    logger.info(" Extracting latest market state...")
    live_df = load_latest_live_data()

    X_live = live_df[FEATURE_COLS]
    preds_proba = model.predict(X_live)

    days_gap = live_df['days_since_prev_row'].values if 'days_since_prev_row' in live_df.columns \
        else np.ones(len(live_df))

    results = pd.DataFrame({
        'Ticker': live_df['ticker_code'].values,
        'Latest Date': live_df['jalali_date'].values,
        'Close Price': live_df['close_price'].astype(int).values,
        'Return (%)': np.round((np.expm1(live_df['stock_return'].values)) * 100, 2),
        'Days Since Prev Trade': days_gap.astype(int),
        'Loss/Lag Prob (Class 0)': np.round(preds_proba[:, 0] * 100, 1),
        'Neutral/Market Prob (Class 1)': np.round(preds_proba[:, 1] * 100, 1),
        'Sharp Growth > 5% Prob (Class 2)': np.round(preds_proba[:, 2] * 100, 1),
    })

    halted = results[results['Days Since Prev Trade'] > 10]
    if not halted.empty:
        logger.warning(f" The following tickers reopened after a trading halt; 'Return (%)' "
                       f"reflects multi-day cumulative return, not single-day return: "
                       f"{halted['Ticker'].tolist()}")

    most_recent_ordinal = results['Latest Date'].apply(_jalali_approx_ordinal).max()
    staleness_days = most_recent_ordinal - results['Latest Date'].apply(_jalali_approx_ordinal)
    is_stale = staleness_days > STALE_DATA_THRESHOLD_DAYS
    results['Is Stale / Suspected Halt'] = is_stale

    if is_stale.any():
        stale_tickers = results.loc[is_stale, 'Ticker'].tolist()
        logger.warning(f" The following tickers have data older than {STALE_DATA_THRESHOLD_DAYS} days "
                       f"(likely halted/suspended) and were excluded from buy signals: {stale_tickers}")
        preds_proba[is_stale.to_numpy(), 2] = 0.0
        preds_proba[is_stale.to_numpy(), 0] = 1.0
        results.loc[is_stale, 'Loss/Lag Prob (Class 0)'] = 100.0
        results.loc[is_stale, 'Sharp Growth > 5% Prob (Class 2)'] = 0.0

    results['Alpha Score'] = (
        results['Sharp Growth > 5% Prob (Class 2)'] - results['Loss/Lag Prob (Class 0)']
    )

    best_class = np.argmax(preds_proba, axis=1)
    class_map = {0: " No Buy / Sell", 1: " Hold / Market Performing", 2: " Buy Signal (Strong Growth)"}
    results['System Signal'] = [class_map[c] for c in best_class]
    results.loc[is_stale, 'System Signal'] = " Stale Data / Likely Halted - Do Not Trade"

    ranking_table = results.sort_values(by='Alpha Score', ascending=False).reset_index(drop=True)

    logger.info(f"\n Ticker Ranking:\n{ranking_table.to_string(index=False)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranking_table.to_excel(output_path, index=False)
    logger.info(f" Output saved to '{output_path}'.")

    return ranking_table


if __name__ == "__main__":
    generate_live_predictions()