import os
import sqlite3
import logging
import warnings

import pandas as pd
import numpy as np

from gdelt_geo_features import load_geo_feature_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("feature_engineering")

DB_NAME = "tsetmc_market_data.db"
FEATURES_DIR = "ai_features_outputs"
LOOKAHEAD_DAYS = 60
GEO_FFILL_LIMIT_ROWS = 10  # max trading rows to carry a geo value forward

# ─── Iranian Market Thresholds ───
MAX_DAILY_RETURN_THRESHOLD = 0.30  # 30% — Returns above this are likely due to capital increases/halts
HALT_GAP_DAYS = 10  # Gaps exceeding this threshold indicate a symbol trading halt
CAPITAL_INCREASE_DROP_THRESHOLD = -0.30  # Drops greater than 30% are likely corporate actions / capital increases
CAPITAL_INCREASE_NEVER_SENTINEL = 3650.0  # Sentinel value indicating no prior capital increase (~10 years)


SAVE_FORMAT = "csv"  # "csv" | "parquet" | "excel"

os.makedirs(FEATURES_DIR, exist_ok=True)


def load_raw_data_from_db(ticker: str) -> pd.DataFrame:
    """Load raw ticker data from SQLite database and handle missing USD exchange rate gaps."""
    conn = sqlite3.connect(DB_NAME)
    try:
        query = """
        SELECT
            CAST(dp.jalali_date AS INTEGER) as jalali_date,
            dp.open_price, dp.high_price, dp.low_price, dp.close_price, dp.volume,
            md.dollar_rate
        FROM daily_prices dp
        JOIN instruments i ON dp.instrument_id = i.id
        LEFT JOIN macro_data md ON CAST(dp.jalali_date AS INTEGER) = CAST(md.jalali_date AS INTEGER)
        WHERE i.ticker = ? AND CAST(dp.jalali_date AS INTEGER) >= 13950101
        ORDER BY jalali_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        return df

    # Check initial USD rate null status
    initial_nan_dollar = df['dollar_rate'].isna().sum()

    if initial_nan_dollar > 0:
        # Fill missing values forward then backward to cover leading dates
        df['dollar_rate'] = df['dollar_rate'].ffill().bfill()
        final_nan_dollar = df['dollar_rate'].isna().sum()

        logger.info(f"🔍 {ticker}: Out of {len(df)} days, {initial_nan_dollar} days lacked dollar rates. "
                    f"After imputation: {final_nan_dollar} NaN days remain.")

        # Handle cases where all macro data entries are missing
        if final_nan_dollar == len(df):
            logger.warning(f"⚠️ {ticker}: USD rate data in macro_data table is entirely missing for this range!")
        elif final_nan_dollar > 0:
            df = df.dropna(subset=['dollar_rate']).reset_index(drop=True)

    return df


def detect_capital_increases(df: pd.DataFrame) -> pd.DataFrame:
    """Detect capital increases based on sharp unadjusted price drops."""
    df = df.sort_values('jalali_date').reset_index(drop=True).copy()
    df['raw_return'] = df['close_price'].pct_change()

    df['is_capital_increase'] = (
        (df['raw_return'] < CAPITAL_INCREASE_DROP_THRESHOLD) &
        (df['raw_return'].notna())
    ).astype(float)

    df['capital_increase_count'] = df['is_capital_increase'].cumsum()

    return df


def add_capital_increase_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lookback temporal features for the model to capture capital increase patterns."""
    df = df.copy()

    if 'days_since_prev_row' in df.columns:
        elapsed_days = df['days_since_prev_row'].fillna(1).cumsum()
    else:
        elapsed_days = pd.Series(np.arange(1, len(df) + 1), index=df.index).astype(float)

    increase_marker = elapsed_days.where(df['is_capital_increase'] == 1)
    last_increase_elapsed = increase_marker.ffill()

    days_since = elapsed_days - last_increase_elapsed
    df['days_since_last_capital_increase'] = (
        days_since.fillna(CAPITAL_INCREASE_NEVER_SENTINEL)
                  .clip(upper=CAPITAL_INCREASE_NEVER_SENTINEL)
    )

    df['capital_increase_freq_252d'] = df['is_capital_increase'].rolling(
        window=252, min_periods=1
    ).sum()

    return df


def calculate_adjusted_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns adjusted for capital increase gaps and normalized trading halts."""
    df['stock_return_raw'] = np.log(df['close_price'] / df['close_price'].shift(1))

    df['stock_return'] = df['stock_return_raw'].where(
        df['is_capital_increase'] == 0, 0.0
    )

    if 'days_since_prev_row' in df.columns:
        normalization_factor = df['days_since_prev_row'].clip(lower=1, upper=30)
        mask = (df['is_capital_increase'] == 0) & (df['days_since_prev_row'] > 1)
        df.loc[mask, 'stock_return'] = df.loc[mask, 'stock_return'] / normalization_factor[mask]

    outlier_mask = df['stock_return'].abs() > MAX_DAILY_RETURN_THRESHOLD
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        logger.warning(f"   ⚠️ {outlier_count} rows with returns > {MAX_DAILY_RETURN_THRESHOLD*100:.0f}% "
                       f"were reset to 0 (likely unflagged capital increases).")
        df.loc[outlier_mask, 'stock_return'] = 0.0

    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pure price technical indicators independent of macroeconomic factors."""
    delta = df['close_price'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss.replace(0, np.nan) + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].clip(0, 100)

    exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
    exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = (exp1 - exp2) / (df['close_price'] + 1e-8)
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    high_low = df['high_price'] - df['low_price']
    high_close_prev = (df['high_price'] - df['close_price'].shift(1)).abs()
    low_close_prev = (df['low_price'] - df['close_price'].shift(1)).abs()
    df['true_range'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    df['atr_14'] = (
        df['true_range'].rolling(window=14, min_periods=14).mean() / (df['close_price'] + 1e-8)
    )

    rolling_max = df['close_price'].rolling(window=20, min_periods=1).max()
    df['drawdown_20d'] = (df['close_price'] - rolling_max) / rolling_max

    ma20 = df['close_price'].rolling(window=20, min_periods=20).mean()
    ma50 = df['close_price'].rolling(window=50, min_periods=50).mean()
    df['dist_ma20'] = (df['close_price'] - ma20) / ma20
    df['dist_ma50'] = (df['close_price'] - ma50) / ma50

    return df


def _approx_calendar_days(jalali_int_series: pd.Series) -> pd.Series:
    """Estimate elapsed calendar days between consecutive Jalali integer dates."""
    s = jalali_int_series.astype('Int64').astype(str).str.zfill(8)
    year = s.str[:4].astype(float)
    month = s.str[4:6].astype(float)
    day = s.str[6:8].astype(float)
    return year * 365 + month * 30 + day


def calculate_ai_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Engineer features with diagnostic monitoring for missing values."""
    min_required = 60 + LOOKAHEAD_DAYS
    if df.empty or len(df) < min_required:
        logger.warning(f"{ticker}: Only {len(df)} rows (< {min_required}); skipped.")
        return pd.DataFrame()

    features_df = df.copy()

    approx_day_num = _approx_calendar_days(features_df['jalali_date'])
    features_df['days_since_prev_row'] = approx_day_num.diff().clip(lower=1)
    features_df['is_post_halt_reopen'] = (
        features_df['days_since_prev_row'] > HALT_GAP_DAYS
    ).astype(float)

    features_df = detect_capital_increases(features_df)
    features_df = calculate_adjusted_returns(features_df)
    features_df = add_capital_increase_recency_features(features_df)

    # Adjusted price index based on compounded return series
    features_df['adjusted_close_price'] = features_df['close_price'].iloc[0] * np.exp(
        features_df['stock_return'].fillna(0.0).cumsum()
    )

    # ─── USD-dependent features (Zero-variance robust implementation) ───
    has_valid_dollar = features_df['dollar_rate'].notna().sum() > 20

    if has_valid_dollar:
        features_df['dollar_return'] = np.log(
            features_df['dollar_rate'] / features_df['dollar_rate'].shift(1)
        ).fillna(0.0)

        features_df['volatility_20d'] = features_df['stock_return'].rolling(window=20, min_periods=20).std()
        dollar_volt_20d = features_df['dollar_return'].rolling(window=20, min_periods=20).std()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            raw_corr = features_df['stock_return'].rolling(window=20, min_periods=20).corr(features_df['dollar_return'])
            features_df['dollar_corr_20d'] = np.where(dollar_volt_20d == 0, 0.0, raw_corr)

            cov = features_df['stock_return'].rolling(20, min_periods=20).cov(features_df['dollar_return'])
            var = features_df['dollar_return'].rolling(20, min_periods=20).var()

            features_df['beta_proxy'] = np.where(var == 0, 0.0, cov / (var + 1e-8))

        dollar_chg_20d = features_df['dollar_rate'].pct_change(20)
        features_df['market_regime_dollar'] = np.where(dollar_chg_20d.isna(), np.nan, (dollar_chg_20d > 0.05).astype(float))

        RELATIVE_PERF_WINDOW = 60
        cum_stock_ret = features_df['stock_return'].rolling(
            window=RELATIVE_PERF_WINDOW, min_periods=RELATIVE_PERF_WINDOW
        ).sum()
        cum_dollar_ret = features_df['dollar_return'].rolling(
            window=RELATIVE_PERF_WINDOW, min_periods=RELATIVE_PERF_WINDOW
        ).sum()
        features_df['relative_dollar_value'] = (cum_stock_ret - cum_dollar_ret).fillna(0.0)
    else:
        logger.warning(f"❌ {ticker}: USD features fallback to default values due to missing data.")
        features_df['dollar_return'] = 0.0
        features_df['volatility_20d'] = features_df['stock_return'].rolling(window=20, min_periods=20).std()
        features_df['dollar_corr_20d'] = 0.0
        features_df['beta_proxy'] = 0.0
        features_df['market_regime_dollar'] = 0.0
        features_df['relative_dollar_value'] = 0.0

    features_df['dollar_corr_20d'] = features_df['dollar_corr_20d'].fillna(0.0)
    features_df['beta_proxy'] = features_df['beta_proxy'].fillna(0.0)
    rolling_volume_avg = features_df['volume'].rolling(window=20, min_periods=20).mean()
    features_df['volume_ratio'] = features_df['volume'] / (rolling_volume_avg + 1e-8)

    features_df['is_locked_queue'] = np.where(
        (features_df['high_price'] == features_df['low_price']) & (features_df['stock_return'].abs() > 1e-6), 1.0, 0.0
    )
    features_df['queue_persistence_5d'] = features_df['is_locked_queue'].rolling(window=5, min_periods=5).sum()

    features_df = calculate_technical_indicators(features_df)

    # ─── Merge geopolitical signals (WorldMonitor) ───
    geo_history = load_geo_feature_history()
    geo_cols = ['geo_cii_score', 'geo_conflict_event_count_7d', 'geo_high_risk_flag']
    if not geo_history.empty:
        geo_history = geo_history.copy()
        geo_history['jalali_date'] = geo_history['jalali_date'].astype('int64')
        features_df = features_df.merge(geo_history, on='jalali_date', how='left')
        # Carry the last known value across short gaps / the newest trading days, instead of
        # writing 0.0 into live rows (0 is outside the training distribution of geo_cii_score).
        present = [c for c in geo_cols if c in features_df.columns]
        features_df[present] = features_df[present].ffill(limit=GEO_FFILL_LIMIT_ROWS)
    for col in geo_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
        else:
            features_df[col] = features_df[col].fillna(0.0)

    # ─── Target Calculation & Labeling ───
    features_df['future_stock_return_60d'] = (
        features_df['adjusted_close_price'].shift(-LOOKAHEAD_DAYS) - features_df['adjusted_close_price']
    ) / features_df['adjusted_close_price']

    if has_valid_dollar and features_df['dollar_rate'].shift(-LOOKAHEAD_DAYS).notna().sum() > 0:
        features_df['future_market_return_60d'] = (
            features_df['dollar_rate'].shift(-LOOKAHEAD_DAYS) - features_df['dollar_rate']
        ) / features_df['dollar_rate']

        conditions = [
            (features_df['future_stock_return_60d'] > features_df['future_market_return_60d'] + 0.05),
            (features_df['future_stock_return_60d'] > features_df['future_market_return_60d']),
        ]
    else:
        features_df['future_market_return_60d'] = 0.05
        conditions = [
            (features_df['future_stock_return_60d'] > 0.15),
            (features_df['future_stock_return_60d'] > 0.05),
        ]

    choices = [2, 1]
    has_future = features_df['future_stock_return_60d'].notna()
    raw_label = np.select(conditions, choices, default=0)
    features_df['label'] = np.where(has_future, raw_label, np.nan)

    # ─── Diagnostics System ───
    required_cols = [
        'rsi_14', 'dist_ma50', 'volatility_20d',
        'dollar_corr_20d', 'beta_proxy', 'market_regime_dollar'
    ]

    print(f"\n📊 [Diagnostics Matrix] Ticker: {ticker}")
    for col in required_cols:
        if col in features_df.columns:
            nan_count = features_df[col].isna().sum()
            print(f"-> {col}: NaN = {nan_count} / {len(features_df)}")
            if nan_count == len(features_df):
                logger.warning(f"🚨 {ticker}: Column {col} is entirely NaN and will cause all rows to be dropped!")

    features_df = features_df.dropna(subset=required_cols).reset_index(drop=True)

    kept_rows = features_df['label'].notna().sum() if 'label' in features_df.columns else 0
    logger.info(f"   {ticker}: Extracted and saved {kept_rows} valid labeled rows.")

    return features_df


def save_features(df: pd.DataFrame, ticker: str):
    """Save processed AI features to disk using the configured format."""
    if SAVE_FORMAT == "csv":
        file_path = os.path.join(FEATURES_DIR, f"{ticker}_AI_Features.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif SAVE_FORMAT == "parquet":
        file_path = os.path.join(FEATURES_DIR, f"{ticker}_AI_Features.parquet")
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
    else:
        file_path = os.path.join(FEATURES_DIR, f"{ticker}_AI_Features.xlsx")
        df.to_excel(file_path, index=False)
    return file_path


def run_feature_engineering_pipeline():
    """Run the end-to-end feature engineering pipeline for all database tickers."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM instruments")
        tickers = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

    if not tickers:
        raise RuntimeError("No tickers found in the database.")

    logger.info(f"🚀 Starting optimized feature engineering pipeline for {len(tickers)} tickers (output: {SAVE_FORMAT})...")

    succeeded, skipped = [], []
    for ticker in tickers:
        try:
            raw_df = load_raw_data_from_db(ticker)
            processed_df = calculate_ai_features(raw_df, ticker)

            if not processed_df.empty and len(processed_df) > 0:
                file_path = save_features(processed_df, ticker)
                logger.info(f"✅ {ticker}: {processed_df.shape} → {os.path.basename(file_path)}")
                succeeded.append(ticker)
            else:
                skipped.append(ticker)
        except Exception as e:
            logger.error(f"❌ Unexpected error for ticker {ticker}: {e}", exc_info=True)
            skipped.append(ticker)

    logger.info(f"🏁 Pipeline finished: {len(succeeded)} tickers succeeded, {len(skipped)} tickers skipped.")


if __name__ == "__main__":
    run_feature_engineering_pipeline()