
import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("portfolio_optimizer")

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "excel_outputs" / "live_market_predictions.xlsx"
DB_NAME = str(BASE_DIR / "tsetmc_market_data.db")
TRADING_DAYS_PER_YEAR = 242
RISK_FREE_ANNUAL = 0.28
ALPHA_TO_RETURN_SCALE = 0.40
MIN_VALID_ALPHA = -20
MAX_SINGLE_STOCK_WEIGHT = 0.35
MIN_ALLOCATION_TOMAN = 100_000
COV_SHRINKAGE = 1e-4
MIN_COV_PERIODS = 30
MAX_CORRELATION = 0.85
MAX_PORTFOLIO_SIZE = 15
MIN_VALID_JALALI_DATE = 13950101

ALPHA_HORIZON_DAYS = 60
ANNUALIZATION_FACTOR = TRADING_DAYS_PER_YEAR / ALPHA_HORIZON_DAYS


def get_historical_returns_batch(tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)
    try:
        placeholders = ",".join(["?"] * len(tickers))
        query = f"""
        SELECT i.ticker, dp.jalali_date, dp.close_price
        FROM daily_prices dp
        JOIN instruments i ON dp.instrument_id = i.id
        WHERE i.ticker IN ({placeholders})
          AND dp.jalali_date >= ?
        ORDER BY i.ticker, dp.jalali_date ASC
        """
        params = tuple(tickers) + (MIN_VALID_JALALI_DATE,)
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    ticker_counts = df.groupby('ticker').size()
    valid_tickers = ticker_counts[ticker_counts > 50].index.tolist()
    df = df[df['ticker'].isin(valid_tickers)].copy()

    df = df.sort_values(['ticker', 'jalali_date'])
    df['daily_return'] = df.groupby('ticker')['close_price'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    returns_df = df.pivot(index='jalali_date', columns='ticker', values='daily_return')

    if len(returns_df) > lookback_days:
        returns_df = returns_df.tail(lookback_days)

    dropped = set(tickers) - set(valid_tickers)
    if dropped:
        logger.warning(f"⚠️ Dropped tickers with insufficient data: {sorted(dropped)}")

    return returns_df


def filter_high_correlation(returns_df: pd.DataFrame,
                            max_corr: float = MAX_CORRELATION) -> pd.DataFrame:
    if returns_df.shape[1] < 2:
        return returns_df

    corr_matrix = returns_df.corr(min_periods=MIN_COV_PERIODS).abs()

    to_drop = set()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper_tri.columns:
        if col in to_drop:
            continue
        highly_correlated = upper_tri.index[upper_tri[col] > max_corr].tolist()
        for hc in highly_correlated:
            if hc not in to_drop:
                var_col = returns_df[col].var()
                var_hc = returns_df[hc].var()
                to_drop.add(hc if var_hc >= var_col else col)

    if to_drop:
        logger.info(f"🔗 High correlation tickers dropped (corr > {max_corr}): {sorted(to_drop)}")
        returns_df = returns_df.drop(columns=list(to_drop))

    return returns_df


def optimize_portfolio(capital: float, risk_appetite: str, time_horizon: str):
    """Optimal allocation engine incorporating unit-mismatch corrections."""

    if capital <= 0:
        raise ValueError("Capital must be positive.")
    if risk_appetite not in ('low', 'medium', 'high'):
        raise ValueError("risk_appetite must be 'low', 'medium', or 'high'.")
    if time_horizon not in ('short', 'mid', 'long'):
        raise ValueError("time_horizon must be 'short', 'mid', or 'long'.")

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError("❌ Predictions file not found.")

    pred_df = pd.read_excel(PREDICTIONS_PATH)
    required_cols = {'Alpha Score', 'Ticker', 'System Signal'}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if 'Is Stale / Suspected Halt' in pred_df.columns:
        n_stale = int(pred_df['Is Stale / Suspected Halt'].sum())
        if n_stale > 0:
            logger.warning(f"⛔ {n_stale} tickers suspected of being halted were excluded from portfolio optimization.")
            pred_df = pred_df[~pred_df['Is Stale / Suspected Halt']].copy()

    valid_stocks = pred_df[pred_df['Alpha Score'] > MIN_VALID_ALPHA].copy()
    if valid_stocks.empty:
        logger.warning("⚠️ All tickers carry loss/downside risk. Entire capital allocated to Fixed Income.")
        return {"Fixed Income Fund": capital}

    tickers = valid_stocks['Ticker'].tolist()

    returns_df = get_historical_returns_batch(tickers)

    if returns_df.empty:
        logger.warning("⚠️ No ticker has sufficient historical data. Entire capital allocated to Fixed Income.")
        return {"Fixed Income Fund": capital}

    returns_df = filter_high_correlation(returns_df)

    cov_matrix_raw = returns_df.cov(min_periods=MIN_COV_PERIODS) * TRADING_DAYS_PER_YEAR

    valid_variance_mask = cov_matrix_raw.to_numpy().diagonal()
    usable_tickers = [
        t for t, v in zip(cov_matrix_raw.columns, valid_variance_mask)
        if pd.notna(v) and v > 0
    ]
    dropped = set(cov_matrix_raw.columns) - set(usable_tickers)
    if dropped:
        logger.warning(f"⚠️ Tickers without valid variance dropped: {sorted(dropped)}")

    if not usable_tickers:
        logger.warning("⚠️ No valid tickers remaining. Entire capital allocated to Fixed Income.")
        return {"Fixed Income Fund": capital}

    if len(usable_tickers) > MAX_PORTFOLIO_SIZE:
        alpha_scores = valid_stocks.set_index('Ticker')['Alpha Score']
        usable_tickers = sorted(
            usable_tickers,
            key=lambda t: alpha_scores.get(t, -999),
            reverse=True
        )[:MAX_PORTFOLIO_SIZE]
        logger.info(f"📊 Constraining portfolio to top {MAX_PORTFOLIO_SIZE} stocks by Alpha Score.")

    cov_matrix = cov_matrix_raw.loc[usable_tickers, usable_tickers]

    if cov_matrix.isna().any().any():
        logger.warning("⚠️ NaN values found in covariance matrix; filled with row means.")
        cov_matrix = cov_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
        cov_matrix = cov_matrix.fillna(0.0)

    tickers = usable_tickers
    valid_stocks = valid_stocks[valid_stocks['Ticker'].isin(tickers)].set_index('Ticker')

    expected_returns_horizon = (
        valid_stocks.loc[tickers, 'Alpha Score'] / 100.0
    ) * ALPHA_TO_RETURN_SCALE
    expected_returns = expected_returns_horizon * ANNUALIZATION_FACTOR

    logger.info(
        f"🩺 Expected returns (annualized, factor={ANNUALIZATION_FACTOR:.2f}): "
        f"{expected_returns.round(3).to_dict()}"
    )
    if (expected_returns < RISK_FREE_ANNUAL).all():
        logger.warning(
            "⚠️ Even after annualization, expected returns for all stocks remain below RISK_FREE_ANNUAL "
            f"({RISK_FREE_ANNUAL:.0%}) -- the optimizer will lean toward minimum variance "
            "rather than chasing alpha. Calibrate ALPHA_TO_RETURN_SCALE or RISK_FREE_ANNUAL "
            "based on observed returns."
        )

    cov_matrix = cov_matrix + np.eye(len(tickers)) * COV_SHRINKAGE

    try:
        cond_number = np.linalg.cond(cov_matrix.to_numpy())
        if cond_number > 1e10:
            logger.warning(f"⚠️ Covariance matrix is ill-conditioned (cond={cond_number:.2e}). "
                           "Results may be unstable.")
    except np.linalg.LinAlgError:
        pass

    max_equity_ratio = {'low': 0.30, 'medium': 0.65, 'high': 0.95}[risk_appetite]
    if time_horizon == 'short':
        max_equity_ratio = min(max_equity_ratio, 0.40)

    from gdelt_geo_features import get_current_risk_brake
    max_equity_ratio = get_current_risk_brake(max_equity_ratio)

    num_assets = len(tickers)

    def portfolio_variance(w):
        return float(w @ cov_matrix @ w)

    def negative_sharpe(w):
        p_ret = float(expected_returns.values @ w)
        p_var = portfolio_variance(w)
        return -(p_ret - RISK_FREE_ANNUAL) / np.sqrt(max(p_var, 1e-12))

    if num_assets == 1:
        raw_weights = np.array([1.0])
    else:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bounds = tuple((0.0, MAX_SINGLE_STOCK_WEIGHT) for _ in range(num_assets))
        init_weights = np.array([1.0 / num_assets] * num_assets)

        optimized = minimize(
            negative_sharpe, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        if not optimized.success:
            logger.warning(f"⚠️ Sharpe optimization failed to converge ({optimized.message}); falling back to minimum variance.")
            optimized = minimize(
                portfolio_variance, init_weights, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            if not optimized.success:
                logger.warning("⚠️ Minimum variance optimization failed to converge; falling back to equal weighting.")
                raw_weights = init_weights
            else:
                raw_weights = optimized.x
        else:
            raw_weights = optimized.x

    equity_capital = capital * max_equity_ratio
    fixed_income_capital = capital - equity_capital

    logger.info(f"🎯 Macro Allocation: Equity {max_equity_ratio*100:.0f}% "
                f"({equity_capital:,.0f} Toman) | Fixed Income {(1-max_equity_ratio)*100:.0f}%")

    portfolio_allocation = []
    weights_fraction = {}
    for i, ticker in enumerate(tickers):
        allocated = raw_weights[i] * equity_capital
        if allocated > MIN_ALLOCATION_TOMAN:
            portfolio_allocation.append({
                'Ticker': ticker,
                'Equity Segment Weight': f"{raw_weights[i]*100:.1f}%",
                'Total Portfolio Weight': f"{(allocated/capital)*100:.1f}%",
                'Allocated Amount (Toman)': f"{allocated:,.0f}",
                'Model Signal': valid_stocks.loc[ticker, 'System Signal'],
            })
            weights_fraction[ticker] = allocated / capital

    portfolio_allocation.append({
        'Ticker': 'Fixed Income Fund',
        'Equity Segment Weight': '0.0%',
        'Total Portfolio Weight': f"{(fixed_income_capital/capital)*100:.1f}%",
        'Allocated Amount (Toman)': f"{fixed_income_capital:,.0f}",
        'Model Signal': '⚖️ Risk Management',
    })
    weights_fraction['Fixed Income Fund'] = fixed_income_capital / capital

    allocation_df = pd.DataFrame(portfolio_allocation)
    logger.info(f"\n📝 Recommended Portfolio:\n{allocation_df.to_string(index=False)}")

    os.makedirs(BASE_DIR / "excel_outputs", exist_ok=True)
    allocation_df.to_excel(BASE_DIR / "excel_outputs" / "optimized_portfolio_suggestion.xlsx", index=False)

    import json
    with open(BASE_DIR / "excel_outputs" / "portfolio_allocation.json", "w", encoding="utf-8") as f:
        json.dump(weights_fraction, f, ensure_ascii=False, indent=2)

    return allocation_df


if __name__ == "__main__":
    optimize_portfolio(50_000_000, 'medium', 'mid')