import os
import glob
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb

from train_model import FEATURE_COLS, apply_diagnostic_corrections

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("backtester")

FEATURES_DIR = "ai_features_outputs"
MODEL_DIR = "ai_models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgb_robo_advisor.txt")

EULER_MASCHERONI = 0.5772156649015329

SELL_FEE_RATE = 0.009
BUY_FEE_RATE = 0.0035
SLIPPAGE_RATE = 0.002
RISK_FREE_ANNUAL = 0.28
TRADING_DAYS_PER_YEAR = 242
TOP_N_BUYS = 5
ALPHA_BUY_THRESHOLD = 0.05
MIN_ALLOCATION_TOMAN = 100_000

MIN_ADJUSTMENT_DENOMINATOR = 0.01


def load_all_processed_data() -> pd.DataFrame:
    """
    Loads feature dataset primarily from CSV files, falling back to Parquet or Excel formats.
    """
    csv_files = glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.csv"))
    parquet_files = glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.parquet"))
    excel_files = glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.xlsx"))

    all_files = csv_files or parquet_files or excel_files
    if not all_files:
        raise FileNotFoundError("❌ No feature files found.")

    combined_list = []
    for f in all_files:
        if f.endswith('.csv'):
            df = pd.read_csv(f)
        elif f.endswith('.parquet'):
            df = pd.read_parquet(f)
        else:
            df = pd.read_excel(f)

        df['ticker_code'] = os.path.basename(f).split('_')[0]

        if 'relative_dollar_value' not in df.columns:
            raise RuntimeError(
                f"❌ File '{f}' is missing the stationary column 'relative_dollar_value'. "
                "Rebuild the entire ai_features_outputs directory using the updated feature_engineering.py."
            )

        required_ci_cols = {'is_capital_increase', 'raw_return'}
        missing_ci = required_ci_cols - set(df.columns)
        if missing_ci:
            raise RuntimeError(
                f"❌ File '{f}' is missing required columns {missing_ci} needed for stock adjustment "
                "on capital increase dates. Rebuild the ai_features_outputs directory using feature_engineering.py."
            )

        combined_list.append(df)

    full_df = pd.concat(combined_list, ignore_index=True)
    return full_df.sort_values(['jalali_date', 'ticker_code']).reset_index(drop=True)


def _expected_max_sharpe_under_null(n_trials: int, sharpe_variance: float) -> float:
    """
    Calculates the expected maximum Sharpe ratio under the null hypothesis using the Euler-Mascheroni approximation.
    """
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    z_a = stats.norm.ppf(1 - 1.0 / n_trials)
    z_b = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_MASCHERONI) * z_a + EULER_MASCHERONI * z_b))


def calculate_deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_observations: int,
    n_strategy_trials: int,
    returns_skew: float = 0.0,
    returns_kurtosis: float = 3.0,
) -> Optional[float]:
    """
    Calculates the Deflated Sharpe Ratio (DSR) to adjust for multiple testing, skewness, and kurtosis.
    """
    if n_observations is None or n_observations <= 1:
        logger.warning("⚠️ DSR: Insufficient number of observations for meaningful calculation.")
        return None

    excess_kurtosis = returns_kurtosis - 3.0
    variance_term = (
        1
        - returns_skew * sharpe_ratio
        + (excess_kurtosis / 4.0) * sharpe_ratio ** 2
    )
    variance_term = max(variance_term, 1e-8)

    sr_estimation_variance = variance_term / (n_observations - 1)
    sr0 = _expected_max_sharpe_under_null(n_strategy_trials, sr_estimation_variance)

    denom = np.sqrt(sr_estimation_variance)
    if denom <= 0:
        return None

    dsr_z = (sharpe_ratio - sr0) / denom
    return float(stats.norm.cdf(dsr_z))


def _build_ticker_history_index(test_universe: pd.DataFrame) -> dict:
    """
    Pre-computes daily ticker history indexed by jalali_date for evaluating capital increase events across rebalance windows.
    """
    history = {}
    for ticker, group in test_universe.groupby('ticker_code'):
        g = group.sort_values('jalali_date').set_index('jalali_date')[['is_capital_increase', 'raw_return']]
        history[ticker] = g
    return history


def _apply_capital_increase_share_adjustment(
    ticker_history: dict,
    previous_date,
    current_date,
    portfolio_shares: dict,
    equal_weight_shares: dict,
) -> Tuple[dict, dict, list]:
    """
    Applies cumulative share adjustment factors for capital increase events occurring within the (previous_date, current_date] interval.
    """
    all_held_tickers = set(portfolio_shares.keys()) | set(equal_weight_shares.keys())
    adjusted = []

    for ticker in all_held_tickers:
        ticker_df = ticker_history.get(ticker)
        if ticker_df is None or ticker_df.empty:
            continue

        if previous_date is None:
            continue

        window_mask = (ticker_df.index > previous_date) & (ticker_df.index <= current_date)
        window = ticker_df.loc[window_mask]

        ci_events = window[window['is_capital_increase'] == 1]
        if ci_events.empty:
            continue

        cumulative_factor = 1.0
        n_valid_events = 0
        for raw_ret in ci_events['raw_return']:
            if pd.isna(raw_ret) or (1.0 + raw_ret) < MIN_ADJUSTMENT_DENOMINATOR:
                logger.warning(f"⚠️ {ticker}: raw_return={raw_ret} is unreliable for share adjustment; skipped.")
                continue
            cumulative_factor *= 1.0 / (1.0 + raw_ret)
            n_valid_events += 1

        if n_valid_events == 0:
            continue

        if ticker in portfolio_shares:
            portfolio_shares[ticker] *= cumulative_factor
        if ticker in equal_weight_shares:
            equal_weight_shares[ticker] *= cumulative_factor

        adjusted.append((ticker, n_valid_events, round(cumulative_factor, 4)))

    return portfolio_shares, equal_weight_shares, adjusted


def run_advanced_backtest(initial_capital: float = 50_000_000,
                          rebalance_period: int = 20,
                          start_date: Optional[int] = None,
                          num_strategy_trials: int = 1):
    """
    Runs the full backtest simulation incorporating model predictions, fees, slippage, and capital increase adjustments.
    """
    if start_date is None:
        raise ValueError("start_date must be explicitly specified.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model file not found.")

    logger.info("🧠 Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    logger.info("📊 Loading data...")
    full_data = load_all_processed_data()

    logger.info("🩺 Applying diagnostic corrections to backtest features...")
    full_data = apply_diagnostic_corrections(full_data)

    test_universe = full_data.dropna(subset=['label']).copy()
    test_universe = test_universe[test_universe['jalali_date'] >= start_date]

    if test_universe.empty:
        raise ValueError(f"❌ No data found on or after {start_date}.")

    ticker_history = _build_ticker_history_index(test_universe)

    grouped_by_date = {
        date: group for date, group in test_universe.groupby('jalali_date')
    }

    unique_dates = sorted(grouped_by_date.keys())
    rebalance_dates = unique_dates[::rebalance_period]

    logger.info(f"🚀 Simulating {len(rebalance_dates)} steps from {unique_dates[0]} to {unique_dates[-1]}")

    current_cash = initial_capital
    portfolio_shares = {}
    last_known_price = {}
    portfolio_history = []
    initial_dollar_rate = None
    equal_weight_shares = {}
    total_capital_increase_adjustments = 0
    previous_date = None

    for rebalance_idx, current_date in enumerate(rebalance_dates):
        market_today = grouped_by_date[current_date]

        for _, row in market_today.iterrows():
            last_known_price[row['ticker_code']] = row['close_price']

        portfolio_shares, equal_weight_shares, adjusted_today = (
            _apply_capital_increase_share_adjustment(
                ticker_history, previous_date, current_date, portfolio_shares, equal_weight_shares
            )
        )
        if adjusted_today:
            total_capital_increase_adjustments += len(adjusted_today)
            logger.info(f"🩹 Capital increase share adjustment up to {current_date} "
                        f"(from {previous_date}): {adjusted_today}")

        if rebalance_idx == 0:
            initial_tickers = market_today['ticker_code'].unique().tolist()
            if initial_tickers:
                capital_per_ticker = initial_capital / len(initial_tickers)
                for ticker in initial_tickers:
                    price = last_known_price.get(ticker, 0)
                    if price > 0:
                        equal_weight_shares[ticker] = capital_per_ticker / price

        equal_weight_value = sum(
            shares * last_known_price.get(ticker, 0)
            for ticker, shares in equal_weight_shares.items()
        )

        total_value = current_cash
        for ticker, shares in portfolio_shares.items():
            price = last_known_price.get(ticker, 0)
            total_value += shares * price

        today_dollar = float(market_today['dollar_rate'].iloc[0])
        if initial_dollar_rate is None:
            initial_dollar_rate = today_dollar
        market_value = initial_capital * (today_dollar / initial_dollar_rate)

        portfolio_history.append({
            'date': current_date,
            'total_value': total_value,
            'market_value': market_value,
            'equal_weight_value': equal_weight_value,
            'cash': current_cash,
            'positions': len(portfolio_shares),
        })

        updated_shares = dict(portfolio_shares)
        for ticker in list(updated_shares.keys()):
            stock_today = market_today[market_today['ticker_code'] == ticker]
            if stock_today.empty:
                continue

            row = stock_today.iloc[0]
            if row['is_locked_queue'] == 1 and row['stock_return'] < 0:
                continue

            close_price = row['close_price'] * (1 - SLIPPAGE_RATE)
            sale_revenue = updated_shares[ticker] * close_price
            fee = sale_revenue * SELL_FEE_RATE
            current_cash += (sale_revenue - fee)
            del updated_shares[ticker]

        if current_cash > MIN_ALLOCATION_TOMAN and not market_today.empty:
            X_today = market_today[FEATURE_COLS]
            preds = model.predict(X_today)
            alpha_scores = preds[:, 2] - preds[:, 0]

            signals = pd.DataFrame({
                'ticker': market_today['ticker_code'].values,
                'close_price': market_today['close_price'].values,
                'alpha': alpha_scores,
            })

            top = signals[signals['alpha'] > ALPHA_BUY_THRESHOLD] \
                .sort_values('alpha', ascending=False) \
                .head(TOP_N_BUYS)

            if not top.empty:
                cash_per_stock = current_cash / len(top)
                for _, row in top.iterrows():
                    buy_price = row['close_price'] * (1 + SLIPPAGE_RATE)
                    available = cash_per_stock * (1 - BUY_FEE_RATE)

                    if buy_price <= 0:
                        continue

                    shares = int(available / buy_price)
                    if shares > 0:
                        current_cash -= cash_per_stock
                        updated_shares[row['ticker']] = (
                            updated_shares.get(row['ticker'], 0) + shares
                        )

        portfolio_shares = updated_shares
        previous_date = current_date

    if total_capital_increase_adjustments > 0:
        logger.info(f"🩹 Total capital increase adjustments applied across entire backtest: {total_capital_increase_adjustments}")
    else:
        logger.info("ℹ️ No capital increase events occurred for held tickers across rebalance periods.")

    perf_df = pd.DataFrame(portfolio_history)
    perf_df['daily_return'] = perf_df['total_value'].pct_change()

    total_return = (perf_df['total_value'].iloc[-1] - initial_capital) / initial_capital
    benchmark_return = (perf_df['market_value'].iloc[-1] - initial_capital) / initial_capital
    equal_weight_return = (perf_df['equal_weight_value'].iloc[-1] - initial_capital) / initial_capital

    perf_df['peak'] = perf_df['total_value'].cummax()
    perf_df['drawdown'] = (perf_df['total_value'] - perf_df['peak']) / perf_df['peak']
    max_dd = perf_df['drawdown'].min()

    rf_period = RISK_FREE_ANNUAL / (TRADING_DAYS_PER_YEAR / rebalance_period)
    mean_ret = perf_df['daily_return'].mean()
    std_ret = perf_df['daily_return'].std()
    periods_per_year = TRADING_DAYS_PER_YEAR / rebalance_period
    sharpe = ((mean_ret - rf_period) / (std_ret + 1e-8)) * np.sqrt(periods_per_year)

    downside = perf_df['daily_return'][perf_df['daily_return'] < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-8
    sortino = ((mean_ret - rf_period) / (downside_std + 1e-8)) * np.sqrt(periods_per_year)

    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    clean_returns = perf_df['daily_return'].dropna()
    n_obs = len(clean_returns)
    if n_obs > 3:
        returns_skew = float(stats.skew(clean_returns))
        returns_kurtosis = float(stats.kurtosis(clean_returns, fisher=False))
    else:
        returns_skew, returns_kurtosis = 0.0, 3.0

    dsr = calculate_deflated_sharpe_ratio(
        sharpe_ratio=sharpe,
        n_observations=n_obs,
        n_strategy_trials=num_strategy_trials,
        returns_skew=returns_skew,
        returns_kurtosis=returns_kurtosis,
    )

    logger.info("\n" + "="*50)
    logger.info("📊 Backtest Results (with full capital increase adjustment fix):")
    logger.info(f"💰 Initial Capital: {initial_capital:,.0f} Toman")
    logger.info(f"💳 Final Portfolio Value (AI): {perf_df['total_value'].iloc[-1]:,.0f} Toman")
    logger.info(f"💵 Final USD Benchmark Value: {perf_df['market_value'].iloc[-1]:,.0f} Toman")
    logger.info(f"📈 Total Return (AI): {total_return:.2%} | USD Benchmark: {benchmark_return:.2%} | Equal Weight Benchmark: {equal_weight_return:.2%}")
    logger.info(f"🎯 Alpha vs USD: {(total_return - benchmark_return):+.2%}")
    logger.info(f"🎯 Alpha vs Equal Weight (Pure Stock Selection): {(total_return - equal_weight_return):+.2%}")
    logger.info(f"📉 Max Drawdown: {max_dd:.2%}")
    logger.info(f"⚖️ Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f}")
    if dsr is not None:
        verdict = "✅ GO (True signal likely)" if dsr >= 0.95 else \
                  "⚠️ SUSPICIOUS (Insufficient evidence to distinguish from noise)" if dsr >= 0.5 else \
                  "🛑 NO-GO (Statistically indistinguishable from random noise)"
        logger.info(f"🧪 Deflated Sharpe Ratio: {dsr:.3f} (n_trials={num_strategy_trials}, "
                    f"n_obs={n_obs}) → {verdict}")
    else:
        logger.warning("🧪 Deflated Sharpe Ratio: Could not be calculated (insufficient data).")
    logger.info("="*50)

    os.makedirs("excel_outputs", exist_ok=True)
    perf_df.to_excel("excel_outputs/backtest_equity_curve.xlsx", index=False)

    import json
    with open("excel_outputs/backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "benchmark_return_dollar": benchmark_return,
            "benchmark_return_equal_weight": equal_weight_return,
            "alpha_vs_equal_weight": total_return - equal_weight_return,
            "capital_increase_adjustments_applied": total_capital_increase_adjustments,
        }, f, ensure_ascii=False, indent=2)

    return perf_df, {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'equal_weight_benchmark_return': equal_weight_return,
        'alpha_vs_dollar': total_return - benchmark_return,
        'alpha_vs_equal_weight': total_return - equal_weight_return,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'deflated_sharpe_ratio': dsr,
        'num_strategy_trials': num_strategy_trials,
        'returns_skew': returns_skew,
        'returns_kurtosis': returns_kurtosis,
        'final_value': perf_df['total_value'].iloc[-1],
        'capital_increase_adjustments_applied': total_capital_increase_adjustments,
    }


if __name__ == "__main__":
    full_data = load_all_processed_data()
    labeled = full_data.dropna(subset=['label'])
    all_dates = sorted(labeled['jalali_date'].unique())
    midpoint = all_dates[len(all_dates) // 2] if all_dates else None

    run_advanced_backtest(50_000_000, 20, midpoint, num_strategy_trials=1)

def run_backtest():
    """
    Main entry point for running historical backtest.
    """
    pass

if __name__ == "__main__":
    run_backtest()
