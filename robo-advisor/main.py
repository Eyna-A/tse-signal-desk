import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")

TARGET_TICKERS = ['خودرو', 'خساپا', 'فملی', 'فولاد', 'شبندر', 'شپنا', 'وبملت', 'شستا']

BASE_DIR = Path(__file__).resolve().parent
PORTFOLIO_ALLOCATION_JSON = BASE_DIR / "excel_outputs" / "portfolio_allocation.json"
BACKTEST_METRICS_JSON = BASE_DIR / "excel_outputs" / "backtest_metrics.json"
BACKTEST_EQUITY_XLSX = BASE_DIR / "excel_outputs" / "backtest_equity_curve.xlsx"
FIXED_INCOME_LABEL = "صندوق درآمد ثابت"


def _predicted_class_from_row(row: pd.Series) -> int:
    """
    Reconstruct the final predicted class from class probabilities using argmax,
    matching the logic used in live_predictor.py.
    """
    probs = {
        0: row["احتمال ریزش/عقب‌ماندگی (کلاس ۰)"],
        1: row["احتمال خنثی/همگام بازار (کلاس ۱)"],
        2: row["احتمال رشد شارپ > ۵٪ (کلاس ۲)"],
    }
    return max(probs, key=probs.get)


def _build_ranking_rows(ranking_table: pd.DataFrame) -> list[dict]:
    """
    Convert ranking_table (output from generate_live_predictions) to the dictionary schema
    expected by dashboard_export.write_rankings_json.
    """
    rows = []
    for _, r in ranking_table.iterrows():
        rows.append({
            "symbol": r["نماد"],
            "alpha_score": float(r["امتیاز خرید (Alpha Score)"]),
            "predicted_class": _predicted_class_from_row(r),
            "price": int(r["قیمت پایانی"]),
            "change_percent": float(r["درصد تغییر"]),
            "drop_prob": float(r["احتمال ریزش/عقب‌ماندگی (کلاس ۰)"]),
            "neutral_prob": float(r["احتمال خنثی/همگام بازار (کلاس ۱)"]),
            "growth_prob": float(r["احتمال رشد شارپ > ۵٪ (کلاس ۲)"]),
            "is_stale": bool(r["داده قدیمی / مشکوک به توقف نماد"]),
            "signal_label": r["سیگنال سیستم"],
        })
    return rows


def _build_portfolio_weights(portfolio_result) -> dict:
    """
    Parse portfolio weight allocations from optimize_portfolio() output.
    If portfolio_result is a dictionary (fallback mode), normalize absolute amounts to fractions.
    Otherwise, read saved fraction weights from portfolio_allocation.json or parse DataFrame columns.
    """
    if isinstance(portfolio_result, dict):
        total = sum(portfolio_result.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in portfolio_result.items()}

    if PORTFOLIO_ALLOCATION_JSON.exists():
        with open(PORTFOLIO_ALLOCATION_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.warning(
        "⚠️ portfolio_allocation.json not found; parsing weights from DataFrame percentage strings "
        "(less precise fallback)."
    )
    weights = {}
    for _, r in portfolio_result.iterrows():
        pct_str = str(r["وزن از کل سبد"]).replace("%", "").strip()
        try:
            weights[r["نماد"]] = float(pct_str) / 100.0
        except ValueError:
            continue
    return weights


def _load_last_backtest_metrics() -> dict:
    """
    Load total_return, sharpe_ratio, and max_drawdown from the most recent manual backtest execution.
    Returns default zero values if backtest metrics file does not exist.
    """
    if not BACKTEST_METRICS_JSON.exists():
        logger.warning(
            "⚠️ backtest_metrics.json not found (backtester.py has not been run yet); "
            "metrics will display as zero in the dashboard."
        )
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    with open(BACKTEST_METRICS_JSON, "r", encoding="utf-8") as f:
        cached = json.load(f)

    logger.info("ℹ️ Backtest return metrics successfully loaded from disk.")
    return {
        "total_return": cached.get("total_return", 0.0),
        "sharpe_ratio": cached.get("sharpe_ratio", 0.0),
        "max_drawdown": cached.get("max_drawdown", 0.0),
    }


def export_dashboard_data(ranking_table: pd.DataFrame, portfolio_result) -> None:
    """
    Export all pipeline predictions and allocation results to JSON format for the React dashboard.
    """
    from dashboard_export import (
        write_rankings_json,
        write_portfolio_json,
        write_equity_curve_json,
        build_equity_curve_from_backtest,
    )

    write_rankings_json(_build_ranking_rows(ranking_table))

    weights = _build_portfolio_weights(portfolio_result)
    if weights:
        metrics = _load_last_backtest_metrics()
        metrics["risk_exposure"] = 1.0 - weights.get(FIXED_INCOME_LABEL, 0.0)
        write_portfolio_json(weights, metrics)
    else:
        logger.warning("⚠️ Portfolio weights are empty; portfolio.json was not generated.")

    if BACKTEST_EQUITY_XLSX.exists():
        points = build_equity_curve_from_backtest(BACKTEST_EQUITY_XLSX)
        write_equity_curve_json(points)
        logger.info("📈 equity_curve.json updated from latest backtest_equity_curve.xlsx.")
    else:
        logger.info(
            "ℹ️ backtest_equity_curve.xlsx does not exist -- run backtester.py separately "
            "to populate equity curve chart data."
        )


def run_full_pipeline():
    """
    Execute the full end-to-end investment robo-advisor pipeline.
    """
    from data_pipeline import run_pipeline
    from feature_engineering import run_feature_engineering_pipeline
    from train_model import train_lightgbm_with_purged_cv
    from live_predictor import generate_live_predictions
    from portfolio_optimizer import optimize_portfolio
    from backtester import run_backtest

    try:
        logger.info("=" * 70)
        logger.info("Stage 1/7: Raw data ingestion and cleaning")
        logger.info("=" * 70)
        run_pipeline(TARGET_TICKERS)

        logger.info("=" * 70)
        logger.info("Stage 2/7: Feature engineering and labeling")
        logger.info("=" * 70)
        run_feature_engineering_pipeline()

        logger.info("=" * 70)
        logger.info("Stage 3/7: Model training (Purged K-Fold CV)")
        logger.info("=" * 70)
        train_lightgbm_with_purged_cv()

        logger.info("=" * 70)
        logger.info("Stage 4/7: Live prediction generation")
        logger.info("=" * 70)
        ranking_table = generate_live_predictions()

        logger.info("=" * 70)
        logger.info("Stage 5/7: Portfolio optimization (Medium Risk)")
        logger.info("=" * 70)
        portfolio_result = optimize_portfolio(capital=50_000_000, risk_appetite='medium', time_horizon='mid')

        logger.info("=" * 70)
        logger.info("Stage 6/7: Exporting data for React dashboard")
        logger.info("=" * 70)
        try:
            export_dashboard_data(ranking_table, portfolio_result)
            logger.info("✅ Dashboard JSON files updated successfully.")
        except Exception as export_err:
            logger.error(f"⚠️ Dashboard export failed (core trading pipeline execution completed): {export_err}")

        logger.info("=" * 70)
        logger.info("Stage 7/7: Historical Backtesting")
        logger.info("=" * 70)
        try:
            run_backtest()
            export_dashboard_data(ranking_table, portfolio_result)
            logger.info("✅ Backtest completed and dashboard metrics refreshed.")
        except Exception as backtest_err:
            logger.error(f"⚠️ Backtest stage failed: {backtest_err}")

        logger.info("✅ Full pipeline (including backtest) completed successfully.")

    except Exception as e:
        logger.error(f"❌ Pipeline terminated with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_full_pipeline()
