import os
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard_api")

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
LIVE_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "live_market_predictions.xlsx")
EQUITY_CURVE_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "backtest_equity_curve.xlsx")
BACKTEST_METRICS_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "backtest_metrics.json")
FIXED_INCOME_KEY = "Fixed Income Fund"

MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_models")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
MODEL_BOOSTER_PATH = os.path.join(MODEL_DIR, "lgb_robo_advisor.txt")

MAIN_PY_PATH = os.path.join(PROJECT_ROOT, "main.py")
PIPELINE_PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)

FRONTEND_DIR = str(Path(__file__).resolve().parent.parent / "frontend")


def load_backtest_metrics() -> Optional[dict]:
    """
    Backtest is executed once for a given scenario, not per risk/horizon;
    thus we return these actual values for every request rather than None.
    """
    if not os.path.exists(BACKTEST_METRICS_PATH):
        return None
    with open(BACKTEST_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def jalali_int_to_label(jalali_date) -> str:
    s = str(int(jalali_date))
    return f"{s[:4]}/{s[4:6]}/{s[6:8]}"


def _ensure_project_root_on_path():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


app = FastAPI(title="Robo Advisor Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/rankings")
def get_rankings():
    """
    Converts output from live_predictor.py (live_market_predictions.xlsx)
    to the format expected by the frontend dashboard.
    """
    if not os.path.exists(LIVE_PREDICTIONS_PATH):
        raise HTTPException(
            status_code=503,
            detail="Live predictions not generated yet. Please run the pipeline first.",
        )

    df = pd.read_excel(LIVE_PREDICTIONS_PATH)

    # Check for expected column structures (handles English and Persian column naming gracefully)
    symbol_col = 'Ticker' if 'Ticker' in df.columns else ('نماد' if 'نماد' in df.columns else None)
    price_col = 'Close Price' if 'Close Price' in df.columns else ('قیمت پایانی' if 'قیمت پایانی' in df.columns else None)
    score_col = 'Alpha Score' if 'Alpha Score' in df.columns else ('امتیاز خرید (Alpha Score)' if 'امتیاز خرید (Alpha Score)' in df.columns else None)
    change_col = 'Change %' if 'Change %' in df.columns else ('درصد تغییر' if 'درصد تغییر' in df.columns else None)
    drop_col = 'Drop Prob (Class 0)' if 'Drop Prob (Class 0)' in df.columns else ('احتمال ریزش/عقب‌ماندگی (کلاس ۰)' if 'احتمال ریزش/عقب‌ماندگی (کلاس ۰)' in df.columns else None)
    neutral_col = 'Neutral Prob (Class 1)' if 'Neutral Prob (Class 1)' in df.columns else ('احتمال خنثی/همگام بازار (کلاس ۱)' if 'احتمال خنثی/همگام بازار (کلاس ۱)' in df.columns else None)
    growth_col = 'Growth Prob (Class 2)' if 'Growth Prob (Class 2)' in df.columns else ('احتمال رشد شارپ > ۵٪ (کلاس ۲)' if 'احتمال رشد شارپ > ۵٪ (کلاس ۲)' in df.columns else None)

    if not all([symbol_col, price_col, score_col, change_col, drop_col, neutral_col, growth_col]):
        raise HTTPException(status_code=500, detail="Required prediction columns were not found in data frame.")

    probs = df[[drop_col, neutral_col, growth_col]].values
    predicted_class = np.argmax(probs, axis=1)

    stale_col = 'Is Stale' if 'Is Stale' in df.columns else 'داده قدیمی / مشکوک به توقف نماد'
    has_stale_col = stale_col in df.columns

    out = []
    for i, row in df.iterrows():
        out.append({
            "symbol": row[symbol_col],
            "alpha_score": round(float(row[score_col]), 2),
            "predicted_class": int(predicted_class[i]),
            "price": int(row[price_col]),
            "change_percent": float(row[change_col]),
            "drop_prob": float(row[drop_col]),
            "neutral_prob": float(row[neutral_col]),
            "growth_prob": float(row[growth_col]),
            "is_stale": bool(row[stale_col]) if has_stale_col else False,
        })

    return out


@app.get("/api/equity-curve")
def get_equity_curve():
    """
    Converts backtest_equity_curve.xlsx to the dashboard format.
    """
    if not os.path.exists(EQUITY_CURVE_PATH):
        raise HTTPException(
            status_code=503,
            detail="Backtest has not been executed yet. Run backtester.py first.",
        )

    df = pd.read_excel(EQUITY_CURVE_PATH)
    required_cols = {'date', 'total_value', 'market_value'}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Columns {missing} were not found in backtest_equity_curve.xlsx.",
        )

    out = [
        {
            "date": jalali_int_to_label(row['date']),
            "portfolio_value": float(row['total_value']),
            "market_value": float(row['market_value']),
        }
        for _, row in df.iterrows()
    ]
    return out


@app.get("/api/backtest-metrics")
def get_backtest_metrics():
    """
    Returns raw JSON metrics from backtest_metrics.json.
    """
    metrics = load_backtest_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail="backtest_metrics.json does not exist. Run backtester.py first.",
        )
    return metrics


@app.get("/api/model-health")
def get_model_health():
    """
    Returns fold metrics and feature importances calculated directly from LightGBM model booster.
    """
    if not os.path.exists(MODEL_METADATA_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet. Run train_model.py first.",
        )

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_importance = []
    if os.path.exists(MODEL_BOOSTER_PATH):
        _ensure_project_root_on_path()
        try:
            import lightgbm as lgb
            from train_model import FEATURE_COLS

            booster = lgb.Booster(model_file=MODEL_BOOSTER_PATH)
            gains = booster.feature_importance(importance_type="gain")
            pairs = sorted(zip(FEATURE_COLS, gains), key=lambda p: p[1], reverse=True)
            feature_importance = [[name, round(float(val), 1)] for name, val in pairs[:10]]
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

    return {
        "trained_at": metadata.get("trained_at"),
        "folds": metadata.get("fold_metrics", []),
        "feature_importance": feature_importance,
    }


class OptimizeRequest(BaseModel):
    capital: float
    risk_appetite: str  # 'low' | 'medium' | 'high'
    time_horizon: str   # 'short' | 'mid' | 'long'


@app.post("/api/portfolio/optimize")
def post_portfolio_optimize(req: OptimizeRequest):
    _ensure_project_root_on_path()

    try:
        from portfolio_optimizer import optimize_portfolio
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"portfolio_optimizer.py not found: {e}")

    if req.risk_appetite not in ('low', 'medium', 'high'):
        raise HTTPException(status_code=400, detail="risk_appetite must be one of low/medium/high.")
    if req.time_horizon not in ('short', 'mid', 'long'):
        raise HTTPException(status_code=400, detail="time_horizon must be one of short/mid/long.")

    try:
        allocation_df = optimize_portfolio(
            capital=req.capital,
            risk_appetite=req.risk_appetite,
            time_horizon=req.time_horizon,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"{e} - Please run live_predictor.py first.",
        )
    except Exception as e:
        logger.exception("optimize_portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))

    if isinstance(allocation_df, dict):
        weights = {k: v / req.capital for k, v in allocation_df.items()}
        bt = load_backtest_metrics() or {}
        metrics = {
            "total_return": bt.get("total_return"),
            "sharpe_ratio": bt.get("sharpe_ratio"),
            "max_drawdown": bt.get("max_drawdown"),
            "risk_exposure": 0.0,
        }
        return {"portfolio_weights": weights, "metrics": metrics}

    weights = {}
    for _, row in allocation_df.iterrows():
        # Safely extract weight column checking both English and Persian keys
        weight_val = row.get('Total Portfolio Weight') or row.get('وزن از کل سبد') or '0%'
        pct_str = str(weight_val).replace('%', '').strip()
        
        symbol_key = row.get('Ticker') or row.get('نماد') or 'Unknown'
        weights[symbol_key] = round(float(pct_str) / 100.0, 4)

    bt = load_backtest_metrics() or {}
    metrics = {
        "total_return": bt.get("total_return"),
        "sharpe_ratio": bt.get("sharpe_ratio"),
        "max_drawdown": bt.get("max_drawdown"),
        "risk_exposure": round(1 - weights.get(FIXED_INCOME_KEY, weights.get("صندوق درآمد ثابت", 0.0)), 4),
    }

    return {"portfolio_weights": weights, "metrics": metrics}


class PipelineRunResponse(BaseModel):
    status: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


@app.post("/api/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline():
    """
    Executes main.py as a subprocess and awaits completion.
    """
    if not os.path.exists(MAIN_PY_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"main.py not found at '{MAIN_PY_PATH}'. Check PROJECT_ROOT environment variable.",
        )

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [PIPELINE_PYTHON, "main.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

    logger.info(f"Executing pipeline: {PIPELINE_PYTHON} main.py (cwd={PROJECT_ROOT})")

    try:
        result = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline execution timed out after 15 minutes.")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to execute Python process ({PIPELINE_PYTHON}): {e}.",
        )

    ok = result.returncode == 0
    if ok:
        logger.info("main.py executed successfully.")
    else:
        logger.error(f"main.py exited with code {result.returncode}:\n{result.stderr[-4000:]}")

    return {
        "status": "ok" if ok else "error",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "live_predictions_exist": os.path.exists(LIVE_PREDICTIONS_PATH),
        "equity_curve_exists": os.path.exists(EQUITY_CURVE_PATH),
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"Serving dashboard frontend from '{FRONTEND_DIR}' at '/'")
else:
    logger.warning(f"Frontend directory not found: '{FRONTEND_DIR}'")
