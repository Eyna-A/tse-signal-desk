from __future__ import annotations

import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("geopolitical_features")

BASE_DIR = Path(__file__).resolve().parent
GEO_DB_PATH = str(BASE_DIR / "geo_signals.db")
MARKET_DB_PATH = str(BASE_DIR / "tsetmc_market_data.db")  # the DB data_pipeline.py actually writes
COUNTRY_CODE = "IR"

# "gdelt" (default, free, same definition as the training history) or "worldmonitor" (paid key).
# Do NOT mix sources on the same geo_signals.db: the two scores are on different definitions and
# mixing them creates train/serve skew.
GEO_SOURCE = os.environ.get("GEO_SOURCE", "gdelt").lower()

WORLDMONITOR_API_KEY_ENV_VAR = "WORLDMONITOR_API_KEY"

CII_HIGH_RISK_THRESHOLD = 65.0
CII_CRITICAL_THRESHOLD = 80.0

_warned_missing_sdk = False
_warned_missing_key = False
_sdk_client = None


def _get_sdk_client():
    """
    Creates a singleton Client instance from worldmonitor-sdk.
    Returns None if the package is missing or API key is not set, logging a warning once following fail-open design.
    """
    global _sdk_client, _warned_missing_sdk, _warned_missing_key

    if _sdk_client is not None:
        return _sdk_client

    try:
        from worldmonitor_sdk import Client
    except ImportError:
        if not _warned_missing_sdk:
            logger.warning(
                "⚠️ worldmonitor-sdk package is not installed. To enable real geopolitical signals: pip install worldmonitor-sdk"
            )
            _warned_missing_sdk = True
        return None

    api_key = os.environ.get(WORLDMONITOR_API_KEY_ENV_VAR)
    if not api_key:
        if not _warned_missing_key:
            logger.warning(
                f"⚠️ Environment variable {WORLDMONITOR_API_KEY_ENV_VAR} is not set. "
                "Real CII data requires an API key -- "
                "obtain a key from https://worldmonitor.app/pro and set it in the "
                f"{WORLDMONITOR_API_KEY_ENV_VAR} environment variable."
            )
            _warned_missing_key = True
        return None

    try:
        _sdk_client = Client(api_key=api_key)
    except Exception as e:
        logger.warning(f"⚠️ Failed to create Client from worldmonitor-sdk: {e}")
        return None

    return _sdk_client


def _get_latest_jalali_date() -> Optional[int]:
    """Latest trading jalali_date in tsetmc_market_data.db (treated as 'today')."""
    if not os.path.exists(MARKET_DB_PATH):
        logger.warning("⚠️ tsetmc_market_data.db not found; cannot determine 'today' for geopolitical features.")
        return None
    try:
        conn = sqlite3.connect(MARKET_DB_PATH)
        try:
            row = conn.execute("SELECT MAX(jalali_date) FROM daily_prices").fetchone()
        finally:
            conn.close()
        return int(row[0]) if row and row[0] is not None else None
    except sqlite3.Error as e:
        logger.warning(f"⚠️ Error reading tsetmc_market_data.db: {e}")
        return None


def _init_geo_db() -> None:
    conn = sqlite3.connect(GEO_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS geo_signals (
            jalali_date INTEGER PRIMARY KEY,
            geo_cii_score REAL,
            geo_conflict_event_count_7d REAL,
            geo_high_risk_flag INTEGER,
            recorded_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _extract_conflict_event_count(client, country: str) -> float:
    """
    Extracts recent conflict event count from client.conflict_events.
    Handles various response structures gracefully in accordance with fail-open design.
    """
    try:
        result = client.conflict_events(country=country, limit=100)
    except Exception as e:
        logger.warning(f"⚠️ client.conflict_events failed for {country}: {e}")
        return 0.0

    if isinstance(result, list):
        return float(len(result))
    if isinstance(result, dict):
        for key in ("events", "data", "results", "items"):
            if key in result and isinstance(result[key], list):
                return float(len(result[key]))
        logger.warning(
            f"⚠️ Unknown response structure for conflict_events (keys: {list(result.keys())}); "
            "defaulting event count to 0. Update this function if necessary."
        )
    return 0.0


def fetch_today_geo_snapshot() -> Optional[dict]:
    """Fetches today's geopolitical snapshot from the official worldmonitor API via Python SDK."""
    client = _get_sdk_client()
    if client is None:
        return None

    try:
        logger.info(f"🌍 Fetching risk score for {COUNTRY_CODE} from worldmonitor API...")
        risk = client.country_risk(COUNTRY_CODE)
    except Exception as e:
        logger.warning(f"⚠️ Fetching country risk from worldmonitor API failed: {e}")
        return None

    if not isinstance(risk, dict):
        logger.warning(f"⚠️ Unexpected response from country_risk (type={type(risk)}); skipping.")
        return None

    cii_score = risk.get("score")
    try:
        cii_score = float(cii_score)
    except (TypeError, ValueError):
        logger.warning(f"⚠️ Field 'score' missing or non-numeric in response: {risk}")
        return None

    is_stale = bool(risk.get("stale", False))
    if is_stale:
        logger.warning(f"⚠️ Returned CII data from API is flagged as stale (cached_at={risk.get('cached_at')}).")

    event_count = _extract_conflict_event_count(client, COUNTRY_CODE)

    return {
        "geo_cii_score": cii_score,
        "geo_conflict_event_count_7d": event_count,
        "geo_high_risk_flag": int(cii_score >= CII_HIGH_RISK_THRESHOLD),
    }


def record_daily_snapshot() -> Optional[dict]:
    """Refreshes geo_signals.db using the configured GEO_SOURCE."""
    if GEO_SOURCE == "gdelt":
        try:
            from gdelt_geo_features import update_geo_signals
            return update_geo_signals()      # incremental: only downloads missing days
        except Exception as e:               # fail-open: never block live predictions
            logger.warning(f"⚠️ GDELT geo update failed ({e}); using the existing geo_signals.db.")
            return None
    return _record_worldmonitor_snapshot()


def _record_worldmonitor_snapshot() -> Optional[dict]:
    """Legacy path: records today's WorldMonitor CII snapshot (requires API key)."""
    jalali_date = _get_latest_jalali_date()
    if jalali_date is None:
        return None

    snapshot = fetch_today_geo_snapshot()
    if snapshot is None:
        return None

    _init_geo_db()
    conn = sqlite3.connect(GEO_DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO geo_signals VALUES (?, ?, ?, ?, datetime('now'))",
        (
            jalali_date,
            snapshot["geo_cii_score"],
            snapshot["geo_conflict_event_count_7d"],
            snapshot["geo_high_risk_flag"],
        ),
    )
    conn.commit()
    conn.close()

    if snapshot["geo_high_risk_flag"]:
        logger.warning(
            f"🚨 High geopolitical risk detected for Iran (CII={snapshot['geo_cii_score']:.1f}) "
            f"recorded for date {jalali_date}."
        )
    else:
        logger.info(f"🌍 Iran CII for today ({jalali_date}): {snapshot['geo_cii_score']:.1f}")

    return snapshot


def load_geo_feature_history() -> pd.DataFrame:
    """Loads the complete collected history of geopolitical features."""
    cols = ["jalali_date", "geo_cii_score", "geo_conflict_event_count_7d", "geo_high_risk_flag"]
    if not os.path.exists(GEO_DB_PATH):
        return pd.DataFrame(columns=cols)
    try:
        conn = sqlite3.connect(GEO_DB_PATH)
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM geo_signals", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        logger.warning(f"⚠️ Error reading geo_signals.db: {e}")
        return pd.DataFrame(columns=cols)


def get_current_risk_brake(default_max_equity_ratio: float) -> float:
    """Automatically reduces the maximum equity portfolio allocation threshold during acute crisis (high CII)."""
    jalali_date = _get_latest_jalali_date()
    if jalali_date is None or not os.path.exists(GEO_DB_PATH):
        return default_max_equity_ratio

    try:
        conn = sqlite3.connect(GEO_DB_PATH)
        row = conn.execute(
            "SELECT jalali_date, geo_cii_score FROM geo_signals "
            "WHERE jalali_date <= ? ORDER BY jalali_date DESC LIMIT 1", (jalali_date,)
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return default_max_equity_ratio

    if not row or row[1] is None:
        return default_max_equity_ratio

    def _ord(d: int) -> int:   # approximate ordinal, same convention as live_predictor
        return (d // 10000) * 360 + ((d // 100) % 100) * 30 + d % 100
    if _ord(jalali_date) - _ord(int(row[0])) > 10:
        logger.warning(f"⚠️ Latest geopolitical row ({row[0]}) is stale vs {jalali_date}; risk brake not applied.")
        return default_max_equity_ratio

    cii_score = row[1]
    if cii_score >= CII_CRITICAL_THRESHOLD:
        logger.warning(f"🚨 Critical CII level ({cii_score:.1f}); maximum equity allocation capped at 20%.")
        return min(default_max_equity_ratio, 0.20)
    if cii_score >= CII_HIGH_RISK_THRESHOLD:
        logger.warning(f"⚠️ High CII level ({cii_score:.1f}); maximum equity allocation threshold reduced by 30%.")
        return default_max_equity_ratio * 0.70
    return default_max_equity_ratio