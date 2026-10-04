
from __future__ import annotations

import argparse
import io
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("gdelt_geo_features")

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV_PATH = Path(os.environ.get("GDELT_RAW_CSV", BASE_DIR / "gdelt_iran_daily.csv"))
MISSING_DAYS_PATH = RAW_CSV_PATH.with_suffix(".missing.txt")
MARKET_DB_PATH = BASE_DIR / "tsetmc_market_data.db"

CII_HIGH_RISK_THRESHOLD = 65.0
CII_CRITICAL_THRESHOLD = 80.0

GDELT_EVENTS_URL = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
GDELT_V1_DAILY_START = date(2013, 4, 1)

# 0-based column positions in the GDELT 1.0 event file (58 tab-separated columns, no header).
# See the GDELT 1.0 Event Codebook.
COL_ACTOR1_COUNTRY = 7      # CAMEO 3-letter (Iran = IRN)
COL_ACTOR2_COUNTRY = 17     # CAMEO 3-letter
COL_QUAD_CLASS = 29         # 1 verbal coop, 2 material coop, 3 verbal conflict, 4 material conflict
COL_GOLDSTEIN = 30
COL_NUM_MENTIONS = 31
COL_AVG_TONE = 34
COL_ACTION_GEO_COUNTRY = 51  # FIPS 2-letter (Iran = IR)
_USECOLS = {
    COL_ACTOR1_COUNTRY: "a1cc", COL_ACTOR2_COUNTRY: "a2cc", COL_QUAD_CLASS: "quad",
    COL_GOLDSTEIN: "goldstein", COL_NUM_MENTIONS: "mentions", COL_AVG_TONE: "tone",
    COL_ACTION_GEO_COUNTRY: "geocc",
}
IRAN_CAMEO, IRAN_FIPS = "IRN", "IR"

# Feature-construction parameters
LAG_DAYS = 1                 # file for day D is published after D ends -> usable from D+1
MIN_EVENTS_PER_DAY = 20      # days with fewer Iran events are treated as missing, not as "calm"
BASELINE_DAYS = 365
MIN_BASELINE_DAYS = 60
SMOOTH_DAYS = 3              # trailing mean so Thu/Fri (non-trading) news still reaches Saturday
MAX_FFILL_DAYS = 5
Z_CLIP = 4.0

REQUEST_TIMEOUT = 120
GEO_COLS = ["geo_cii_score", "geo_conflict_event_count_7d", "geo_high_risk_flag"]
RAW_COLS = ["date", "n_events", "n_conflict", "avg_tone", "avg_goldstein"]
RECENT_DAYS_ON_REFRESH = 14


def create_session(use_system_proxy: Optional[bool] = None) -> requests.Session:
    """Direct session by default (the original script's proxy 407/502 workaround).
    Set GDELT_USE_SYSTEM_PROXY=1 to honour HTTP(S)_PROXY instead."""
    if use_system_proxy is None:
        use_system_proxy = os.environ.get("GDELT_USE_SYSTEM_PROXY", "0") == "1"
    s = requests.Session()
    s.trust_env = use_system_proxy
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504],
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (geo-feature-builder)"})
    return s


def aggregate_iran_events(df: pd.DataFrame) -> dict:
    """df: columns a1cc,a2cc,geocc (str) and quad,goldstein,mentions,tone (numeric)."""
    mask = (df["a1cc"] == IRAN_CAMEO) | (df["a2cc"] == IRAN_CAMEO) | (df["geocc"] == IRAN_FIPS)
    d = df[mask]
    out = {"n_events": int(len(d)), "n_conflict": int((d["quad"] == 4).sum()),
           "avg_tone": np.nan, "avg_goldstein": np.nan}
    if d.empty:
        return out
    w = d["mentions"].fillna(1).clip(lower=1)           # weight by media attention

    def wavg(col: str) -> float:
        ok = d[col].notna()
        return float(np.average(d.loc[ok, col], weights=w[ok])) if ok.any() else np.nan

    out["avg_tone"], out["avg_goldstein"] = wavg("tone"), wavg("goldstein")
    return out


def parse_day_zip(content: bytes) -> dict:
    """Parse one YYYYMMDD.export.CSV.zip into the daily Iran aggregate."""
    df = pd.read_csv(io.BytesIO(content), compression="zip", sep="\t", header=None,
                     usecols=list(_USECOLS), dtype=str, on_bad_lines="skip",
                     encoding_errors="replace", quoting=3)
    df = df.rename(columns=_USECOLS)
    for c in ("quad", "goldstein", "mentions", "tone"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Schema guard: if GDELT ever changes the layout, fail loudly instead of writing garbage.
    if len(df) and (df["quad"].dropna().between(1, 4).mean() < 0.9):
        raise ValueError("Unexpected GDELT column layout (QuadClass column not in 1..4).")
    return aggregate_iran_events(df)


def _http_get(url: str, session: requests.Session) -> Optional[bytes]:
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def fetch_day(day: date, session: requests.Session):
    """Returns aggregate dict, or None if GDELT has no file for that day (404)."""
    content = _http_get(GDELT_EVENTS_URL.format(ymd=day.strftime("%Y%m%d")), session)
    if content is None:
        return None
    row = parse_day_zip(content)
    row["date"] = day.isoformat()
    return row


def load_raw_cache() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        return pd.DataFrame(columns=RAW_COLS)
    df = pd.read_csv(RAW_CSV_PATH)
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{RAW_CSV_PATH} is missing columns: {sorted(missing)}")
    df = df[RAW_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def save_raw_cache(df: pd.DataFrame) -> None:
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates("date", keep="last").sort_values("date").to_csv(RAW_CSV_PATH, index=False)


def _load_missing() -> set:
    return set(MISSING_DAYS_PATH.read_text().split()) if MISSING_DAYS_PATH.exists() else set()


def _daterange(start: date, end: date) -> Iterable[date]:
    for i in range((end - start).days + 1):
        yield start + timedelta(days=i)


def update_raw_cache(start: date, end: date, workers: int = 3) -> pd.DataFrame:
    """Download only the days not already cached. Resumable: progress is saved every 50 days."""
    start = max(start, GDELT_V1_DAILY_START)
    raw = load_raw_cache()
    have, missing = set(raw["date"].astype(str)), _load_missing()
    todo = [d for d in _daterange(start, end)
            if d.isoformat() not in have and d.isoformat() not in missing]
    if not todo:
        logger.info("GDELT cache already covers %s..%s", start, end)
        return raw

    logger.info("Downloading %d GDELT daily files (%d workers)...", len(todo), workers)
    session = create_session()
    new_rows, new_missing, failed = [], set(), []
    recent_cutoff = date.today() - timedelta(days=10)

    def job(d):
        try:
            return d, fetch_day(d, session), None
        except Exception as e:                      # network/parse error -> retry on next run
            return d, None, e

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, d) for d in todo]
        for fut in as_completed(futs):
            d, row, err = fut.result()
            done += 1
            if err is not None:
                failed.append(d)
                logger.warning("GDELT %s failed: %s", d, err)
            elif row is None:
                if d < recent_cutoff:               # genuine gap in GDELT's archive
                    new_missing.add(d.isoformat())
                # else: not published yet -> try again on a later run
            else:
                new_rows.append(row)
            if done % 50 == 0 and new_rows:
                raw = pd.concat([raw, pd.DataFrame(new_rows)[RAW_COLS]], ignore_index=True)
                save_raw_cache(raw)
                new_rows = []
                logger.info("  ...%d/%d days processed", done, len(todo))

    if new_rows:
        raw = pd.concat([raw, pd.DataFrame(new_rows)[RAW_COLS]], ignore_index=True)
    if new_missing:
        MISSING_DAYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        MISSING_DAYS_PATH.write_text("\n".join(sorted(missing | new_missing)))
    save_raw_cache(raw)
    if failed:
        logger.warning("%d days failed and will be retried on the next run.", len(failed))
    if raw.empty:
        raise RuntimeError("No GDELT data could be downloaded and no cache exists. "
                           "Check connectivity (GDELT_USE_SYSTEM_PROXY=1 if you need a proxy) "
                           "or import data with --from-csv. Refusing to fabricate zero features.")
    return load_raw_cache()


def _trailing_z(s: pd.Series) -> pd.Series:
    """z-score vs. a trailing baseline that EXCLUDES the current day."""
    mu = s.rolling(BASELINE_DAYS, min_periods=MIN_BASELINE_DAYS).mean().shift(1)
    sd = s.rolling(BASELINE_DAYS, min_periods=MIN_BASELINE_DAYS).std().shift(1)
    return (s - mu) / sd.where(sd > 1e-9)


def build_geo_features(raw: pd.DataFrame, lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """raw daily Iran aggregates -> DataFrame[jalali_date(int), 3 geo columns]."""
    import jdatetime

    if raw.empty:
        raise ValueError("Raw GDELT aggregate is empty; nothing to build features from.")

    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.drop_duplicates("date", keep="last").set_index("date").sort_index()
    r = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq="D"))

    valid = r["n_events"] >= MIN_EVENTS_PER_DAY
    conflict_share = (r["n_conflict"] / r["n_events"]).where(valid)   # volume-drift robust
    tone = r["avg_tone"].where(valid)
    goldstein = r["avg_goldstein"].where(valid)

    # Higher = more stress: more conflict share, more negative tone, more negative Goldstein.
    z = pd.concat([_trailing_z(conflict_share), _trailing_z(-tone), _trailing_z(-goldstein)], axis=1)
    composite = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= 2)
    composite = composite.clip(-Z_CLIP, Z_CLIP).rolling(SMOOTH_DAYS, min_periods=1).mean()

    out = pd.DataFrame(index=r.index)
    out["geo_cii_score"] = 100.0 / (1.0 + np.exp(-composite))           # 50 = normal, in (0,100)
    out["geo_conflict_event_count_7d"] = r["n_conflict"].rolling(7, min_periods=3).sum()
    out = out.dropna(subset=["geo_cii_score", "geo_conflict_event_count_7d"])

    # Availability lag, then carry forward across short gaps only.
    out.index = out.index + pd.Timedelta(days=lag_days)
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))
    out = out.ffill(limit=MAX_FFILL_DAYS).dropna()

    out["geo_high_risk_flag"] = (out["geo_cii_score"] >= CII_HIGH_RISK_THRESHOLD).astype(float)

    def to_jalali_int(ts: pd.Timestamp) -> int:
        j = jdatetime.date.fromgregorian(date=ts.date())
        return j.year * 10000 + j.month * 100 + j.day

    out["jalali_date"] = [to_jalali_int(ts) for ts in out.index]
    return out[["jalali_date"] + GEO_COLS].reset_index(drop=True)


_EMPTY = pd.DataFrame(columns=["jalali_date"] + GEO_COLS)
_history_cache: dict = {}


def load_geo_feature_history() -> pd.DataFrame:
    """DataFrame[jalali_date:int64, 3 geo columns]; empty (with a warning) if no usable CSV."""
    if not RAW_CSV_PATH.exists():
        logger.warning("GDELT CSV not found at %s; geopolitical features will be 0.", RAW_CSV_PATH)
        return _EMPTY.copy()
    key = (str(RAW_CSV_PATH), RAW_CSV_PATH.stat().st_mtime_ns)
    if key not in _history_cache:
        try:
            feats = build_geo_features(load_raw_cache())
        except Exception as e:                       # fail-open for training / live
            logger.warning("Could not build geo features from %s: %s", RAW_CSV_PATH, e)
            return _EMPTY.copy()
        _history_cache.clear()
        _history_cache[key] = feats
    return _history_cache[key].copy()


def record_daily_snapshot() -> Optional[dict]:
    """Top up the CSV with any missing recent days (cheap, incremental), then return the latest
    feature row. Never raises: the live predictor must keep working offline."""
    try:
        raw = load_raw_cache()
        yesterday = date.today() - timedelta(days=1)
        start = (pd.to_datetime(raw["date"]).max().date() + timedelta(days=1)) if len(raw) \
            else yesterday - timedelta(days=RECENT_DAYS_ON_REFRESH)
        if start <= yesterday:
            update_raw_cache(start, yesterday)
    except Exception as e:
        logger.warning("GDELT refresh failed (%s); continuing with the existing CSV.", e)
    hist = load_geo_feature_history()
    if hist.empty:
        return None
    last = hist.iloc[-1]
    logger.info("Iran geo stress (jalali %d): cii=%.1f, high_risk=%d",
                last.jalali_date, last.geo_cii_score, int(last.geo_high_risk_flag))
    return last.to_dict()


def _latest_trading_jalali_date() -> Optional[int]:
    if not MARKET_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(MARKET_DB_PATH)
        try:
            row = conn.execute("SELECT MAX(jalali_date) FROM daily_prices").fetchone()
        finally:
            conn.close()
        return int(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None


def _jalali_ordinal(d: int) -> int:                  # approximate, same convention as live_predictor
    return (d // 10000) * 360 + ((d // 100) % 100) * 30 + d % 100


def get_current_risk_brake(default_max_equity_ratio: float) -> float:
    """Lower the maximum equity share during acute geopolitical stress."""
    hist = load_geo_feature_history()
    if hist.empty:
        return default_max_equity_ratio
    today = _latest_trading_jalali_date() or int(hist["jalali_date"].iloc[-1])
    eligible = hist[hist["jalali_date"] <= today]
    if eligible.empty:
        return default_max_equity_ratio
    row = eligible.iloc[-1]
    if _jalali_ordinal(today) - _jalali_ordinal(int(row.jalali_date)) > 10:
        logger.warning("Latest geo row (%d) is stale vs %d; risk brake not applied.", row.jalali_date, today)
        return default_max_equity_ratio
    cii = float(row.geo_cii_score)
    if cii >= CII_CRITICAL_THRESHOLD:
        logger.warning("Critical geo stress (%.1f); maximum equity allocation capped at 20%%.", cii)
        return min(default_max_equity_ratio, 0.20)
    if cii >= CII_HIGH_RISK_THRESHOLD:
        logger.warning("High geo stress (%.1f); maximum equity allocation reduced by 30%%.", cii)
        return default_max_equity_ratio * 0.70
    return default_max_equity_ratio


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def verify_day(day: date) -> None:
    """Re-download one day and compare with the CSV -> confirms this module's Iran filter and
    weighting reproduce how your historical CSV was produced."""
    fresh = fetch_day(day, create_session())
    old = load_raw_cache()
    old = old[old["date"] == day.isoformat()]
    print("downloaded now :", fresh)
    print("in your CSV    :", old.iloc[0].to_dict() if len(old) else "(not in CSV)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=_parse_date, default=None, metavar="YYYY-MM-DD")
    args = ap.parse_args()
    if args.verify:
        verify_day(args.verify)
    else:
        print(record_daily_snapshot())
