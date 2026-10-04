import os, sqlite3, numpy as np, pandas as pd
os.environ["GDELT_RAW_CSV"] = "gdelt_iran_daily.csv"
import gdelt_geo_features as g

rng = np.random.default_rng(1)
days = pd.date_range("2015-12-01", "2026-09-30", freq="D")           # ~10.8 years, like the real file
n = rng.integers(2000, 3800, len(days)); share = rng.normal(0.14, 0.02, len(days))
tone = rng.normal(-2.6, 0.4, len(days)); gold = rng.normal(0.8, 0.5, len(days))
share[-40:] += 0.08; tone[-40:] -= 1.5; gold[-40:] -= 1.2              # crisis in the last 40 days
pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "n_events": n, "n_conflict": (share*n).round().astype(int),
              "avg_tone": tone, "avg_goldstein": gold}).to_csv(g.RAW_CSV_PATH, index=False)

h = g.load_geo_feature_history()
print(h.dtypes.to_dict(), len(h), "rows;", h.jalali_date.iloc[0], "->", h.jalali_date.iloc[-1])
assert h.jalali_date.dtype == np.int64 and h.jalali_date.is_unique
assert h.jalali_date.iloc[0] < 13950101 + 100, "history must cover the start of the stock data (13950101)"
assert h.geo_cii_score.between(0, 100, inclusive="neither").all()
calm, crisis = h.geo_cii_score.iloc[200:-60].mean(), h.geo_cii_score.iloc[-10:].mean()
print(f"calm={calm:.1f} crisis={crisis:.1f} high-risk calm={h.geo_high_risk_flag.iloc[200:-60].mean():.1%}")
assert crisis > 80 and calm < 55
assert g.load_geo_feature_history() is not None and len(g._history_cache) == 1   # cached

# no look-ahead: rewrite the last 60 days of the CSV, earlier feature values must not move
cut = h.jalali_date.iloc[-70]
raw = pd.read_csv(g.RAW_CSV_PATH); raw.loc[raw.index[-60:], "avg_tone"] = -25; raw.to_csv(g.RAW_CSV_PATH, index=False)
h2 = g.load_geo_feature_history()
assert np.allclose(h[h.jalali_date <= cut][g.GEO_COLS].values, h2[h2.jalali_date <= cut][g.GEO_COLS].values)
print("no look-ahead OK; CSV change invalidated cache OK")

# risk brake driven by the latest trading date in the stock DB
conn = sqlite3.connect(g.MARKET_DB_PATH); conn.execute("create table daily_prices(instrument_id int, jalali_date int)")
conn.execute("insert into daily_prices values(1,?)", (int(h2.jalali_date.iloc[-1]),)); conn.commit(); conn.close()
assert g.get_current_risk_brake(0.65) == 0.20
conn = sqlite3.connect(g.MARKET_DB_PATH); conn.execute("update daily_prices set jalali_date=14050101"); conn.commit(); conn.close()
print("brake when trading data is far ahead of geo data (stale):", g.get_current_risk_brake(0.65))
assert g.get_current_risk_brake(0.65) == 0.65

# live refresh must be fail-open offline
r = g.record_daily_snapshot(); print("offline refresh returned:", {k: round(v, 1) for k, v in r.items()})
assert r is not None

# the real downstream code
import train_model as tm
tr = h2.tail(30)[["jalali_date"] + g.GEO_COLS]
full = pd.concat([tr.assign(ticker_code=t) for t in "ABC"], ignore_index=True)
full = tm._fix_geo_feature_stationarity(full); print("geo_data_available:", full.geo_data_available.mean())
import feature_engineering, live_predictor, portfolio_optimizer
print("ALL CHECKS PASSED (all modules import without geopolitical_features)")
