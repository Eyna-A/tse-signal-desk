"""
geo_regime_check.py -- is geopolitical stress useful for ALLOCATION (how much to hold in equities)?

Geo features are identical for every stock on a given day, so they cannot rank stocks against each
other; the only place they can help is deciding how much risk to take. This script asks that directly,
with no model training involved:

  1. Bin dates by geo_cii_score (<50 normal, 50-65 elevated, 65-80 high, >=80 critical -- the same
     thresholds the risk brake uses) and show what the NEXT 60 days looked like for the equal-weight
     basket and for USD.
  2. Spearman correlation between geo_cii_score and the next-60d basket / USD returns, with a
     circular-shift permutation p-value (keeps the series' autocorrelation, destroys the date link).
  3. The risk brake as a rule: fixed 65% risky share vs 65% reduced to 45.5% when cii>=65 and to 20%
     when cii>=80 (thresholds fixed in advance, not fitted), versus the same risky share held
     constant. Reports profit rate on non-overlapping windows with a 95% lower bound.

Run from the project folder:   python geo_regime_check.py
Caveat: ~10 years = roughly 15-20 independent 60-day windows and very few geopolitical episodes.
Treat everything here as weak evidence either way.
"""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import beta

import train_model as tm
from gdelt_geo_features import CII_CRITICAL_THRESHOLD, CII_HIGH_RISK_THRESHOLD, load_geo_feature_history

logging.getLogger("train_model").setLevel(logging.WARNING)

HORIZON = 60
RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR = 0.28, 242      # same as portfolio_optimizer.py
BINS = [(-np.inf, 50, "normal  (<50)"), (50, CII_HIGH_RISK_THRESHOLD, "elevated (50-65)"),
        (CII_HIGH_RISK_THRESHOLD, CII_CRITICAL_THRESHOLD, "high (65-80)"),
        (CII_CRITICAL_THRESHOLD, np.inf, "critical (>=80)")]
pct = lambda x: f"{x:.1%}"


def cp_lower(k, n, a=0.05):
    return 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))


def load_daily() -> pd.DataFrame:
    df = tm.load_and_combine_features()
    daily = df.groupby("jalali_date").agg(
        basket=("future_stock_return_60d", "mean"), usd=("future_market_return_60d", "mean"),
        n=("ticker_code", "nunique")).dropna()
    daily = daily[daily["n"] >= 3].reset_index()
    geo = load_geo_feature_history()
    if geo.empty:
        raise SystemExit("No GDELT geo history found (gdelt_iran_daily.csv).")
    out = daily.merge(geo[["jalali_date", "geo_cii_score"]], on="jalali_date", how="inner")
    return out.sort_values("jalali_date").reset_index(drop=True)


def bin_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, name in BINS:
        s = d[(d["geo_cii_score"] >= lo) & (d["geo_cii_score"] < hi)]
        if s.empty:
            continue
        rows.append({"geo regime": name, "dates": len(s), "share": len(s) / len(d),
                     "basket mean": s["basket"].mean(), "basket median": s["basket"].median(),
                     "basket profit": (s["basket"] > 0).mean(), "basket p5": s["basket"].quantile(.05),
                     "usd mean": s["usd"].mean(), "usd profit": (s["usd"] > 0).mean()})
    return pd.DataFrame(rows)


def perm_test(x: np.ndarray, y: np.ndarray, n_perm=2000, seed=0):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan")
    real = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    n = len(x)
    shifts = rng.integers(HORIZON * 2, max(HORIZON * 2 + 1, n - HORIZON * 2), size=n_perm)
    null = np.array([np.corrcoef(np.roll(rx, s), ry)[0, 1] for s in shifts])
    return real, float((np.abs(null) >= abs(real)).mean())


def brake_rule(cii: np.ndarray, base: float) -> np.ndarray:
    return np.where(cii >= CII_CRITICAL_THRESHOLD, min(base, 0.20),
                    np.where(cii >= CII_HIGH_RISK_THRESHOLD, base * 0.70, base))


def evaluate(port: np.ndarray) -> dict:
    ind = port[::HORIZON]
    k, n = int((ind > 0).sum()), len(ind)
    return {"profit (all dates)": (port > 0).mean(), "profit (indep)": k / n, "n_indep": n,
            "95% lower": cp_lower(k, n), "mean": port.mean(), "p5": np.percentile(port, 5), "worst": port.min()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    d = load_daily()
    fi = (1 + RISK_FREE_ANNUAL) ** (HORIZON / TRADING_DAYS_PER_YEAR) - 1
    print(f"\n{len(d)} dates ({d['jalali_date'].iloc[0]} -> {d['jalali_date'].iloc[-1]}), "
          f"~{len(d) // HORIZON} independent 60d windows. Fixed income 60d = {fi:.2%}\n")

    print("=== 1. What happened in the next 60 days, by geo regime ===")
    t = bin_table(d)
    print(t.to_string(index=False, formatters={c: pct for c in t.columns if c not in ("geo regime", "dates")}))

    print("\n=== 2. Rank correlation with next-60d returns (circular-shift permutation p-value) ===")
    for col in ("basket", "usd"):
        r, p = perm_test(d["geo_cii_score"].to_numpy(), d[col].to_numpy())
        if np.isnan(r):
            print(f"geo_cii_score vs next-60d {col:<6}: n/a (series is constant)")
        else:
            print(f"geo_cii_score vs next-60d {col:<6}: spearman = {r:+.3f}   p = {p:.3f}")

    print("\n=== 3. Does the risk brake help? (equal-weight basket as the risky sleeve) ===")
    cii, basket = d["geo_cii_score"].to_numpy(), d["basket"].to_numpy()
    rows = []
    for base in (0.65, 0.95):
        w_brake = brake_rule(cii, base)
        rows.append({"policy": f"constant {base:.0%} risky", **evaluate(base * (basket - 0.01) + (1 - base) * fi)})
        rows.append({"policy": f"{base:.0%} with geo brake", **evaluate(w_brake * (basket - 0.01) + (1 - w_brake) * fi)})
    rows.append({"policy": "fixed income only", **evaluate(np.full(len(d), fi))})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, formatters={c: pct for c in r.columns if c not in ("policy", "n_indep")}))
    share_braked = float((cii >= CII_HIGH_RISK_THRESHOLD).mean())
    print(f"\nBrake active on {share_braked:.0%} of dates. If the brake rows do not clearly improve "
          "'95% lower' / 'p5' / 'worst' versus the constant rows, geo is not earning its place as a "
          "risk control either -- and its thresholds should not be tuned on this same sample.")


if __name__ == "__main__":
    main()
