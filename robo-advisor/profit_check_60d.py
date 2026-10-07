"""
profit_check_60d.py -- "if I had followed this system, how often did I end up with a profit after 60 days?"

Run from the project folder (needs ai_features_outputs/, train_model.py and ablate_geo.py):
    python profit_check_60d.py
    python profit_check_60d.py --k 3 --cost 0.01 --rounds 60

What it does
  1. Re-creates the purged walk-forward folds of train_model.py and collects ONLY out-of-sample
     predictions (each row is predicted by a model that never saw its period).
  2. For every OOS date it picks the top-K tickers by Alpha Score (P2 - P0), exactly like the
     optimizer's ranking, and holds them equal-weighted for 60 trading days.
  3. It mixes that equity sleeve with a fixed-income sleeve (RISK_FREE_ANNUAL from the optimizer)
     for several equity shares, then reports the share of 60-day windows that ended in profit,
     the share that beat the dollar, and the tail (5th percentile, worst).

What it cannot do: tell you the NEXT 60 days will be profitable. It tells you how often that held
in the past, which is the only honest basis for sizing risk. Not modelled: trading halts, price-limit
queues that stop you selling, slippage beyond --cost, taxes.
"""
import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_model as tm
from ablate_geo import PARAMS

logging.getLogger("train_model").setLevel(logging.WARNING)

# Same constants as portfolio_optimizer.py
RISK_FREE_ANNUAL = 0.28
TRADING_DAYS_PER_YEAR = 242
HORIZON = 60
MIN_VALID_ALPHA = -0.20          # optimizer: Alpha Score > -20 (percentage points) -> fraction here
EQUITY_SHARES = (0.0, 0.20, 0.30, 0.40, 0.65, 0.95)


def oos_predictions(df, splits, rounds, seeds) -> pd.DataFrame:
    parts = []
    for fold, (tr_dates, te_dates) in enumerate(splits, 1):
        tr, te = df[df["jalali_date"].isin(tr_dates)], df[df["jalali_date"].isin(te_dates)]
        if tr.empty or te.empty:
            continue
        ytr = tr["label"].astype(int)
        _, wmap = tm._compute_dampened_class_weights(ytr)
        w = ytr.map(wmap).to_numpy()
        p = np.zeros((len(te), 3))
        for sd in seeds:
            m = lgb.train({**PARAMS, "random_state": sd},
                          lgb.Dataset(tr[tm.FEATURE_COLS], ytr, weight=w), num_boost_round=rounds)
            p += m.predict(te[tm.FEATURE_COLS]) / len(seeds)
        part = te[["jalali_date", "ticker_code", "future_stock_return_60d",
                   "future_market_return_60d"]].copy()
        part[["p0", "p1", "p2"]] = p
        part["fold"] = fold
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def window_returns(oos: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per OOS date: model-picked equity return, basket return, dollar return."""
    rows = []
    for d, g in oos.groupby("jalali_date"):
        g = g.dropna(subset=["future_stock_return_60d"])
        if len(g) < 3:
            continue
        alpha = g["p2"] - g["p0"]
        picks = g.loc[alpha[alpha > MIN_VALID_ALPHA].nlargest(k).index]
        rows.append({
            "jalali_date": d, "fold": int(g["fold"].iloc[0]),
            "picked": float(picks["future_stock_return_60d"].mean()) if len(picks) else np.nan,
            "basket": float(g["future_stock_return_60d"].mean()),
            "usd": float(g["future_market_return_60d"].mean()),
        })
    return pd.DataFrame(rows).sort_values("jalali_date").reset_index(drop=True)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta
    return 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))


def summarize(port: np.ndarray, usd: np.ndarray, fi: float) -> dict:
    """port: one 60d portfolio return per OOS date (overlapping). Non-overlapping = every 60th date."""
    ind = port[::HORIZON]
    k, n = int((ind > 0).sum()), len(ind)
    return {
        "profit_all": float((port > 0).mean()),
        "profit_indep": k / n, "n_indep": n, "ci95_low": clopper_pearson_lower(k, n),
        "beat_fi": float((port > fi).mean()), "beat_usd": float((port > usd).mean()),
        "mean": float(port.mean()), "median": float(np.median(port)),
        "p5": float(np.percentile(port, 5)), "worst": float(port.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="number of tickers held")
    ap.add_argument("--cost", type=float, default=0.01, help="assumed round-trip cost on risky sleeve (ASSUMPTION)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    df = tm.load_and_combine_features()
    df = df[df["label"].notna()].reset_index(drop=True)
    splits = tm.purged_walk_forward_splits(df["jalali_date"].unique())
    oos = oos_predictions(df, splits, args.rounds, list(range(args.seeds)))
    w = window_returns(oos, args.k)

    fi = (1 + RISK_FREE_ANNUAL) ** (HORIZON / TRADING_DAYS_PER_YEAR) - 1
    sleeves = {
        "model top-%d" % args.k: w["picked"].fillna(fi).to_numpy(),
        "equal-weight basket": w["basket"].to_numpy(),
        "USD": w["usd"].to_numpy(),
    }
    usd = w["usd"].to_numpy()
    print(f"\nOOS dates: {len(w)} | non-overlapping 60d windows: {len(w[::HORIZON])}  "
          f"(overlapping dates are NOT independent -> use the ci95_low column)\n"
          f"Fixed-income 60d return assumed: {fi:.2%} (RISK_FREE_ANNUAL={RISK_FREE_ANNUAL:.0%}; check it is realistic)\n"
          f"Cost on the risky sleeve: {args.cost:.1%} round trip (assumption). Folds present: "
          f"{sorted(w['fold'].unique().tolist())}\n")

    rows = []
    for e in EQUITY_SHARES:
        for name, r in sleeves.items():
            cost = 0.0 if name == "USD" else args.cost
            port = e * (r - cost) + (1 - e) * fi
            rows.append({"risky_share": e, "sleeve": name, **summarize(port, usd, fi)})
    t = pd.DataFrame(rows)
    t.to_csv("profit_check_60d_summary.csv", index=False)
    pd.set_option("display.width", 220)
    pct = lambda x: f"{x:.1%}"
    fmt = {c: pct for c in ["risky_share", "profit_all", "profit_indep", "ci95_low", "beat_fi", "beat_usd",
                            "mean", "median", "p5", "worst"]}
    print(t.to_string(index=False, formatters=fmt))

    print("\nPer-fold profit rate, model top-%d with 65%% risky share:" % args.k)
    port = 0.65 * (sleeves["model top-%d" % args.k] - args.cost) + 0.35 * fi
    byfold = pd.Series(port).groupby(w["fold"].to_numpy()).agg(
        profit_rate=lambda s: (s > 0).mean(), worst="min", windows="size")
    print(byfold.to_string(formatters={"profit_rate": pct, "worst": pct}))
    print("\nHow to read: compare the three sleeves at the SAME risky_share. If 'model' does not beat "
          "'equal-weight basket' there, stock picking adds nothing and only the share of capital in "
          "fixed income controls how often you profit.")


if __name__ == "__main__":
    main()
