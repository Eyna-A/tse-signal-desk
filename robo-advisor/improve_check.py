"""
improve_check.py -- where does the model's (weak) stock-picking signal come from, and does a
better-posed learning problem extract more of it?

Run from the project folder:   python improve_check.py        (a few minutes)

Part 1  Univariate rank-IC of every feature against the realised 60d stock return, per date
        (cross-sectional Spearman across your tickers). Shows which features carry ANY signal,
        and whether the sign is stable across time blocks.
Part 2  Same purged walk-forward folds as train_model.py, several ways of posing the problem:
          V0 current   3-class label, current LightGBM params, alpha = P2 - P0
          V1 regular.  3-class label, much smaller / more regularised trees
          V2 rank-tgt  regress the within-date RANK of the 60d return (uses the full ordering
                       instead of 3 thresholded buckets)
          V3 rank+x    V2 and every feature rank-normalised within each date (removes "which period
                       is this" information; leaves only "how does this stock compare today")
          V4 rank+x-ci V3 without the capital-increase features (they dominate importance and were
                       flagged as unstable)
          V5 rank-tgt-ci V2 without the capital-increase features (cleaner test of the same question)
Part 3  The same univariate IC after removing each ticker's own (past-only, expanding) mean of the
        feature. A feature that is mostly a ticker "identity" (e.g. one stock simply had more
        capital increases and happened to do worse) loses its IC here; a real signal keeps it.
        Metrics: mean daily rank-IC with a t-stat on NON-overlapping dates (60-date block means),
        and top-K minus equal-weight-basket 60d return (the thing the optimizer needs).
Decision rule: a variant is only an improvement if t-stat > ~2 AND top-K beats the basket in most
non-overlapping windows. Otherwise it is noise.
"""
import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_model as tm

logging.getLogger("train_model").setLevel(logging.WARNING)

HORIZON = 60
Y = "future_stock_return_60d"
CAP_INC = ["days_since_last_capital_increase_scaled", "capital_increase_freq_252d"]
CHECKPOINTS = (40, 120)

CLS_PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l1": tm.REG_ALPHA_L1,
    "lambda_l2": tm.REG_LAMBDA_L2, "min_child_samples": tm.MIN_CHILD_SAMPLES,
    "extra_trees": tm.USE_EXTRA_TREES, "verbose": -1,
}
REG_CLS = {**CLS_PARAMS, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
           "min_child_samples": 200, "lambda_l2": 10.0, "feature_fraction": 0.6}
REG_RANK = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
            "min_child_samples": 200, "lambda_l2": 10.0, "feature_fraction": 0.6, "bagging_fraction": 0.8,
            "bagging_freq": 5, "extra_trees": True, "verbose": -1}


def per_date_ic(dates, score, realized, min_names=5) -> pd.Series:
    d = pd.DataFrame({"d": dates, "s": score, "r": realized})
    out = {}
    for k, g in d.groupby("d"):
        if len(g) >= min_names and g["s"].nunique() > 1 and g["r"].nunique() > 1:
            out[k] = g["s"].rank().corr(g["r"].rank())
    return pd.Series(out)


def block_means(s: pd.Series) -> np.ndarray:
    """Means of consecutive HORIZON-date blocks: uses all data, avoids picking a lucky phase."""
    v = s.sort_index().dropna().to_numpy()
    n = len(v) // HORIZON
    return v[: n * HORIZON].reshape(n, HORIZON).mean(axis=1) if n else np.array([])


def tstat_blocks(s: pd.Series):
    b = block_means(s)
    if len(b) < 3 or b.std() == 0:
        return np.nan, len(b)
    return float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))), len(b)


def feature_ic_table(df: pd.DataFrame, feats) -> pd.DataFrame:
    dates = np.sort(df["jalali_date"].unique())
    blocks = np.array_split(dates, 5)
    rows = []
    for f in feats:
        ic = per_date_ic(df["jalali_date"].to_numpy(), df[f].to_numpy(), df[Y].to_numpy())
        t, n = tstat_blocks(ic)
        by_block = [ic[ic.index.isin(b)].mean() for b in blocks]
        rows.append({"feature": f, "mean_IC": ic.mean(), "t_blocks": t,
                     "blocks_same_sign": int(sum(np.sign(x) == np.sign(ic.mean()) for x in by_block)),
                     "of": len(blocks)})
    return pd.DataFrame(rows).sort_values("t_blocks", key=lambda s: s.abs(), ascending=False)


def rank_norm(df, cols):
    return df.groupby("jalali_date")[cols].rank(pct=True) - 0.5


def make_variants(df, feats):
    nocap = [f for f in feats if f not in CAP_INC]
    ranked = df.copy()
    ranked[feats] = rank_norm(df, feats)
    ranked["rank_target"] = df.groupby("jalali_date")[Y].rank(pct=True) - 0.5
    df = df.assign(rank_target=ranked["rank_target"])
    return {
        "V0 current   (3-class, current params)": (df, feats, "cls", CLS_PARAMS),
        "V1 regularised 3-class": (df, feats, "cls", REG_CLS),
        "V2 rank target": (df, feats, "rank", REG_RANK),
        "V3 rank target + rank features": (ranked, feats, "rank", REG_RANK),
        "V4 V3 without capital-increase feats": (ranked, nocap, "rank", REG_RANK),
        "V5 V2 without capital-increase feats": (df, nocap, "rank", REG_RANK),
    }


def run(data, feats, kind, params, splits, rounds, seeds, k):
    ic_parts, spread_parts = [], []
    for tr_dates, te_dates in splits:
        tr, te = data[data["jalali_date"].isin(tr_dates)], data[data["jalali_date"].isin(te_dates)]
        if tr.empty or te.empty:
            continue
        preds = {c: np.zeros(len(te)) for c in CHECKPOINTS}
        for sd in seeds:
            if kind == "cls":
                y = tr["label"].astype(int)
                _, wmap = tm._compute_dampened_class_weights(y)
                m = lgb.train({**params, "random_state": sd},
                              lgb.Dataset(tr[feats], y, weight=y.map(wmap).to_numpy()), rounds)
                for c in CHECKPOINTS:
                    p = m.predict(te[feats], num_iteration=c)
                    preds[c] += (p[:, 2] - p[:, 0]) / len(seeds)
            else:
                m = lgb.train({**params, "random_state": sd},
                              lgb.Dataset(tr[feats], tr["rank_target"]), rounds)
                for c in CHECKPOINTS:
                    preds[c] += m.predict(te[feats], num_iteration=c) / len(seeds)
        for c in CHECKPOINTS:
            t = te[["jalali_date", Y]].copy()
            t["score"] = preds[c]
            ic_parts.append(per_date_ic(t["jalali_date"].to_numpy(), t["score"].to_numpy(),
                                        t[Y].to_numpy()).rename(c))
            sp = {}
            for d, g in t.groupby("jalali_date"):
                if len(g) >= max(5, k + 2):
                    sp[d] = g.nlargest(k, "score")[Y].mean() - g[Y].mean()
            spread_parts.append(pd.Series(sp, name=c))
    res = {}
    for c in CHECKPOINTS:
        ic = pd.concat([s for s in ic_parts if s.name == c]).sort_index()
        sp = pd.concat([s for s in spread_parts if s.name == c]).sort_index()
        t_ic, n = tstat_blocks(ic)
        ind = block_means(sp)
        res[c] = {"IC": ic.mean(), "t(IC)": t_ic, "n_indep": n, f"top{k}-basket": sp.mean(),
                  "beats_basket": float((ind > 0).mean()) if len(ind) else np.nan}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--extra", nargs="*", default=["is_locked_queue", "queue_persistence_5d", "volume_ratio",
                                                   "days_since_prev_row"],
                    help="columns that exist in the feature files but are NOT model features; "
                         "they are only added to Part 1 / Part 3 (univariate IC) so you can see "
                         "whether they deserve a place in the model")
    args = ap.parse_args()

    df = tm.load_and_combine_features()
    df = df[df["label"].notna() & df[Y].notna()].reset_index(drop=True)
    feats = list(tm.FEATURE_COLS)
    extras = [c for c in args.extra if c in df.columns and c not in feats]
    ic_feats = feats + extras
    pd.set_option("display.width", 200)

    print(f"{len(df)} rows, {df['ticker_code'].nunique()} tickers, {df['jalali_date'].nunique()} dates; "
          f"non-model columns also tested in Parts 1/3: {extras or 'none found'}\n")
    if "geo_cii_score" in df.columns:
        spread = df.groupby("jalali_date")["geo_cii_score"].nunique()
        bad = spread[spread > 1]
        print(f"Part 0: dates where geo_cii_score differs ACROSS tickers: {len(bad)} "
              f"(should be 0; first/last: {bad.index.min() if len(bad) else '-'} / "
              f"{bad.index.max() if len(bad) else '-'})\n")
    print("=== Part 1: univariate rank-IC of each feature vs realised 60d return ===")
    t1 = feature_ic_table(df, ic_feats)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n(|t| < ~2 on 60-date blocks = indistinguishable from zero; "
          "'blocks_same_sign' = in how many of 5 time blocks the IC had the same sign)\n")

    print("=== Part 3: raw IC vs within-ticker IC (ticker's own past mean removed) ===")
    within = df.copy()
    for f in ic_feats:
        within[f] = df[f] - df.groupby("ticker_code")[f].transform(
            lambda s: s.expanding(min_periods=60).mean().shift(1))
    t3 = feature_ic_table(within.dropna(subset=ic_feats), ic_feats)[["feature", "mean_IC", "t_blocks"]] \
        .rename(columns={"mean_IC": "IC_within", "t_blocks": "t_within"})
    t3 = t1[["feature", "mean_IC", "t_blocks"]].rename(columns={"mean_IC": "IC_raw", "t_blocks": "t_raw"}) \
        .merge(t3, on="feature")
    print(t3.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n(large drop from raw to within = the feature mostly identifies WHICH ticker, "
          "not WHEN it will outperform; with 8 tickers that is ~8 data points)\n")

    splits = tm.purged_walk_forward_splits(df["jalali_date"].unique())
    print("=== Part 2: ways of posing the problem (same purged walk-forward folds) ===")
    rows = []
    for name, (data, f, kind, params) in make_variants(df, feats).items():
        print(f"  running {name} ...", flush=True)
        res = run(data, f, kind, params, splits, max(CHECKPOINTS), list(range(args.seeds)), args.k)
        for c, r in res.items():
            rows.append({"variant": name, "rounds": c, **r})
    t2 = pd.DataFrame(rows)
    print()
    print(t2.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nImprovement = higher t(IC) AND positive top-K-minus-basket AND beats_basket well above 50% "
          "on the non-overlapping windows. With only ~20 such windows, differences of a few percent are noise.")


if __name__ == "__main__":
    main()
