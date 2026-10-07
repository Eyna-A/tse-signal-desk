"""
ablate_geo.py -- does the GDELT geo block carry real out-of-sample information?

Run from the project folder (needs ai_features_outputs/ and train_model.py):
    python ablate_geo.py                 # ~2-5 min
    python ablate_geo.py --placebos 12 --seeds 5

Same purged walk-forward folds as train_model.py, but:
  * a FIXED number of boosting rounds (no early stopping on the test fold), evaluated at 25 and 100
    rounds, so the result does not depend on when a noisy metric happens to stall;
  * three variants
      A  all FEATURE_COLS (real geo)
      B  FEATURE_COLS without geo_cii_score / geo_high_risk_flag / geo_data_available
      P  A, but the geo series is circularly shifted by a random number of dates (placebo).
         Same smoothness and same distribution as the real series, but pointing at the wrong
         dates. If real geo does not beat the placebos, it is only acting as a "which period is
         this" fingerprint, not as information.
  * metrics: multi_logloss (lower = better), accuracy, macro-F1, and mean daily cross-sectional
    Spearman IC between Alpha Score (P2 - P0) and the realised 60d stock return -- the quantity the
    optimizer actually uses.
"""
import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, log_loss

import train_model as tm

logging.getLogger("train_model").setLevel(logging.WARNING)

GEO_FEATURES = ["geo_cii_score", "geo_high_risk_flag", "geo_data_available"]
PLACEBO_COLS = ["geo_cii_score", "geo_high_risk_flag"]
CHECKPOINTS = (25, 100)

# Same values as `params` inside train_model.train_lightgbm_with_purged_cv()
PARAMS = {
    "objective": "multiclass", "num_class": 3, "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": tm.REG_ALPHA_L1, "lambda_l2": tm.REG_LAMBDA_L2,
    "min_child_samples": tm.MIN_CHILD_SAMPLES, "extra_trees": tm.USE_EXTRA_TREES,
    "verbose": -1,
}


def daily_ic(dates, score, realized, min_names=5) -> float:
    d = pd.DataFrame({"d": dates, "s": score, "r": realized})
    ics = []
    for _, g in d.groupby("d"):
        if len(g) >= min_names and g["s"].nunique() > 1 and g["r"].nunique() > 1:
            ics.append(spearmanr(g["s"], g["r"])[0])
    return float(np.nanmean(ics)) if ics else np.nan


def make_placebo(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    daily = df.groupby("jalali_date")[PLACEBO_COLS].first().sort_index()
    rolled = pd.DataFrame(np.roll(daily.to_numpy(), shift, axis=0),
                          index=daily.index, columns=PLACEBO_COLS)
    out = df.copy()
    out[PLACEBO_COLS] = rolled.reindex(out["jalali_date"]).to_numpy()
    return out


def run_variant(df, features, splits, seeds, n_rounds) -> pd.DataFrame:
    rows = []
    for fold, (tr_dates, te_dates) in enumerate(splits, 1):
        tr, te = df[df["jalali_date"].isin(tr_dates)], df[df["jalali_date"].isin(te_dates)]
        if tr.empty or te.empty:
            continue
        ytr, yte = tr["label"].astype(int), te["label"].astype(int).to_numpy()
        _, wmap = tm._compute_dampened_class_weights(ytr)
        w = ytr.map(wmap).to_numpy()
        probs = {k: np.zeros((len(te), 3)) for k in CHECKPOINTS}
        for sd in seeds:
            m = lgb.train({**PARAMS, "random_state": sd},
                          lgb.Dataset(tr[features], ytr, weight=w), num_boost_round=n_rounds)
            for k in CHECKPOINTS:
                probs[k] += m.predict(te[features], num_iteration=k) / len(seeds)
        for k in CHECKPOINTS:
            p = probs[k]
            pred = p.argmax(1)
            rows.append({
                "fold": fold, "rounds": k,
                "logloss": log_loss(yte, p, labels=[0, 1, 2]),
                "acc": accuracy_score(yte, pred),
                "f1": f1_score(yte, pred, average="macro"),
                "ic": daily_ic(te["jalali_date"].to_numpy(), p[:, 2] - p[:, 0],
                               te["future_stock_return_60d"].to_numpy()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebos", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=max(CHECKPOINTS))
    args = ap.parse_args()

    df = tm.load_and_combine_features()
    df = df[df["label"].notna()].reset_index(drop=True)
    splits = tm.purged_walk_forward_splits(df["jalali_date"].unique())
    n_dates = df["jalali_date"].nunique()
    seeds = list(range(args.seeds))
    all_feats = list(tm.FEATURE_COLS)
    no_geo = [f for f in all_feats if f not in GEO_FEATURES]
    print(f"{len(df)} labeled rows, {n_dates} dates, {len(splits)} folds, "
          f"{len(all_feats)} features ({len(no_geo)} without geo)\n")

    def summarize(res):
        return res.groupby("rounds")[["logloss", "acc", "f1", "ic"]].mean()

    res_a = summarize(run_variant(df, all_feats, splits, seeds, args.rounds))
    res_b = summarize(run_variant(df, no_geo, splits, seeds, args.rounds))

    rng = np.random.default_rng(0)
    shifts = rng.integers(250, max(251, n_dates - 250), size=args.placebos)
    placebo = []
    for i, s in enumerate(shifts, 1):
        print(f"  placebo {i}/{args.placebos} (shift {int(s)} dates)...", flush=True)
        placebo.append(summarize(run_variant(make_placebo(df, int(s)), all_feats, splits, [0], args.rounds)))

    print("\n=== mean over walk-forward folds ===")
    for k in CHECKPOINTS:
        a, b = res_a.loc[k], res_b.loc[k]
        p = pd.DataFrame([r.loc[k] for r in placebo])
        print(f"\n--- {k} boosting rounds ---")
        print(f"{'variant':<22}{'logloss':>9}{'acc':>8}{'f1':>8}{'IC':>9}")
        print(f"{'A  real geo':<22}{a.logloss:>9.4f}{a.acc:>8.3f}{a.f1:>8.3f}{a.ic:>9.4f}")
        print(f"{'B  no geo':<22}{b.logloss:>9.4f}{b.acc:>8.3f}{b.f1:>8.3f}{b.ic:>9.4f}")
        print(f"{'P  placebo (mean)':<22}{p.logloss.mean():>9.4f}{p.acc.mean():>8.3f}{p.f1.mean():>8.3f}{p.ic.mean():>9.4f}")
        print(f"{'P  placebo (std)':<22}{p.logloss.std():>9.4f}{p.acc.std():>8.3f}{p.f1.std():>8.3f}{p.ic.std():>9.4f}")
        beat_ll = int((a.logloss < p.logloss).sum())
        beat_ic = int((a.ic > p.ic).sum())
        print(f"A - B logloss = {a.logloss - b.logloss:+.4f} (negative = geo helps)   "
              f"A - B IC = {a.ic - b.ic:+.4f}")
        print(f"real geo beats {beat_ll}/{len(p)} placebos on logloss, {beat_ic}/{len(p)} on IC")
    print("\nRule of thumb: treat geo as informative only if A beats B AND beats nearly all placebos "
          "at both checkpoints. With ~10 years there are only a handful of independent "
          "geopolitical episodes, so even then it is weak evidence.")


if __name__ == "__main__":
    main()
