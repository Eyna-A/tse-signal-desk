import os
import glob
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_model")

FEATURES_DIR = "ai_features_outputs"
MODEL_DIR = "ai_models"
LOOKAHEAD_DAYS = 60
SAFETY_MARGIN_DAYS = 5
MIN_TRAIN_DATES = 750
TEST_WINDOW_DATES = 300
CLASS_WEIGHT_DAMPEN_EXPONENT = 0.5

CAPITAL_INCREASE_RECENCY_WINDOW = 252
CAPITAL_INCREASE_RECENCY_MIN_PERIODS = 20
CAPITAL_INCREASE_RECENCY_NEUTRAL_FILL = 0.5

REG_ALPHA_L1 = 0.1
REG_LAMBDA_L2 = 0.5
MIN_CHILD_SAMPLES = 40
USE_EXTRA_TREES = True

USE_RECENCY_WEIGHTING = True
RECENCY_HALF_LIFE_DATES = 500

# Cross-sectional demeaned label configuration to mitigate macro-dominance
USE_CROSS_SECTIONAL_DEMEANED_LABEL = True
LABEL_THRESHOLD_STRONG = 0.05  # Demeaned margin threshold for class 2 (strong growth vs market mean)
LABEL_THRESHOLD_WEAK = 0.0     # Demeaned margin threshold for class 1 (outperforming market mean)

os.makedirs(MODEL_DIR, exist_ok=True)


def _rolling_recency_percentile(window: pd.Series) -> float:
    current = window.iloc[-1]
    if pd.isna(current):
        return np.nan
    return float((window >= current).mean())


def _fix_geo_feature_stationarity(full_df: pd.DataFrame) -> pd.DataFrame:
    """Fix stationarity and availability tracking for geopolitical features."""
    geo_cols = ['geo_cii_score', 'geo_conflict_event_count_7d', 'geo_high_risk_flag']
    if not set(geo_cols).issubset(full_df.columns):
        full_df['geo_data_available'] = 0.0
        return full_df

    nonzero_mask = (full_df[geo_cols] != 0).any(axis=1)

    if not nonzero_mask.any():
        logger.warning(" No non-zero geopolitical observations found. Check the worldmonitor module. "
                       "geo_data_available will be set to 0.0 for all rows.")
        full_df['geo_data_available'] = 0.0
        return full_df

    tracking_start_date = full_df.loc[nonzero_mask, 'jalali_date'].min()
    is_tracked_period = full_df['jalali_date'] >= tracking_start_date
    full_df['geo_data_available'] = is_tracked_period.astype(float)

    observed_mean = full_df.loc[nonzero_mask, 'geo_cii_score'].mean()
    if pd.isna(observed_mean):
        observed_mean = 0.0

    pretracking_mask = ~is_tracked_period
    n_backfilled = int(pretracking_mask.sum())
    if n_backfilled > 0:
        full_df.loc[pretracking_mask, 'geo_cii_score'] = observed_mean
        logger.info(f" geo_cii_score: Filled {n_backfilled} rows prior to tracking start with mean value "
                    f"({observed_mean:.2f}).")

    return full_df


# Final feature set (20 features)
USE_GEO_FEATURES = os.environ.get("USE_GEO_FEATURES", "1") == "1"

BASE_FEATURE_COLS = [
    'stock_return', 'volatility_20d', 'dollar_corr_20d',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist', 'atr_14', 'drawdown_20d',
    'dist_ma20', 'dist_ma50', 'market_regime_dollar', 'beta_proxy', 'relative_dollar_value',
    'capital_increase_freq_252d',

    'dollar_macro_trend',
    'days_since_last_capital_increase_scaled',
]

GEO_FEATURE_COLS = [
    'geo_cii_score',
    'geo_high_risk_flag',
    'geo_data_available',
]

FEATURE_COLS = BASE_FEATURE_COLS + (GEO_FEATURE_COLS if USE_GEO_FEATURES else [])

logger.info(
    f" Geopolitical features: {'ENABLED' if USE_GEO_FEATURES else 'DISABLED'}"
)
logger.info(f" Number of model features: {len(FEATURE_COLS)}")

# Removed features from previous versions:
# - dollar_return_sign     : zero mutual information
# - volume_ratio           : zero mutual information
# - days_since_prev_row    : zero mutual information


def apply_diagnostic_corrections(full_df: pd.DataFrame) -> pd.DataFrame:
    """Apply diagnostic corrections and feature stationarity adjustments."""
    if 'ticker_code' not in full_df.columns:
        raise ValueError(" apply_diagnostic_corrections requires 'ticker_code' column.")

    full_df = full_df.sort_values(['ticker_code', 'jalali_date']).reset_index(drop=True)

    if 'dollar_return' in full_df.columns:
        full_df['dollar_return_sign'] = np.sign(full_df['dollar_return'])
        full_df['dollar_macro_trend'] = full_df.groupby('ticker_code')['dollar_return'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
    else:
        full_df['dollar_return_sign'] = 0.0
        full_df['dollar_macro_trend'] = 0.0

    if 'days_since_last_capital_increase' in full_df.columns:
        full_df['days_since_last_capital_increase_scaled'] = (
            full_df.groupby('ticker_code')['days_since_last_capital_increase']
            .transform(
                lambda s: s.rolling(
                    window=CAPITAL_INCREASE_RECENCY_WINDOW,
                    min_periods=CAPITAL_INCREASE_RECENCY_MIN_PERIODS,
                ).apply(_rolling_recency_percentile, raw=False)
            )
        )
        full_df['days_since_last_capital_increase_scaled'] = (
            full_df['days_since_last_capital_increase_scaled']
            .fillna(CAPITAL_INCREASE_RECENCY_NEUTRAL_FILL)
        )
    else:
        full_df['days_since_last_capital_increase_scaled'] = CAPITAL_INCREASE_RECENCY_NEUTRAL_FILL

    full_df = _fix_geo_feature_stationarity(full_df)

    missing = [c for c in FEATURE_COLS if c not in full_df.columns]
    if missing:
        for col in missing:
            full_df[col] = 0.0
    return full_df


def _relabel_cross_sectionally(full_df: pd.DataFrame) -> pd.DataFrame:
    """Relabel targets cross-sectionally by de-meaning margin against daily market mean."""
    required = {'future_stock_return_60d', 'future_market_return_60d', 'jalali_date'}
    if not required.issubset(full_df.columns):
        logger.warning(" Raw future_*_60d columns missing for cross-sectional relabeling. "
                       "Original labels retained.")
        return full_df

    margin = full_df['future_stock_return_60d'] - full_df['future_market_return_60d']
    day_mean_margin = margin.groupby(full_df['jalali_date']).transform('mean')
    cross_sectional_margin = margin - day_mean_margin

    conditions = [
        cross_sectional_margin > LABEL_THRESHOLD_STRONG,
        cross_sectional_margin > LABEL_THRESHOLD_WEAK,
    ]
    choices = [2, 1]
    has_future = full_df['future_stock_return_60d'].notna()
    raw_label = np.select(conditions, choices, default=0)
    full_df['label'] = np.where(has_future, raw_label, np.nan)

    valid_labels = raw_label[has_future.to_numpy()]
    if len(valid_labels) > 0:
        new_dist = pd.Series(valid_labels).value_counts(normalize=True).sort_index()
        logger.info(f" New label distribution (cross-sectional demeaned daily): "
                    f"{new_dist.round(3).to_dict()}")

    return full_df


def load_and_combine_features():
    """Load, combine, adjust features, and optionally apply cross-sectional relabeling."""
    csv_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.csv")))
    parquet_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.parquet")))
    excel_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_AI_Features.xlsx")))
    all_files = csv_files or parquet_files or excel_files

    if not all_files:
        raise FileNotFoundError(" No feature files found.")

    file_ext = os.path.splitext(all_files[0])[1]
    logger.info(f" Loading {len(all_files)} feature files ({file_ext})...")

    combined_list = []
    for f in all_files:
        if f.endswith('.csv'):
            df = pd.read_csv(f)
        elif f.endswith('.parquet'):
            df = pd.read_parquet(f)
        else:
            df = pd.read_excel(f)
        df['ticker_code'] = os.path.basename(f).split('_')[0]
        combined_list.append(df)

    full_df = pd.concat(combined_list, ignore_index=True)

    logger.info(" Applying diagnostic corrections to features...")
    full_df = apply_diagnostic_corrections(full_df)

    if USE_CROSS_SECTIONAL_DEMEANED_LABEL:
        logger.info(" Relabeling cross-sectionally to mitigate macro-dominance...")
        full_df = _relabel_cross_sectionally(full_df)

    float_cols = full_df.select_dtypes(include=['float64']).columns
    full_df[float_cols] = full_df[float_cols].astype('float32')

    return full_df.sort_values(['jalali_date', 'ticker_code']).reset_index(drop=True)


def purged_walk_forward_splits(unique_dates, min_train_days=MIN_TRAIN_DATES,
                                test_window_days=TEST_WINDOW_DATES,
                                embargo_days=LOOKAHEAD_DAYS + SAFETY_MARGIN_DAYS):
    unique_dates = np.array(sorted(unique_dates))
    n = len(unique_dates)

    splits = []
    test_start_idx = min_train_days
    while test_start_idx < n:
        test_end_idx = min(test_start_idx + test_window_days, n)

        purge_cutoff_idx = max(0, test_start_idx - embargo_days)
        train_dates = set(unique_dates[:purge_cutoff_idx].tolist())

        test_dates_start_idx = min(test_start_idx + embargo_days, test_end_idx)
        test_dates = set(unique_dates[test_dates_start_idx:test_end_idx].tolist())

        if train_dates and test_dates:
            splits.append((train_dates, test_dates))

        test_start_idx = test_end_idx

    return splits


def _compute_dampened_class_weights(y: pd.Series, dampen_exponent: float = CLASS_WEIGHT_DAMPEN_EXPONENT):
    present_classes = np.sort(y.unique())
    raw_weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=y.values)
    raw_map = dict(zip(present_classes.tolist(), raw_weights.tolist()))
    dampened_weights = raw_weights ** dampen_exponent
    dampened_map = dict(zip(present_classes.tolist(), dampened_weights.tolist()))
    return raw_map, dampened_map


def _ordinal_macro_penalty(preds, train_data):
    n_classes = 3
    y_true = train_data.get_label().astype(int)
    preds = np.asarray(preds).reshape(n_classes, -1).T
    pred_class = np.argmax(preds, axis=1)
    ordinal_dist = float(np.abs(pred_class - y_true).mean())
    return 'ordinal_dist', ordinal_dist, False


def train_lightgbm_with_purged_cv():
    df = load_and_combine_features()

    live_mask = df['label'].isna()
    train_val_set = df[~live_mask].reset_index(drop=True)
    live_set = df[live_mask].reset_index(drop=True)

    X_full = train_val_set[FEATURE_COLS]
    y_full = train_val_set['label'].astype(int)
    dates_full = train_val_set['jalali_date']

    logger.info(f" Training dataset: {X_full.shape[0]} rows | Live dataset: {live_set.shape[0]} rows")
    logger.info(f" Overall dataset class distribution: {y_full.value_counts(normalize=True).sort_index().round(3).to_dict()}")

    raw_class_weight_map, _ = _compute_dampened_class_weights(y_full)

    unique_dates = dates_full.unique()
    splits = purged_walk_forward_splits(unique_dates)

    if not splits:
        raise RuntimeError(" No valid folds generated.")

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': REG_ALPHA_L1,
        'lambda_l2': REG_LAMBDA_L2,
        'min_child_samples': MIN_CHILD_SAMPLES,
        'extra_trees': USE_EXTRA_TREES,
        'verbose': -1,
        'random_state': 42,
    }

    models, fold_metrics, fold_importances = [], [], []

    for fold_idx, (train_dates, test_dates) in enumerate(splits, 1):
        train_mask = dates_full.isin(train_dates)
        test_mask = dates_full.isin(test_dates)

        X_train, y_train = X_full[train_mask], y_full[train_mask]
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        _, fold_dampened_map = _compute_dampened_class_weights(y_train)
        w_train = y_train.map(fold_dampened_map).to_numpy()
        w_test = y_test.map(fold_dampened_map).to_numpy()

        fold_class_dist = y_train.value_counts(normalize=True).sort_index().to_dict()
        logger.info(f"Fold {fold_idx} train class distribution: "
                    f"{ {k: round(v, 3) for k, v in fold_class_dist.items()} }")

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, weight=w_test)

        model = lgb.train(
            params, train_data, num_boost_round=1500,
            valid_sets=[test_data],
            feval=_ordinal_macro_penalty,
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )

        preds = model.predict(X_test)
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, pred_labels)
        f1_macro = f1_score(y_test, pred_labels, average='macro')
        ordinal_dist = float(np.abs(pred_labels - y_test.to_numpy()).mean())

        logger.info(f" Fold {fold_idx}: Acc={acc:.2%} | F1={f1_macro:.3f} | "
                    f"OrdinalDist={ordinal_dist:.3f} | best_iter={model.best_iteration}")

        models.append(model)
        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'ordinal_distance': ordinal_dist,
            'best_iteration': model.best_iteration or 0,
            'train_class_distribution': fold_class_dist,
        })
        fold_importances.append(
            pd.Series(model.feature_importance(importance_type='gain'), index=FEATURE_COLS)
        )

    metrics_df = pd.DataFrame(fold_metrics)

    if fold_importances:
        importance_matrix = pd.concat(fold_importances, axis=1)
        importance_matrix.columns = [f"fold_{i+1}" for i in range(len(fold_importances))]
        imp_mean = importance_matrix.mean(axis=1)
        imp_std = importance_matrix.std(axis=1)
        imp_cv = (imp_std / (imp_mean + 1e-9)).sort_values(ascending=False)
        unstable_features = imp_cv[(imp_mean > imp_mean.median()) & (imp_cv > 1.0)]
        if not unstable_features.empty:
            logger.warning(
                " Features with high gain but high variance across folds "
                f"(suspected non-stationarity/leakage): {unstable_features.round(2).to_dict()}"
            )


    healthy_iters = [
    m['best_iteration']
    for m in fold_metrics
    if m['best_iteration'] > 5
]

    if not healthy_iters:
        raise RuntimeError(
            "No healthy fold found: all best_iteration values <= 5. "
            "Training stopped because no reliable boosting depth was identified."
        )

    median_best_iter = int(np.median(healthy_iters))

    logger.info(
        f"Training final model with {median_best_iter} trees "
        f"(median best_iteration across healthy folds)..."
    )

    final_sample_weight_full = y_full.map(raw_class_weight_map).to_numpy()

    if USE_RECENCY_WEIGHTING:
        sorted_unique_dates = np.sort(dates_full.unique())
        date_rank_map = {d: i for i, d in enumerate(sorted_unique_dates)}
        date_ranks = dates_full.map(date_rank_map).to_numpy()
        max_rank = date_ranks.max()

        recency_weight = np.exp(
            -np.log(2) * (max_rank - date_ranks) / RECENCY_HALF_LIFE_DATES
        )

        final_sample_weight_full = (
            final_sample_weight_full * recency_weight
        )

        logger.info(
            f"Recency weighting enabled "
            f"(half-life = {RECENCY_HALF_LIFE_DATES} trading dates)."
        )

    final_data = lgb.Dataset(
        X_full,
        label=y_full,
        weight=final_sample_weight_full
    )

    final_model = lgb.train(
        params,
        final_data,
        num_boost_round=median_best_iter
    )

    """healthy_iters = [m['best_iteration'] for m in fold_metrics if m['best_iteration'] > 5]
    avg_best_iter = int(np.median(healthy_iters)) if healthy_iters else 100
    if not healthy_iters:
        logger.warning(
            " No fold achieved best_iteration > 5. Signal is indistinguishable from noise. "
            "Resolve this before architectural changes."
        )
    else:
        n_weak = sum(1 for m in fold_metrics if m['best_iteration'] <= 5)
        if n_weak > 0:
            logger.warning(f" {n_weak} out of {len(fold_metrics)} folds had best_iteration <= 5 "
                           "(weak signal in specific regimes).")

    logger.info(f" Training final model with {avg_best_iter} trees (Ideal median)...")

    final_sample_weight_full = y_full.map(raw_class_weight_map).to_numpy()

    if USE_RECENCY_WEIGHTING:
        sorted_unique_dates = np.sort(dates_full.unique())
        date_rank_map = {d: i for i, d in enumerate(sorted_unique_dates)}
        date_ranks = dates_full.map(date_rank_map).to_numpy()
        max_rank = date_ranks.max()
        recency_weight = np.exp(-np.log(2) * (max_rank - date_ranks) / RECENCY_HALF_LIFE_DATES)
        final_sample_weight_full = final_sample_weight_full * recency_weight
        logger.info(f" Recency weighting enabled (half-life = {RECENCY_HALF_LIFE_DATES} trading dates).")

    final_data = lgb.Dataset(X_full, label=y_full, weight=final_sample_weight_full)
    final_model = lgb.train(params, final_data, num_boost_round=max(avg_best_iter, 50))"""

    model_path = os.path.join(MODEL_DIR, "lgb_robo_advisor.txt")
    final_model.save_model(model_path)

    def convert_numpy_types(obj):
        if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)): return [convert_numpy_types(i) for i in obj]
        elif isinstance(obj, (np.bool_, bool)): return bool(obj)
        elif isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    metadata = convert_numpy_types({
        'trained_at': datetime.now().isoformat(),
        'num_features': len(FEATURE_COLS),
        'feature_cols': FEATURE_COLS,
        'train_rows': len(X_full),
        'median_best_iter': median_best_iter,
        'params': params,
        'use_recency_weighting': USE_RECENCY_WEIGHTING,
        'recency_half_life_dates': RECENCY_HALF_LIFE_DATES,
        'use_cross_sectional_demeaned_label': USE_CROSS_SECTIONAL_DEMEANED_LABEL,
        'fold_metrics': fold_metrics,
    })

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    logger.info(f"\n Top 10 Feature Importances:\n{importance.head(10).to_string(index=False)}")
    return final_model, metrics_df


if __name__ == "__main__":
    train_lightgbm_with_purged_cv()