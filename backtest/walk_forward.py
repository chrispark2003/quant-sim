"""Walk-forward validation with purged k-fold support, to guard against
lookahead bias and overlapping-window leakage between train and test splits.

*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk.metrics import full_metrics_report


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_folds(
    index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int, purge_days: int = 0,
) -> list[Fold]:
    """Rolling-window folds: [train_start, train_end) -> purge gap -> [test_start, test_end).
    The purge gap drops observations immediately adjacent to the train/test
    boundary so features computed with trailing windows (e.g. a 20-day
    rolling mean) can't leak information across the split."""
    if index.empty:
        return []
    start = index.min()
    end = index.max()
    folds = []
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=purge_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        cursor = cursor + pd.Timedelta(days=step_days)

    return folds


def purged_kfold_splits(index: pd.DatetimeIndex, n_splits: int, purge_days: int = 2) -> list[Fold]:
    """Non-overlapping k-fold splits across the sample, each with a purge
    gap carved out of the training set on both sides of the test fold to
    prevent leakage from overlapping feature windows."""
    if index.empty or n_splits < 2:
        return []
    start, end = index.min(), index.max()
    total_days = (end - start).days
    fold_days = total_days // n_splits

    folds = []
    for i in range(n_splits):
        test_start = start + pd.Timedelta(days=i * fold_days)
        test_end = start + pd.Timedelta(days=(i + 1) * fold_days)
        train_start = test_start + pd.Timedelta(days=purge_days)
        train_end = test_end - pd.Timedelta(days=purge_days)
        folds.append(Fold(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
    return folds


class WalkForwardValidator:
    def __init__(self, config: dict):
        self.train_days = config.get("train_days", 180)
        self.test_days = config.get("test_days", 45)
        self.step_days = config.get("step_days", 45)
        self.purge_days = config.get("purge_days", 2)

    def run(self, backtest_fn, features: pd.DataFrame, signal_scores: pd.Series) -> pd.DataFrame:
        """`backtest_fn(features_slice, scores_slice) -> {'equity_curve': pd.Series}`
        is invoked once per out-of-sample test window. Returns a DataFrame of
        per-fold OOS performance metrics."""
        folds = generate_walk_forward_folds(features.index, self.train_days, self.test_days, self.step_days, self.purge_days)
        rows = []
        for i, fold in enumerate(folds):
            test_mask = (features.index >= fold.test_start) & (features.index < fold.test_end)
            test_features = features.loc[test_mask]
            test_scores = signal_scores.reindex(test_features.index).fillna(0.0)
            if test_features.empty:
                continue

            result = backtest_fn(test_features, test_scores)
            equity_curve = result["equity_curve"]
            metrics = full_metrics_report(equity_curve)
            metrics.update({
                "fold": i,
                "train_start": fold.train_start, "train_end": fold.train_end,
                "test_start": fold.test_start, "test_end": fold.test_end,
                "n_bars": len(test_features),
            })
            rows.append(metrics)

        return pd.DataFrame(rows)
