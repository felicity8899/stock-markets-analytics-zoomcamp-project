"""Walk-forward validation (Phase 3).

Why this module exists
----------------------
The standard pipeline trains once on the first 70% of weeks and tests on
the last 30%. That single train/test split has two weaknesses a course
reviewer (and any practitioner) will spot immediately:

1. **One window, one regime.** The test window is a single stretch of
   history (e.g. Sep 2025 -- Nov 2026). A strategy can look brilliant in
   one regime and fail in another; a single window cannot tell you which
   one you have. It is the backtesting equivalent of judging a fund by
   one year's return.
2. **One training set, one model.** The model is trained on data that
   ends months before the last test week. In live trading you would
   retrain periodically as new data arrives; a model frozen in 2025 is
   not the model you would actually trade in 2026. Walk-forward
   retraining measures the *process* (retrain -> trade -> retrain),
   which is what you would really run.

Walk-forward validation (also called rolling-origin evaluation) fixes
both: split the history into consecutive folds; for each fold, train on
all data *before* the fold and test on the fold's weeks; then roll
forward. Each fold is a fresh out-of-sample test with a freshly trained
model, and the spread of per-fold results is the honest measure of
stability.

Window choice: **expanding** (train on everything before the fold).
Rationale: more training data helps the models, and it matches how a
live strategy would actually be run (you would not throw away old data
unless you believed the market had structurally changed). A rolling
window is the alternative if you suspect regime change; it is not
implemented here, but ``walk_forward_folds`` documents where the switch
would go (train start = fold start - train window).

LSTM note: walk-forward retrains every model once per fold, and an LSTM
fit is ~10x slower than a tree fit. The walk-forward run therefore
covers RF / XGBoost / FF only, and LSTM is reported as
walk-forward-untested (its single-split AUC is still reported by the
standard run). This is a runtime choice, not a methodological one.

Usage: ``python main.py --walk-forward --years 8`` (8 years of history
gives enough folds; the default 3 years is too short -- see
``run_walk_forward``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import models as models_mod
from . import simulate as sim_mod


def walk_forward_folds(
    weeks: list[pd.Timestamp],
    initial_train_weeks: int = 104,
    fold_weeks: int = 13,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate (train_end, test_end) cutoffs for expanding walk-forward.

    Parameters
    ----------
    weeks : sorted list of distinct week dates in the panel.
    initial_train_weeks : number of weeks in the first training set
        (default 104 = 2 years -- enough history for the 12-week
        momentum / 26-week HRP lookbacks to be meaningful).
    fold_weeks : weeks per out-of-sample fold (default 13 = one
        quarter, matching a realistic quarterly retraining cadence).

    Returns
    -------
    List of (train_end, test_end) timestamp pairs. For fold ``i``,
    train = all weeks ``< train_end``, test = weeks in
    ``[train_end, test_end)``. The last fold is dropped if it would
    have fewer than ``fold_weeks`` weeks (a partial fold would not be
    comparable with the full ones). If the last full fold ends exactly
    at the final week, its test_end is set one week past the end so the
    half-open interval still captures all ``fold_weeks`` weeks.
    """
    n = len(weeks)
    folds = []
    start = initial_train_weeks
    while start + fold_weeks <= n:
        if start + fold_weeks < n:
            test_end = weeks[start + fold_weeks]
        else:
            # Last full fold ends at the final week: push the exclusive
            # boundary one week past the end.
            test_end = weeks[-1] + pd.Timedelta(weeks=1)
        folds.append((weeks[start], test_end))
        start += fold_weeks
    return folds


def run_walk_forward(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 5,
    initial_train_weeks: int = 104,
    fold_weeks: int = 13,
    use_xgb: bool = True,
) -> dict:
    """Run walk-forward validation over the panel.

    For each fold: train RF (+ XGBoost, FF) on all data before the fold,
    score the fold's weeks, record per-fold ROC AUC, and simulate
    Strategy A (RF top-N equal-weight) on the fold's weeks. The per-fold
    weekly returns are concatenated into one out-of-sample series, so
    the *pooled* strategy metrics are also reported (the pooled series
    is the return of the retraining process itself, not of any single
    model).

    Also reports Strategy A grouped by calendar year (multi-window
    stability) and the SPY benchmark on the same weeks, for the same
    grouping.

    Returns a dict with the per-fold AUC table, the per-fold strategy
    table, pooled metrics, per-year tables, and the concatenated weekly
    out-of-sample returns.
    """
    weeks = sorted(pd.to_datetime(panel["date"].unique()))
    folds = walk_forward_folds(weeks, initial_train_weeks, fold_weeks)
    if not folds:
        raise ValueError(
            f"Not enough weeks ({len(weeks)}) for walk-forward with "
            f"{initial_train_weeks} initial train weeks and {fold_weeks}-"
            f"week folds. Use more history (e.g. --years 8)."
        )
    print(f"\nWalk-forward: {len(weeks)} weeks, "
          f"{initial_train_weeks} initial train weeks, {fold_weeks}-week "
          f"folds -> {len(folds)} folds, expanding window.")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    fold_auc_rows: list[dict] = []
    fold_strategy_rows: list[dict] = []
    oos_weekly: list[pd.DataFrame] = []

    for i, (train_end, test_end) in enumerate(folds, start=1):
        train = panel[panel["date"] < train_end]
        fold = panel[(panel["date"] >= train_end)
                     & (panel["date"] < test_end)]
        label = f"fold {i}"

        # --- Train + score each model on this fold -------------------
        rf = models_mod.train_random_forest(
            train[feature_cols], train["target"])
        fold_rf = models_mod.add_scores(
            fold, rf, models_mod.score_random_forest, feature_cols)
        auc_row = {
            "fold": label,
            "train_weeks": train["date"].nunique(),
            "test_weeks": fold["date"].nunique(),
            "test_start": train_end.date(),
            "test_end": (test_end - pd.Timedelta(weeks=1)).date(),
            "RF AUC": roc_auc_score(fold["target"], fold_rf["score"]),
        }

        if use_xgb and models_mod.xgboost_available():
            xgb = models_mod.train_xgboost(
                train[feature_cols], train["target"])
            fold_xgb = models_mod.add_scores(
                fold, xgb, models_mod.score_xgboost, feature_cols)
            auc_row["XGB AUC"] = roc_auc_score(
                fold["target"], fold_xgb["score"])
        ff = models_mod.train_factor_regression(
            train[feature_cols], train["next_ret"])
        fold_ff = models_mod.add_scores(
            fold, ff, models_mod.score_factor_regression, feature_cols)
        # FF outputs predicted returns, not probabilities, but AUC only
        # needs a ranking, so it is comparable as a ranking metric.
        auc_row["FF AUC"] = roc_auc_score(
            fold["target"], fold_ff["score"])
        fold_auc_rows.append(auc_row)

        # --- Simulate Strategy A (RF top-N equal-weight) on the fold --
        m_a, weekly_a = sim_mod.simulate(
            fold_rf, prices, top_n=top_n, strategy="equal")
        m_a["fold"] = label
        m_a["test_start"] = train_end.date()
        m_a["test_end"] = (test_end - pd.Timedelta(weeks=1)).date()
        fold_strategy_rows.append(m_a)
        weekly_a = weekly_a.copy()
        weekly_a["fold"] = label
        oos_weekly.append(weekly_a)

        print(f"  {label}: {train_end.date()} -> "
              f"{(test_end - pd.Timedelta(weeks=1)).date()} | "
              f"RF AUC {auc_row['RF AUC']:.3f}"
              + (f" | XGB AUC {auc_row['XGB AUC']:.3f}"
                 if "XGB AUC" in auc_row else "")
              + f" | FF AUC {auc_row['FF AUC']:.3f} | "
              f"Strategy A {m_a['total_return']:+.1%}")

    auc_table = pd.DataFrame(fold_auc_rows)
    strategy_table = pd.DataFrame(fold_strategy_rows)
    oos = pd.concat(oos_weekly, ignore_index=True)

    # --- Pooled out-of-sample metrics ---------------------------------
    # The pooled series is the return of the *retraining process*:
    # each week's return comes from the model trained on all data
    # before that week. It is the single most honest headline number
    # in the project.
    pooled = sim_mod._metrics(oos["ret"], oos["turnover"])
    pooled["strategy"] = "A (walk-forward pooled)"

    # --- Multi-window stability: Strategy A by calendar year ----------
    oos["year"] = pd.to_datetime(oos["date"]).dt.year
    per_year = sim_mod.grouped_metrics(oos, "year")

    # Same grouping for the benchmark, on the same weeks.
    bench = sim_mod.benchmark_weekly(spy, oos["date"])
    bench["year"] = pd.to_datetime(bench["date"]).dt.year
    bench_year = sim_mod.grouped_metrics(bench, "year")

    # --- Report --------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n=== Walk-forward per-fold ROC AUC ===")
    print(auc_table.to_string(index=False,
                              float_format=lambda v: f"{v:,.3f}"))
    auc_cols = [c for c in auc_table.columns if c.endswith("AUC")]
    print("\nMean AUC across folds:")
    for c in auc_cols:
        print(f"  {c}: {auc_table[c].mean():.3f} "
              f"(min {auc_table[c].min():.3f}, "
              f"max {auc_table[c].max():.3f}, "
              f"std {auc_table[c].std():.3f})")

    print("\n=== Walk-forward per-fold Strategy A (RF top-5 equal) ===")
    show = strategy_table[["fold", "test_start", "test_end",
                           "total_return", "cagr", "sharpe",
                           "max_drawdown", "win_rate"]]
    print(show.to_string(index=False,
                         float_format=lambda v: f"{v:,.4f}"))
    print(f"\nPooled out-of-sample Strategy A: "
          f"total {pooled['total_return']:+.2%} over "
          f"{len(oos)} weeks, CAGR {pooled['cagr']:+.2%}, "
          f"Sharpe {pooled['sharpe']:.2f}, "
          f"max DD {pooled['max_drawdown']:.2%}, "
          f"win rate {pooled['win_rate']:.2%}")

    print("\n=== Strategy A by calendar year (multi-window) ===")
    print(per_year.to_string(index=False,
                             float_format=lambda v: f"{v:,.4f}"))
    print("\n=== SPY buy-and-hold by calendar year (same weeks) ===")
    print(bench_year.to_string(index=False,
                               float_format=lambda v: f"{v:,.4f}"))

    return {
        "auc_table": auc_table,
        "strategy_table": strategy_table,
        "pooled_metrics": pooled,
        "per_year": per_year,
        "benchmark_per_year": bench_year,
        "oos_weekly": oos,
    }
