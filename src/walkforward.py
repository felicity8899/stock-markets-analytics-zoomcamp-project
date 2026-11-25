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

Phase 5 (volatility-adjusted ranking): the same walk-forward loop also
simulates Strategy A with the top-N selected by ``score / vol`` instead
of raw score (``score_vol4`` = score / 4-week realised vol,
``score_vol12`` = score / 12-week realised vol), and with a **vol cap**
(top 10% most volatile names excluded each week, then ranked by raw
score), and with **inverse-vol sizing** (same top-5 selection as raw,
but each holding weighted by 1 / vol_4w instead of 20% each). The
model itself is unchanged -- only the *selection* or *sizing* rule
changes, so these are tests of the strategy layer, not the model. The
motivation is the MU finding (2026-11-25): the RF persistently ranks
the highest-volatility stock #1, because high vol makes a stock likely
to land in the top return quintile, but the realised next-week returns
on those picks were negative on average in the recent window. Dividing
by vol asks the portfolio to prefer "high score *for its volatility*"
instead of "high score, full stop"; the cap asks it to simply skip the
extreme names without rewarding the calm ones; inverse-vol sizing asks
it to keep the model's picks but shrink the riskiest positions. MU pick
statistics (how often MU was selected, and its realised next-week
return when selected) are reported per variant, so the question "did
the vol adjustment fix the MU false picks?" is answered directly rather
than inferred from the pooled return; for the sizing variant, MU's
weight when held and its return contribution (weight x next-week
return, vs the 20% equal-weight contribution) are reported instead,
because sizing changes how much MU matters, not whether it is picked.
Pooled metrics also report the **Sortino ratio**
(Phase 5): like Sharpe, but the denominator is downside deviation
only, so upside jumps are not penalised -- the right risk-adjusted
lens for a strategy whose edge is catching large up weeks.

Usage: ``python main.py --walk-forward --years 8`` (8 years of history
gives enough folds; the default 3 years is too short -- see
``run_walk_forward``).
"""

from __future__ import annotations

import numpy as np
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
    exposure: pd.Series | None = None,
    exposure_band: pd.Series | None = None,
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

    Phase 4 regime filter: if ``exposure`` is given (a weekly 1.0/0.0
    series from ``src.regime``, aligned to the panel's week dates),
    Strategy A is additionally simulated with the filter applied, and
    the filtered pooled metrics, per-year table, whipsaw count, and %
    of weeks invested are reported side by side with the unfiltered
    numbers. ``exposure_band`` (the 1% hysteresis-band variant) is
    reported the same way when given. The filter is point-in-time
    correct by construction (each week's exposure uses only closes
    through that week's Friday -- see ``src.regime``), so using the
    full-period series inside walk-forward introduces no lookahead:
    no fold's exposure depends on data after that fold's weeks.

    Returns a dict with the per-fold AUC table, the per-fold strategy
    table, pooled metrics, per-year tables, and the concatenated weekly
    out-of-sample returns (plus the filtered equivalents when an
    exposure series is given).
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
    # Phase 4: filtered Strategy A series (binary filter + band variant).
    oos_weekly_filt: list[pd.DataFrame] = []
    oos_weekly_band: list[pd.DataFrame] = []
    # Phase 5: volatility-adjusted ranking variants (same RF scores,
    # top-N selected by score / vol instead of raw score).
    oos_weekly_vol4: list[pd.DataFrame] = []
    oos_weekly_vol12: list[pd.DataFrame] = []
    # Phase 5: vol-cap variant -- same RF scores, top-N by raw score
    # after excluding the week's top 10% most volatile names.
    oos_weekly_volcap: list[pd.DataFrame] = []
    # Phase 5: inverse-vol sizing variant -- same top-5 selection as
    # raw, weights by 1 / vol_4w.
    oos_weekly_invvol: list[pd.DataFrame] = []
    # Phase 5: MU weight diagnostics for the sizing variant -- one row
    # per OOS week MU was in the top-5: its inverse-vol weight and its
    # return contribution (weight x next-week return) vs the 20%
    # equal-weight contribution on the same week.
    mu_invvol_rows: list[dict] = []
    # Phase 5: MU pick diagnostics -- one row per OOS week per variant:
    # MU's rank within the week under that variant's ranking, and its
    # realised next-week return. A "pick" is rank <= top_n.
    mu_pick_rows: list[dict] = []

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

        # --- Phase 5: same RF scores, top-N by score / vol -----------
        # The ranking columns are point-in-time correct: vol_4w /
        # vol_12w use only returns through the week's Friday close,
        # exactly like every other feature. A zero vol (a stock that
        # did not move for 4 weeks) would divide by zero, so zeros are
        # treated as missing -- such a stock ranks last under nlargest.
        fold_rf["score_vol4"] = (
            fold_rf["score"]
            / fold_rf["vol_4w"].replace(0, np.nan)
        )
        fold_rf["score_vol12"] = (
            fold_rf["score"]
            / fold_rf["vol_12w"].replace(0, np.nan)
        )
        for col, bucket in (("score_vol4", oos_weekly_vol4),
                            ("score_vol12", oos_weekly_vol12)):
            _, weekly_v = sim_mod.simulate(
                fold_rf, prices, top_n=top_n, strategy="equal",
                rank_col=col)
            weekly_v = weekly_v.copy()
            weekly_v["fold"] = label
            bucket.append(weekly_v)

        # --- Phase 5: vol-cap variant (raw score, top-vol decile out) -
        _, weekly_c = sim_mod.simulate(
            fold_rf, prices, top_n=top_n, strategy="equal",
            vol_cap_quantile=0.9, vol_col="vol_4w")
        weekly_c = weekly_c.copy()
        weekly_c["fold"] = label
        oos_weekly_volcap.append(weekly_c)

        # --- Phase 5: inverse-vol sizing variant ----------------------
        _, weekly_iv = sim_mod.simulate(
            fold_rf, prices, top_n=top_n, strategy="invvol",
            vol_col="vol_4w")
        weekly_iv = weekly_iv.copy()
        weekly_iv["fold"] = label
        oos_weekly_invvol.append(weekly_iv)

        # --- Phase 5: MU weight diagnostics (sizing variant) ----------
        # Sizing does not change whether MU is picked -- it changes how
        # much it weighs. Record MU's inverse-vol weight on the weeks
        # it was in the top-5, and its return contribution under both
        # weightings on the same week, so the comparison is apples to
        # apples: same picks, same weeks, only the weight differs.
        for wk_date, wk in fold_rf.groupby("date", sort=True):
            top5 = wk.nlargest(top_n, "score")
            w = sim_mod._invvol_weights(top5, "vol_4w")
            mu_row = top5[top5["ticker"] == "MU"]
            if not mu_row.empty:
                next_ret = float(mu_row["next_ret"].iloc[0])
                mu_w = float(w["MU"])
                mu_invvol_rows.append({
                    "date": wk_date,
                    "weight": mu_w,
                    "next_ret": next_ret,
                    "contrib_invvol": mu_w * next_ret,
                    "contrib_equal": (1.0 / top_n) * next_ret,
                })

        # --- Phase 5: MU pick diagnostics -----------------------------
        # For each ranking rule, MU's rank within each week of this
        # fold and its realised next-week return. The comparison that
        # matters is not "was MU picked less often" in the abstract,
        # but "when it was picked, did it lose less / win more".
        for variant, col in (("raw", "score"),
                             ("vol4", "score_vol4"),
                             ("vol12", "score_vol12")):
            ranked = fold_rf.copy()
            ranked["rank_in_week"] = (
                ranked.groupby("date")[col]
                .rank(ascending=False, method="first")
            )
            mu_weeks = ranked[ranked["ticker"] == "MU"]
            for _, r in mu_weeks.iterrows():
                mu_pick_rows.append({
                    "variant": variant,
                    "date": r["date"],
                    "rank_in_week": float(r["rank_in_week"]),
                    "picked": bool(r["rank_in_week"] <= top_n),
                    "next_ret": float(r["next_ret"]),
                })

        # Vol-cap variant: MU is picked only if it survives the cap
        # (vol_4w at or below the week's 90th percentile) AND ranks
        # top-N by raw score within the capped set. Weeks where the cap
        # removes MU have no rank -- they count as not picked.
        wk = fold_rf.copy()
        thr = wk.groupby("date")["vol_4w"].transform(
            lambda s: s.quantile(0.9))
        wk["capped_out"] = wk["vol_4w"] > thr
        capped = wk[~wk["capped_out"]].copy()
        capped["rank_in_week"] = (
            capped.groupby("date")["score"]
            .rank(ascending=False, method="first")
        )
        mu_all = wk[wk["ticker"] == "MU"][["date", "next_ret"]]
        mu_ranked = capped[capped["ticker"] == "MU"][
            ["date", "rank_in_week"]]
        mu_merged = mu_all.merge(mu_ranked, on="date", how="left")
        for _, r in mu_merged.iterrows():
            rank = r["rank_in_week"]
            mu_pick_rows.append({
                "variant": "volcap",
                "date": r["date"],
                "rank_in_week": float(rank) if pd.notna(rank) else np.nan,
                "picked": bool(pd.notna(rank) and rank <= top_n),
                "next_ret": float(r["next_ret"]),
            })

        # --- Phase 4: same fold with the regime filter applied -------
        # The model training above is shared; only the cheap weekly
        # simulation is re-run with the exposure series.
        filt_note = ""
        if exposure is not None:
            _, weekly_f = sim_mod.simulate(
                fold_rf, prices, top_n=top_n, strategy="equal",
                exposure=exposure)
            weekly_f = weekly_f.copy()
            weekly_f["fold"] = label
            oos_weekly_filt.append(weekly_f)
            filt_note = (f" | filtered "
                         f"{sim_mod._metrics(weekly_f['ret'], weekly_f['turnover'])['total_return']:+.1%}")
        if exposure_band is not None:
            _, weekly_b = sim_mod.simulate(
                fold_rf, prices, top_n=top_n, strategy="equal",
                exposure=exposure_band)
            weekly_b = weekly_b.copy()
            weekly_b["fold"] = label
            oos_weekly_band.append(weekly_b)

        print(f"  {label}: {train_end.date()} -> "
              f"{(test_end - pd.Timedelta(weeks=1)).date()} | "
              f"RF AUC {auc_row['RF AUC']:.3f}"
              + (f" | XGB AUC {auc_row['XGB AUC']:.3f}"
                 if "XGB AUC" in auc_row else "")
              + f" | FF AUC {auc_row['FF AUC']:.3f} | "
              f"Strategy A {m_a['total_return']:+.1%}{filt_note}")

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

    result = {
        "auc_table": auc_table,
        "strategy_table": strategy_table,
        "pooled_metrics": pooled,
        "per_year": per_year,
        "benchmark_per_year": bench_year,
        "oos_weekly": oos,
    }

    # --- Phase 4: regime-filtered comparison --------------------------
    if exposure is not None and oos_weekly_filt:
        oos_f = pd.concat(oos_weekly_filt, ignore_index=True)
        pooled_f = sim_mod._metrics(oos_f["ret"], oos_f["turnover"])
        pooled_f["strategy"] = "A (walk-forward pooled, regime-filtered)"
        oos_f["year"] = pd.to_datetime(oos_f["date"]).dt.year
        per_year_f = sim_mod.grouped_metrics(oos_f, "year")

        # Whipsaw count: exposure switches (in->out or out->in) across
        # the out-of-sample weeks. Each switch trades the full
        # portfolio once (5 bps), so the count is also the switching
        # cost in disguise: switches x 5 bps of extra turnover cost.
        exp_series = oos_f.sort_values("date")["exposure"]
        switches = int((exp_series.diff().fillna(0.0) != 0).sum())
        pct_invested = float(exp_series.mean()) * 100.0
        # Weeks invested by calendar year (where was the filter out?).
        invested_by_year = (
            oos_f.groupby("year")["exposure"].mean() * 100.0
        ).round(1).to_dict()
        # Win rate over invested weeks only. The pooled win rate counts
        # cash weeks (0% return) as non-wins, which mechanically lowers
        # it for the filtered variant -- the invested-weeks win rate is
        # the comparable number.
        invested_weeks = oos_f[oos_f["exposure"] > 0]
        win_rate_invested = float(
            (invested_weeks["ret"] > 0).mean()
        ) if len(invested_weeks) else 0.0
        filter_stats = {
            "switches": switches,
            "pct_weeks_invested": pct_invested,
            "pct_invested_by_year": invested_by_year,
            "win_rate_invested_weeks": win_rate_invested,
        }

        print("\n=== Phase 4: Strategy A with the 200-day MA regime filter ===")
        print(f"Filter: invested {pct_invested:.1f}% of the "
              f"{len(oos_f)} out-of-sample weeks, "
              f"{switches} exposure switches (whipsaws).")
        print("Invested % by year: "
              + ", ".join(f"{y}: {v:.0f}%" for y, v in invested_by_year.items()))
        print(f"Win rate on invested weeks only: "
              f"{win_rate_invested:.1%} (the pooled win rate counts "
              f"cash weeks as non-wins -- see LEARNING_NOTES.md).")
        comp = pd.DataFrame([
            {"variant": "A unfiltered", **pooled},
            {"variant": "A + 200d MA filter", **pooled_f},
        ])
        show_cols = ["variant", "weeks", "total_return", "cagr",
                     "sharpe", "max_drawdown", "avg_turnover", "win_rate"]
        print("\nPooled out-of-sample, filtered vs unfiltered:")
        print(comp[show_cols].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))
        print("\n=== Filtered Strategy A by calendar year ===")
        print(per_year_f.to_string(index=False,
                                   float_format=lambda v: f"{v:,.4f}"))

        result.update({
            "pooled_metrics_filtered": pooled_f,
            "per_year_filtered": per_year_f,
            "oos_weekly_filtered": oos_f,
            "filter_stats": filter_stats,
        })

        if exposure_band is not None and oos_weekly_band:
            oos_b = pd.concat(oos_weekly_band, ignore_index=True)
            pooled_b = sim_mod._metrics(oos_b["ret"], oos_b["turnover"])
            pooled_b["strategy"] = ("A (walk-forward pooled, "
                                    "regime filter, 1% band)")
            exp_b = oos_b.sort_values("date")["exposure"]
            switches_b = int((exp_b.diff().fillna(0.0) != 0).sum())
            print("\n--- 1% hysteresis-band variant ---")
            print(f"Band filter: invested "
                  f"{float(exp_b.mean()) * 100.0:.1f}% of weeks, "
                  f"{switches_b} switches (vs {switches} for the plain "
                  f"filter), total {pooled_b['total_return']:+.2%}, "
                  f"CAGR {pooled_b['cagr']:+.2%}, "
                  f"Sharpe {pooled_b['sharpe']:.2f}, "
                  f"max DD {pooled_b['max_drawdown']:.2%}.")
            result.update({
                "pooled_metrics_filtered_band": pooled_b,
                "oos_weekly_filtered_band": oos_b,
                "filter_stats_band": {
                    "switches": switches_b,
                    "pct_weeks_invested": float(exp_b.mean()) * 100.0,
                },
            })

    # --- Phase 5: volatility-adjusted ranking comparison -------------
    oos_v4 = pd.concat(oos_weekly_vol4, ignore_index=True)
    oos_v12 = pd.concat(oos_weekly_vol12, ignore_index=True)
    oos_vc = pd.concat(oos_weekly_volcap, ignore_index=True)
    oos_iv = pd.concat(oos_weekly_invvol, ignore_index=True)
    pooled_v4 = sim_mod._metrics(oos_v4["ret"], oos_v4["turnover"])
    pooled_v4["strategy"] = "A (walk-forward pooled, rank by score/vol4)"
    pooled_v12 = sim_mod._metrics(oos_v12["ret"], oos_v12["turnover"])
    pooled_v12["strategy"] = "A (walk-forward pooled, rank by score/vol12)"
    pooled_vc = sim_mod._metrics(oos_vc["ret"], oos_vc["turnover"])
    pooled_vc["strategy"] = "A (walk-forward pooled, vol cap 90th pct)"
    pooled_iv = sim_mod._metrics(oos_iv["ret"], oos_iv["turnover"])
    pooled_iv["strategy"] = "A (walk-forward pooled, inverse-vol sizing)"
    oos_v4["year"] = pd.to_datetime(oos_v4["date"]).dt.year
    per_year_v4 = sim_mod.grouped_metrics(oos_v4, "year")
    oos_vc["year"] = pd.to_datetime(oos_vc["date"]).dt.year
    per_year_vc = sim_mod.grouped_metrics(oos_vc, "year")
    oos_iv["year"] = pd.to_datetime(oos_iv["date"]).dt.year
    per_year_iv = sim_mod.grouped_metrics(oos_iv, "year")

    # MU pick statistics per variant. "MU mean next_ret (all OOS
    # weeks)" is the reference: what MU did on average regardless of
    # whether the model picked it. If the vol adjustment works, the
    # picked-weeks mean should move toward (or above) that reference
    # -- the raw-score failure was picking MU exactly on its bad weeks.
    mu_df = pd.DataFrame(mu_pick_rows)
    mu_stats_rows = []
    mu_all_mean = float(mu_df["next_ret"].mean())
    for variant in ("raw", "vol4", "vol12", "volcap"):
        sub = mu_df[mu_df["variant"] == variant]
        picked = sub[sub["picked"]]
        mu_stats_rows.append({
            "variant": variant,
            "weeks": len(sub),
            "picked_weeks": int(len(picked)),
            "picked_pct": (len(picked) / len(sub) * 100.0) if len(sub) else 0.0,
            "mean_next_ret_when_picked": (
                float(picked["next_ret"].mean()) if len(picked) else 0.0
            ),
            "win_rate_when_picked": (
                float((picked["next_ret"] > 0).mean())
                if len(picked) else 0.0
            ),
            "best_picked_week": (
                float(picked["next_ret"].max()) if len(picked) else 0.0
            ),
            "worst_picked_week": (
                float(picked["next_ret"].min()) if len(picked) else 0.0
            ),
        })
    mu_stats = pd.DataFrame(mu_stats_rows)

    print("\n=== Phase 5: Strategy A with volatility-adjusted ranking ===")
    print("Same RF scores, same walk-forward folds -- only the top-N "
          "selection or sizing rule changes.")
    comp5 = pd.DataFrame([
        {"variant": "A raw score", **pooled},
        {"variant": "A score/vol4", **pooled_v4},
        {"variant": "A score/vol12", **pooled_v12},
        {"variant": "A vol cap 90th pct", **pooled_vc},
        {"variant": "A inverse-vol sizing", **pooled_iv},
    ])
    show_cols5 = ["variant", "weeks", "total_return", "cagr",
                  "sharpe", "sortino", "max_drawdown", "avg_turnover",
                  "win_rate"]
    print("\nPooled out-of-sample, ranking rule comparison:")
    print(comp5[show_cols5].to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n=== A (score/vol4) by calendar year ===")
    print(per_year_v4.to_string(index=False,
                                float_format=lambda v: f"{v:,.4f}"))
    print("\n=== A (vol cap 90th pct) by calendar year ===")
    print(per_year_vc.to_string(index=False,
                                float_format=lambda v: f"{v:,.4f}"))
    print("\n=== A (inverse-vol sizing) by calendar year ===")
    print(per_year_iv.to_string(index=False,
                                float_format=lambda v: f"{v:,.4f}"))
    print(f"\nMU mean next-week return over ALL {len(mu_df) // 4} "
          f"OOS weeks (any variant): {mu_all_mean:+.2%}")
    print("\n=== MU pick statistics by ranking rule ===")
    print(mu_stats.to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))

    # MU under inverse-vol sizing: same picks as raw, smaller weight.
    mu_iv = pd.DataFrame(mu_invvol_rows)
    mu_iv_summary = {
        "weeks_held": len(mu_iv),
        "mean_weight_when_held": float(mu_iv["weight"].mean()),
        "median_weight_when_held": float(mu_iv["weight"].median()),
        "min_weight_when_held": float(mu_iv["weight"].min()),
        "max_weight_when_held": float(mu_iv["weight"].max()),
        "mean_next_ret_when_held": float(mu_iv["next_ret"].mean()),
        "mean_contrib_invvol": float(mu_iv["contrib_invvol"].mean()),
        "mean_contrib_equal": float(mu_iv["contrib_equal"].mean()),
        "total_contrib_invvol": float(mu_iv["contrib_invvol"].sum()),
        "total_contrib_equal": float(mu_iv["contrib_equal"].sum()),
    }
    print("\n=== MU under inverse-vol sizing (same picks as raw) ===")
    print(f"MU held in {mu_iv_summary['weeks_held']} of 260 weeks; "
          f"mean weight when held "
          f"{mu_iv_summary['mean_weight_when_held']:.1%} "
          f"(median {mu_iv_summary['median_weight_when_held']:.1%}, "
          f"range {mu_iv_summary['min_weight_when_held']:.1%}–"
          f"{mu_iv_summary['max_weight_when_held']:.1%}) "
          f"vs 20.0% equal weight.")
    print(f"Mean next-week return when held: "
          f"{mu_iv_summary['mean_next_ret_when_held']:+.2%}")
    print(f"Mean weekly contribution to portfolio return: "
          f"{mu_iv_summary['mean_contrib_invvol']:+.2%} (invvol) vs "
          f"{mu_iv_summary['mean_contrib_equal']:+.2%} (equal).")
    print(f"Total contribution over all held weeks: "
          f"{mu_iv_summary['total_contrib_invvol']:+.2%} (invvol) vs "
          f"{mu_iv_summary['total_contrib_equal']:+.2%} (equal).")

    result.update({
        "pooled_metrics_vol4": pooled_v4,
        "pooled_metrics_vol12": pooled_v12,
        "pooled_metrics_volcap": pooled_vc,
        "pooled_metrics_invvol": pooled_iv,
        "oos_weekly_vol4": oos_v4,
        "oos_weekly_vol12": oos_v12,
        "oos_weekly_volcap": oos_vc,
        "oos_weekly_invvol": oos_iv,
        "per_year_vol4": per_year_v4,
        "per_year_volcap": per_year_vc,
        "per_year_invvol": per_year_iv,
        "mu_pick_stats": mu_stats,
        "mu_pick_rows": mu_df,
        "mu_mean_next_ret_all_weeks": mu_all_mean,
        "mu_invvol_rows": mu_iv,
        "mu_invvol_summary": mu_iv_summary,
    })

    return result
