"""End-to-end pipeline: download -> features -> train -> predict -> simulate.

Usage (from the project root, venv activated)::

    python main.py                      # test run: 30 tickers, 3 years
    python main.py --tickers 50 --years 5   # fuller run
    python main.py --universe market-cap    # largest stocks first
    python main.py --walk-forward --years 8 # Phase 3 walk-forward validation
    python main.py --borrow-fee 0.05        # stress-test the short leg at 5%/yr

Macro source (Phase 3): FRED (DGS10, VIXCLS, T10Y2Y, DGS2) is used when
the FRED_API_KEY environment variable is set; otherwise the pipeline
falls back to the yfinance ^VIX/^TNX proxies. See README for setup.

Prints a strategy comparison table (RandomForest equal-weight vs HRP vs
Fama-French regression vs XGBoost vs LSTM vs long-short, plus the Phase 3
top-10 variants A10/B10 and the long-short strategy with and without
borrow fees) and the latest week's top-5 trades from a model retrained
on all available data.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data as data_mod
from src import features as feat_mod
from src import lstm_model as lstm_mod
from src import models as models_mod
from src import simulate as sim_mod
from src import walkforward as wf_mod
from src.simulate import BORROW_FEE_ANNUAL
from src.target import make_target, time_split


def _load_panel(
    n_tickers: int,
    years: int,
    universe: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Download data and build the modelling panel.

    Macro source selection (Phase 3): FRED is primary when the
    FRED_API_KEY environment variable is set (DGS10, VIXCLS, T10Y2Y,
    DGS2); otherwise the yfinance ^VIX/^TNX proxies are used. If the
    FRED download fails for any reason, the pipeline falls back to the
    proxies with a printed warning instead of crashing -- a macro-data
    outage should never block the weekly run.

    Returns (feat, panel, prices, spy, feature_cols): ``feat`` is the
    pre-target feature frame (kept because the latest week has no target
    yet and is needed for the live-trade section); ``panel`` is the
    target-labelled frame used for training and backtesting.
    """
    # --- 1. Data -------------------------------------------------------
    tickers = data_mod.get_universe(n=n_tickers, order=universe)
    prices = data_mod.download_prices(
        tickers, years=years, force_refresh=force_refresh)
    french = data_mod.download_french_weekly(force_refresh=force_refresh)
    spy = data_mod.download_benchmark(years=years, force_refresh=force_refresh)

    fred = None
    macro = None
    if os.environ.get("FRED_API_KEY"):
        try:
            fred = data_mod.download_fred(
                years=years, force_refresh=force_refresh)
            print("Macro source: FRED (DGS10, VIXCLS, T10Y2Y, DGS2).")
        except Exception as exc:  # noqa: BLE001 - any FRED failure
            # download_fred's errors are sanitised (no key, no URL), so
            # printing them here is safe.
            print(f"WARNING: FRED download failed ({exc}); falling back "
                  f"to yfinance ^VIX/^TNX proxies.")
    if fred is None:
        macro = data_mod.download_macro(
            years=years, force_refresh=force_refresh)
        print("Macro source: yfinance ^VIX/^TNX proxies "
              "(set FRED_API_KEY to use FRED instead).")

    # --- 2. Features + target ------------------------------------------
    feat = feat_mod.build_features(
        prices, macro=macro, french=french, fred=fred)
    panel = make_target(feat)
    numeric, dummies = feat_mod.feature_columns(panel)
    feature_cols = numeric + dummies
    print(f"Model features: {len(feature_cols)} "
          f"({len(numeric)} numeric + {len(dummies)} dummies)")
    return feat, panel, prices, spy, feature_cols


def run(
    n_tickers: int = 30,
    years: int = 3,
    top_n: int = 5,
    train_frac: float = 0.7,
    use_xgb: bool = True,
    use_lstm: bool = True,
    universe: str = "name",
    force_refresh: bool = False,
    borrow_fee: float = BORROW_FEE_ANNUAL,
) -> dict:
    # --- 1-2. Data + features + target (shared with walk-forward) -----
    feat, panel, prices, spy, feature_cols = _load_panel(
        n_tickers, years, universe, force_refresh)

    # --- 3. Time-based split (never shuffle) ---------------------------
    train, test, cutoff = time_split(panel, train_frac=train_frac)

    # --- 4. Train models -----------------------------------------------
    results: dict[str, dict] = {}

    # 4a. RandomForest (course baseline).
    rf = models_mod.train_random_forest(
        train[feature_cols], train["target"])
    test_rf = models_mod.add_scores(
        test, rf, models_mod.score_random_forest, feature_cols)
    rep = models_mod.classification_report_simple(
        test["target"], test_rf["score"], threshold=models_mod.RF_THRESHOLD)
    print(f"\nRandomForest test: ROC AUC {rep['roc_auc']:.3f}, "
          f"precision@{rep['threshold']} {rep['precision@thr']:.3f}, "
          f"recall@{rep['threshold']} {rep['recall@thr']:.3f}")

    m_a, _ = sim_mod.simulate(
        test_rf, prices, top_n=top_n, strategy="equal")
    results["A: RF top-5 equal-weight"] = m_a
    m_b, _ = sim_mod.simulate(
        test_rf, prices, top_n=top_n, strategy="hrp")
    results["B: RF top-5 HRP-weight"] = m_b

    # 4a1. Top-10 variants (Phase 3): the fairer HRP test. With only 5
    # holdings, HRP's diversification has little room to work -- the
    # weights cannot differ much from equal weight. At 10 holdings the
    # covariance structure has enough breadth for HRP to express a
    # genuinely different portfolio. See LEARNING_NOTES.md section 3.
    m_a10, _ = sim_mod.simulate(
        test_rf, prices, top_n=10, strategy="equal")
    results["A10: RF top-10 equal-weight"] = m_a10
    m_b10, _ = sim_mod.simulate(
        test_rf, prices, top_n=10, strategy="hrp")
    results["B10: RF top-10 HRP-weight"] = m_b10

    # 4a2. Long-short on the same RF scores (Phase 2), reported both
    # WITHOUT borrow fees (Phase 2 convention) and WITH the default
    # borrow fee (Phase 3), so the fee drag is visible side by side.
    m_ls, _ = sim_mod.simulate_long_short(test_rf, top_n=top_n)
    results["F: RF long-short top-5/bottom-5"] = m_ls
    m_ls_fee, _ = sim_mod.simulate_long_short(
        test_rf, top_n=top_n, borrow_fee_annual=borrow_fee)
    results[f"F2: RF long-short, borrow fee {borrow_fee:.0%}/yr"] = m_ls_fee

    # 4b. Fama-French-style factor regression (Ridge on next-week return).
    ff = models_mod.train_factor_regression(
        train[feature_cols], train["next_ret"])
    test_ff = models_mod.add_scores(
        test, ff, models_mod.score_factor_regression, feature_cols)
    m_ff, _ = sim_mod.simulate(
        test_ff, prices, top_n=top_n, strategy="equal")
    results["C: FF-regression top-5 equal-weight"] = m_ff

    # 4c. XGBoost (optional).
    if use_xgb and models_mod.xgboost_available():
        xgb = models_mod.train_xgboost(train[feature_cols], train["target"])
        test_xgb = models_mod.add_scores(
            test, xgb, models_mod.score_xgboost, feature_cols)
        rep_x = models_mod.classification_report_simple(
            test["target"], test_xgb["score"], threshold=0.5)
        print(f"XGBoost test: ROC AUC {rep_x['roc_auc']:.3f}, "
              f"precision@0.5 {rep_x['precision@thr']:.3f}, "
              f"recall@0.5 {rep_x['recall@thr']:.3f}")
        m_x, _ = sim_mod.simulate(
            test_xgb, prices, top_n=top_n, strategy="equal")
        results["D: XGBoost top-5 equal-weight"] = m_x
    elif use_xgb:
        print("XGBoost: package not installed, skipping (see README TODO).")

    # 4d. LSTM comparison model (Phase 2). Sequences are built from the
    # full panel and split by the SAME cutoff as the tree models, so the
    # test AUC is directly comparable.
    if use_lstm and lstm_mod.tensorflow_available():
        X_seq, y_seq, meta = lstm_mod.build_sequences(panel, feature_cols)
        tr_mask = (pd.to_datetime(meta["date"]) < cutoff).to_numpy()
        te_mask = ~tr_mask
        lstm_model, scaler, _, _ = lstm_mod.train_lstm(
            X_seq[tr_mask], y_seq[tr_mask], meta.loc[tr_mask, "date"])
        test_meta = meta[te_mask].copy()
        test_meta["score"] = lstm_mod.score_lstm(
            lstm_model, scaler, X_seq[te_mask])
        rep_l = models_mod.classification_report_simple(
            test_meta["target"], test_meta["score"], threshold=0.5)
        print(f"LSTM test: ROC AUC {rep_l['roc_auc']:.3f}, "
              f"precision@0.5 {rep_l['precision@thr']:.3f}, "
              f"recall@0.5 {rep_l['recall@thr']:.3f}")
        m_l, _ = sim_mod.simulate(
            test_meta, prices, top_n=top_n, strategy="equal")
        results["E: LSTM top-5 equal-weight"] = m_l
    elif use_lstm:
        print("LSTM: tensorflow not installed, skipping "
              "(pip install tensorflow-cpu).")

    # --- 5. Benchmark + report -----------------------------------------
    results["Benchmark: SPY buy-and-hold"] = sim_mod.benchmark_metrics(
        spy, test["date"].unique())
    table = sim_mod.metrics_table(results)
    pd.set_option("display.width", 200)
    print("\n=== Strategy comparison (test period) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # --- 5b. Exact row-by-row simulation of Strategy A (Phase 2) -------
    m_exact, _ = sim_mod.simulate_exact(
        test_rf, prices, top_n=top_n, strategy="equal")
    gap = m_exact["total_return"] - m_a["total_return"]
    print("\n=== Exact simulation check (Strategy A) ===")
    print(f"Vector backtest total return: {m_a['total_return']:+.4%}")
    print(f"Exact simulation total return: {m_exact['total_return']:+.4%}")
    print(f"Gap (exact - vector): {gap:+.4%} "
          f"(expected: tiny negative -- costs are charged before the "
          f"week's return accrues; see README)")
    print(f"Exact: final capital ${m_exact['final_capital']:,.0f} "
          f"from $100,000; total costs ${m_exact['total_costs']:,.0f}")

    # --- 6. Latest trades (model retrained on all data) -----------------
    rf_full = models_mod.train_random_forest(
        panel[feature_cols], panel["target"])
    latest_date = feat["date"].max()
    latest = feat[feat["date"] == latest_date].copy()
    latest["score"] = models_mod.score_random_forest(
        rf_full, latest[feature_cols])
    picks = latest.nlargest(top_n, "score")[
        ["ticker", "close", "score"]]
    print(f"\n=== Latest trades (week of {latest_date.date()}) ===")
    print("Top-5 by RandomForest P(top-quintile), retrained on all data:")
    print(picks.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    return {"metrics": results, "latest_picks": picks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=int, default=30,
                        help="universe size (default: 30)")
    parser.add_argument("--years", type=int, default=3,
                        help="history in years (default: 3)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="stocks held each week (default: 5)")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="fraction of weeks for training (default: 0.7)")
    parser.add_argument("--no-xgb", action="store_true",
                        help="skip the XGBoost model")
    parser.add_argument("--no-lstm", action="store_true",
                        help="skip the LSTM model (faster runs)")
    parser.add_argument("--universe", choices=["name", "market-cap"],
                        default="name",
                        help="universe ordering: Wikipedia name order "
                             "(default) or largest by market cap")
    parser.add_argument("--force-refresh", action="store_true",
                        help="re-download data instead of using the cache")
    parser.add_argument("--borrow-fee", type=float,
                        default=BORROW_FEE_ANNUAL,
                        help="annualised borrow fee on short notional for "
                             "strategy F2 (default: 0.02 = 2%/yr)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="run Phase 3 walk-forward validation instead "
                             "of the single train/test split "
                             "(use --years 8 for enough folds)")
    parser.add_argument("--fold-weeks", type=int, default=13,
                        help="weeks per walk-forward fold "
                             "(default: 13 = one quarter)")
    parser.add_argument("--initial-train-weeks", type=int, default=104,
                        help="weeks in the first walk-forward training set "
                             "(default: 104 = 2 years)")
    args = parser.parse_args()

    if args.walk_forward:
        # Walk-forward needs the same panel as the standard run, but
        # replaces the single split + one-shot training with per-fold
        # retraining. LSTM is not retrained per fold (runtime); see
        # src/walkforward.py for the documented reasoning.
        _, panel, prices, spy, feature_cols = _load_panel(
            args.tickers, args.years, args.universe, args.force_refresh)
        wf_mod.run_walk_forward(
            panel, prices, spy, feature_cols,
            top_n=args.top_n,
            initial_train_weeks=args.initial_train_weeks,
            fold_weeks=args.fold_weeks,
            use_xgb=not args.no_xgb,
        )
        return

    run(
        n_tickers=args.tickers,
        years=args.years,
        top_n=args.top_n,
        train_frac=args.train_frac,
        use_xgb=not args.no_xgb,
        use_lstm=not args.no_lstm,
        universe=args.universe,
        force_refresh=args.force_refresh,
        borrow_fee=args.borrow_fee,
    )


if __name__ == "__main__":
    main()
