"""Precompute walk-forward validation results for the dashboard.

Runs the same Phase 3/4 walk-forward validation as
``python main.py --tickers 50 --years 8 --walk-forward --regime-filter``
(20 folds, 260 out-of-sample weeks, RF/XGB/FF per fold, Strategy A
unfiltered + 200-day MA filtered + 1% band) and writes the results to
``dashboard_data/walkforward_results.json``.

The dashboard's Walk-forward tab reads this file instead of re-running
the validation on every page load (the full run takes minutes). The
file is labelled with its as-of date in the app; re-run this script
(and redeploy) to refresh it.

Usage (from the project root, venv activated)::

    python scripts/precompute_walkforward.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as main_mod
from src import regime as regime_mod
from src import walkforward as wf_mod

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "dashboard_data" / "walkforward_results.json"
)


def _clean(value):
    """Make a value JSON-serialisable (numpy scalars, dates, NaN)."""
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        return str(value.date()) if isinstance(value, pd.Timestamp) else str(value)
    return value


def _df_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return _clean(df.to_dict(orient="records"))


def main() -> None:
    _, panel, prices, spy, feature_cols = main_mod._load_panel(
        50, 8, "name", False)
    daily, exposure, exposure_band = main_mod._regime_exposures(
        panel, 8, False)
    print(regime_mod.describe_current_state(daily))

    result = wf_mod.run_walk_forward(
        panel, prices, spy, feature_cols,
        top_n=5,
        initial_train_weeks=104,
        fold_weeks=13,
        use_xgb=True,
        exposure=exposure,
        exposure_band=exposure_band,
    )

    auc = result["auc_table"]
    auc_summary = {}
    for col in [c for c in auc.columns if c.endswith("AUC")]:
        auc_summary[col] = {
            "mean": float(auc[col].mean()),
            "min": float(auc[col].min()),
            "max": float(auc[col].max()),
            "std": float(auc[col].std()),
        }

    oos = result["oos_weekly"].sort_values("date")
    oos_curve = _df_records(
        oos[["date", "ret"]].assign(
            equity=(1.0 + oos["ret"]).cumprod()))
    payload = {
        "generated_date": str(date.today()),
        "config": {
            "tickers": 50,
            "years": 8,
            "universe": "name",
            "initial_train_weeks": 104,
            "fold_weeks": 13,
            "top_n": 5,
        },
        "n_folds": int(len(auc)),
        "oos_weeks": int(len(oos)),
        "oos_start": str(pd.to_datetime(oos["date"]).min().date()),
        "oos_end": str(pd.to_datetime(oos["date"]).max().date()),
        "auc_summary": auc_summary,
        "auc_table": _df_records(auc),
        "pooled_metrics": _clean(result["pooled_metrics"]),
        "per_year": _df_records(result["per_year"]),
        "benchmark_per_year": _df_records(result["benchmark_per_year"]),
        "fold_strategy": _df_records(result["strategy_table"]),
        "oos_equity_curve": oos_curve,
    }
    if "pooled_metrics_filtered" in result:
        oos_f = result["oos_weekly_filtered"].sort_values("date")
        payload["pooled_metrics_filtered"] = _clean(
            result["pooled_metrics_filtered"])
        payload["per_year_filtered"] = _df_records(
            result["per_year_filtered"])
        payload["filter_stats"] = _clean(result["filter_stats"])
        payload["oos_equity_curve_filtered"] = _df_records(
            oos_f[["date", "ret"]].assign(
                equity=(1.0 + oos_f["ret"]).cumprod()))
    if "pooled_metrics_filtered_band" in result:
        payload["pooled_metrics_filtered_band"] = _clean(
            result["pooled_metrics_filtered_band"])
        payload["filter_stats_band"] = _clean(result["filter_stats_band"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
