"""Observation dashboard for the Stock Markets Analytics Zoomcamp project.

Streamlit app (run: ``streamlit run app.py``). Reuses the pipeline in
``src/`` -- it does not re-implement any data, feature, model, or
simulation logic.

Tabs: Regime (200-day MA filter state), This week's picks (RF top-5 /
bottom-5 + ticker search), Full ranking (all 50 stocks), Strategies
(standard-run backtest + equity curves), Walk-forward (precomputed
validation results), About (how it works + honest caveats).

Data & performance notes
------------------------
* Only the RandomForest model is used for live picks (fast; the LSTM /
  XGBoost comparisons live in the README and notebooks, not here).
* All loaders are cached with ``st.cache_data`` (12h TTL). The
  "Refresh data" button in the sidebar bumps a cache key, which forces
  a fresh download from yfinance on the next load.
* The app never depends on a local ``data/`` cache persisting: on a
  cold start (e.g. Render free tier after spin-down) it re-downloads
  from yfinance, which takes under a minute. A spinner is shown while
  loading.
* Macro source: FRED when the ``FRED_API_KEY`` environment variable is
  set, otherwise the yfinance ^VIX/^TNX proxies (existing fallback in
  ``main._load_panel``). The key is never read from a file and is never
  printed -- the sidebar only reports whether it is set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as main_mod
from src import models as models_mod
from src import regime as regime_mod
from src import simulate as sim_mod
from src.target import time_split

# Dashboard configuration (matches the project's canonical runs).
N_TICKERS = 50
YEARS = 5
TOP_N = 5
TRAIN_FRAC = 0.7
REGIME_YEARS = 3  # daily SPY history for the regime tab (3y + 1y buffer)
CACHE_TTL_SECONDS = 12 * 3600  # 12 hours

WF_RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "dashboard_data" / "walkforward_results.json"
)

UNIVERSE_LABELS = {
    "market-cap": "Market-cap (largest first)",
    "name": "Name order (A-Z)",
}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_panel(universe: str, refresh_key: int = 0):
    """Panel for one universe (50 tickers, 5 years).

    ``refresh_key`` is a cache-buster: the sidebar Refresh button bumps
    it, which forces a fresh yfinance download instead of the parquet
    cache. Returns (feat, panel, prices, spy, feature_cols) exactly as
    ``main._load_panel`` does.
    """
    return main_mod._load_panel(
        N_TICKERS, YEARS, universe, force_refresh=(refresh_key > 0))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_ranking(universe: str, refresh_key: int = 0) -> dict:
    """RF ranking for the latest week: RF retrained on all data.

    Returns the ranked frame (rank, ticker, close, score), the latest
    week date, and the test AUC of the single-split run (for context).
    """
    feat, panel, _prices, _spy, feature_cols = load_panel(
        universe, refresh_key)
    rf_full = models_mod.train_random_forest(
        panel[feature_cols], panel["target"])
    latest_date = feat["date"].max()
    latest = feat[feat["date"] == latest_date].copy()
    latest["score"] = models_mod.score_random_forest(
        rf_full, latest[feature_cols])
    ranked = latest.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked = ranked[["rank", "ticker", "close", "score"]]
    return {"ranked": ranked, "latest_date": pd.Timestamp(latest_date).date()}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_regime(refresh_key: int = 0) -> dict:
    """Daily SPY + 200-day MA: current state and 2-year chart frame."""
    daily = regime_mod.download_spy_daily(
        years=REGIME_YEARS, force_refresh=(refresh_key > 0))
    state = regime_mod.current_state(daily)
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    daily["ma200"] = daily["close"].rolling(200).mean()
    chart = daily.dropna(subset=["ma200"]).tail(2 * 252).copy()
    chart = chart.set_index("date")[["close", "ma200"]]
    chart.columns = ["SPY close", "200-day MA"]
    return {"state": state, "chart": chart}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_strategies(refresh_key: int = 0) -> dict:
    """Standard-run backtest (name-order universe): RF strategies only.

    Mirrors ``main.run`` but skips XGBoost/LSTM for speed. Returns the
    metrics table (as a display-ready DataFrame), the equity-curve
    frame, and the test-period dates.
    """
    _feat, panel, prices, spy, feature_cols = load_panel("name", refresh_key)
    train, test, _cutoff = time_split(panel, train_frac=TRAIN_FRAC)

    rf = models_mod.train_random_forest(train[feature_cols], train["target"])
    test_rf = models_mod.add_scores(
        test, rf, models_mod.score_random_forest, feature_cols)

    results: dict[str, dict] = {}
    curves: dict[str, pd.Series] = {}

    m_a, w_a = sim_mod.simulate(test_rf, prices, top_n=TOP_N, strategy="equal")
    results["A: RF top-5 equal-weight"] = m_a
    curves["A: RF top-5 equal"] = (1.0 + w_a.set_index("date")["ret"]).cumprod()

    m_b, w_b = sim_mod.simulate(test_rf, prices, top_n=TOP_N, strategy="hrp")
    results["B: RF top-5 HRP-weight"] = m_b
    curves["B: RF top-5 HRP"] = (1.0 + w_b.set_index("date")["ret"]).cumprod()

    m_a10, _ = sim_mod.simulate(test_rf, prices, top_n=10, strategy="equal")
    results["A10: RF top-10 equal-weight"] = m_a10
    m_b10, _ = sim_mod.simulate(test_rf, prices, top_n=10, strategy="hrp")
    results["B10: RF top-10 HRP-weight"] = m_b10

    m_ls, w_ls = sim_mod.simulate_long_short(test_rf, top_n=TOP_N)
    results["F: RF long-short top-5/bottom-5"] = m_ls
    curves["F: long-short"] = (1.0 + w_ls.set_index("date")["ret"]).cumprod()

    m_ls_fee, _ = sim_mod.simulate_long_short(
        test_rf, top_n=TOP_N, borrow_fee_annual=sim_mod.BORROW_FEE_ANNUAL)
    results["F2: long-short, 2%/yr borrow fee"] = m_ls_fee

    ff = models_mod.train_factor_regression(
        train[feature_cols], train["next_ret"])
    test_ff = models_mod.add_scores(
        test, ff, models_mod.score_factor_regression, feature_cols)
    m_ff, _ = sim_mod.simulate(test_ff, prices, top_n=TOP_N, strategy="equal")
    results["C: FF-regression top-5 equal-weight"] = m_ff

    results["Benchmark: SPY buy-and-hold"] = sim_mod.benchmark_metrics(
        spy, test["date"].unique())
    bench = sim_mod.benchmark_weekly(spy, test["date"].unique())
    curves["SPY buy-and-hold"] = (
        (1.0 + bench.set_index("date")["ret"]).cumprod())

    table = sim_mod.metrics_table(results)
    equity = pd.DataFrame(curves).sort_index()
    return {
        "table": table,
        "equity": equity,
        "test_start": pd.to_datetime(test["date"]).min().date(),
        "test_end": pd.to_datetime(test["date"]).max().date(),
        "test_weeks": int(test["date"].nunique()),
    }


def load_walkforward() -> dict:
    """Precomputed walk-forward results (see scripts/precompute_walkforward.py)."""
    with open(WF_RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt_pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}%" if isinstance(value, float) else str(value)


def _pct(value) -> str:
    """Format a 0-1 fraction as a percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:+.1f}%"


def _pct_plain(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def metrics_display(table: pd.DataFrame) -> pd.DataFrame:
    """Format a metrics table (fractions -> readable strings)."""
    out = table.copy()
    out["total_return"] = out["total_return"].map(_pct)
    out["CAGR"] = out["CAGR"].map(_pct)
    out["Sharpe"] = out["Sharpe"].map(
        lambda v: f"{v:.2f}" if v is not None else "—")
    out["max_drawdown"] = out["max_drawdown"].map(_pct_plain)
    out["avg_turnover"] = out["avg_turnover"].map(
        lambda v: f"{v:.2f}" if v is not None else "—")
    out["win_rate"] = out["win_rate"].map(_pct_plain)
    out.columns = ["Strategy", "Weeks", "Total return", "CAGR", "Sharpe",
                   "Max drawdown", "Avg turnover", "Win rate"]
    return out


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_regime(refresh_key: int) -> None:
    st.header("Regime filter — 200-day moving average")
    st.caption("Faber (2007) trend rule: invested when SPY is above its "
               "200-day MA, in cash when below. The filter gates *whether* "
               "the strategy is invested, never *which* stocks it picks.")
    with st.spinner("Loading daily SPY data…"):
        data = load_regime(refresh_key)
    s = data["state"]
    in_market = s["in_market"]
    badge_color = "#1a7f37" if in_market else "#cf222e"
    badge_text = "IN — INVESTED" if in_market else "OUT — CASH"
    st.markdown(
        f"<div style='background:{badge_color}; color:white; "
        f"font-size:28px; font-weight:700; text-align:center; "
        f"padding:14px; border-radius:8px;'>{badge_text}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPY close", f"${s['close']:,.2f}")
    c2.metric("200-day MA", f"${s['ma']:,.2f}")
    c3.metric("Distance from MA", f"{s['distance_pct']:+.1f}%")
    c4.metric("As of", str(s["date"]))
    st.line_chart(data["chart"], height=320)
    st.caption("SPY daily close vs its 200-day moving average, last ~2 "
               "years. The as-of date is the last trading day in the "
               "data — if today's bar has not settled yet, it is "
               "yesterday's close.")


def tab_picks(universe: str, refresh_key: int) -> None:
    st.header("This week's picks")
    st.caption("RandomForest, retrained on all data. Score = P(next-week "
               "return in the top quintile). Baseline is 0.20 — a score "
               "of 0.40 is already 2x the base rate; 0.50 is not the bar.")
    with st.spinner("Loading data and scoring…"):
        data = load_ranking(universe, refresh_key)
    ranked = data["ranked"]
    st.subheader(f"Week of {data['latest_date']} "
                 f"({UNIVERSE_LABELS[universe]})")
    left, right = st.columns(2)
    with left:
        st.markdown("**Top 5 — long**")
        top5 = ranked.head(TOP_N).copy()
        top5["close"] = top5["close"].map(lambda v: f"${v:,.2f}")
        top5["score"] = top5["score"].map(lambda v: f"{v:.3f}")
        st.dataframe(top5, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Bottom 5 — short leg**")
        bot5 = ranked.tail(TOP_N).iloc[::-1].copy()
        bot5["close"] = bot5["close"].map(lambda v: f"${v:,.2f}")
        bot5["score"] = bot5["score"].map(lambda v: f"{v:.3f}")
        st.dataframe(bot5, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Ticker search")
    query = st.text_input("Ticker (e.g. NVDA)", value="",
                          placeholder="Type a ticker in this universe")
    query = query.strip().upper()
    if query:
        hit = ranked[ranked["ticker"] == query]
        if not hit.empty:
            row = hit.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rank", f"{int(row['rank'])} / {len(ranked)}")
            c2.metric("Score", f"{row['score']:.3f}")
            c3.metric("Close", f"${row['close']:,.2f}")
        else:
            st.warning(
                f"{query} is not in the {UNIVERSE_LABELS[universe]} "
                f"universe. Try the other universe from the sidebar — "
                f"e.g. NVDA is in market-cap but not in name order.")


def tab_ranking(universe: str, refresh_key: int) -> None:
    st.header("Full ranking")
    st.caption("All 50 stocks by RF score. Click a column header to sort. "
               "The score bar is the probability of a top-quintile week "
               "(baseline 0.20).")
    with st.spinner("Loading data and scoring…"):
        data = load_ranking(universe, refresh_key)
    ranked = data["ranked"]
    st.subheader(f"Week of {data['latest_date']} "
                 f"({UNIVERSE_LABELS[universe]})")
    st.dataframe(
        ranked,
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "close": st.column_config.NumberColumn(
                "Close", format="$%.2f"),
            "score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )


def tab_strategies(refresh_key: int) -> None:
    st.header("Strategies — standard backtest")
    st.caption("Single train/test split (first 70% of weeks train, last "
               "30% test), 50 tickers, name order, 5 years. RF strategies "
               "only — the XGBoost/LSTM comparisons are in the README, "
               "not recomputed here.")
    with st.spinner("Running the backtest…"):
        data = load_strategies(refresh_key)
    st.subheader(f"Test period: {data['test_start']} → {data['test_end']} "
                 f"({data['test_weeks']} weeks)")
    st.dataframe(metrics_display(data["table"]), hide_index=True,
                 use_container_width=True)
    st.subheader("Equity curves ($1 invested at test start)")
    st.line_chart(data["equity"], height=320)
    st.caption("Costs: 5 bps one-way on traded notional; F2 also charges "
               "a 2%/yr borrow fee on the short leg. One short window — "
               "see the Walk-forward tab for the honest multi-year read.")


def tab_walkforward() -> None:
    st.header("Walk-forward validation")
    st.caption("The honest headline: quarterly retraining across 20 folds "
               "(expanding window), 260 out-of-sample weeks. Each week's "
               "return comes from a model trained only on data before "
               "that week.")
    try:
        wf = load_walkforward()
    except FileNotFoundError:
        st.error("Precomputed walk-forward results are missing "
                 "(dashboard_data/walkforward_results.json). Re-run "
                 "`python scripts/precompute_walkforward.py`.")
        return
    st.caption(f"Precomputed results, as of {wf['generated_date']} "
               f"({wf['n_folds']} folds, {wf['oos_weeks']} out-of-sample "
               f"weeks, {wf['oos_start']} → {wf['oos_end']}).")

    st.subheader("Pooled out-of-sample — Strategy A")
    rows = [{
        "Variant": "A unfiltered",
        "Total return": _pct(wf["pooled_metrics"]["total_return"]),
        "CAGR": _pct(wf["pooled_metrics"]["cagr"]),
        "Sharpe": f"{wf['pooled_metrics']['sharpe']:.2f}",
        "Max drawdown": _pct_plain(wf["pooled_metrics"]["max_drawdown"]),
        "Win rate": _pct_plain(wf["pooled_metrics"]["win_rate"]),
    }]
    if "pooled_metrics_filtered" in wf:
        pf = wf["pooled_metrics_filtered"]
        rows.append({
            "Variant": "A + 200d MA filter",
            "Total return": _pct(pf["total_return"]),
            "CAGR": _pct(pf["cagr"]),
            "Sharpe": f"{pf['sharpe']:.2f}",
            "Max drawdown": _pct_plain(pf["max_drawdown"]),
            "Win rate": _pct_plain(pf["win_rate"]) + " *",
        })
    if "pooled_metrics_filtered_band" in wf:
        pb = wf["pooled_metrics_filtered_band"]
        rows.append({
            "Variant": "A + filter, 1% band",
            "Total return": _pct(pb["total_return"]),
            "CAGR": _pct(pb["cagr"]),
            "Sharpe": f"{pb['sharpe']:.2f}",
            "Max drawdown": _pct_plain(pb["max_drawdown"]),
            "Win rate": _pct_plain(pb["win_rate"]) + " *",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)
    st.caption("* Filtered win rates count cash weeks (0% return) as "
               "non-wins. On invested weeks only, the filtered win rate "
               f"is {_pct_plain(wf.get('filter_stats', {}).get('win_rate_invested_weeks'))} "
               f"(unfiltered: {_pct_plain(wf['pooled_metrics']['win_rate'])}).")

    if "filter_stats" in wf:
        fs = wf["filter_stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Filter: weeks invested",
                  f"{fs['pct_weeks_invested']:.1f}%")
        c2.metric("Filter: exposure switches", fs["switches"])
        c3.metric("Mean RF AUC across folds",
                  f"{wf['auc_summary']['RF AUC']['mean']:.3f}")

    st.subheader("Pooled equity curve ($1 at OOS start)")
    curve = pd.DataFrame(wf["oos_equity_curve"])
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date")[["equity"]].rename(
        columns={"equity": "A unfiltered"})
    if "oos_equity_curve_filtered" in wf:
        curve_f = pd.DataFrame(wf["oos_equity_curve_filtered"])
        curve_f["date"] = pd.to_datetime(curve_f["date"])
        curve_f = curve_f.set_index("date")[["equity"]].rename(
            columns={"equity": "A + 200d MA filter"})
        curve = curve.join(curve_f, how="outer")
    st.line_chart(curve, height=300)

    st.subheader("Strategy A by calendar year vs SPY")
    per_year = pd.DataFrame(wf["per_year"])
    bench = pd.DataFrame(wf["benchmark_per_year"])
    year_rows = []
    for _, r in per_year.iterrows():
        b = bench[bench["year"] == r["year"]]
        spy_ret = b["total_return"].iloc[0] if not b.empty else None
        spy_sharpe = b["sharpe"].iloc[0] if not b.empty else None
        year_rows.append({
            "Year": int(r["year"]),
            "Weeks": int(r["weeks"]),
            "Strategy A": _pct(r["total_return"]),
            "SPY": _pct(spy_ret),
            "A Sharpe": f"{r['sharpe']:.2f}",
            "SPY Sharpe": f"{spy_sharpe:.2f}" if spy_sharpe is not None else "—",
            "A max DD": _pct_plain(r["max_drawdown"]),
        })
    st.dataframe(pd.DataFrame(year_rows), hide_index=True,
                 use_container_width=True)
    st.caption("2022 is the year that matters: the strategy lost ~3x the "
               "market. The 200-day MA filter (Phase 4) more than halves "
               "that year's loss — see the pooled table above.")


def tab_about() -> None:
    st.header("About this dashboard")
    st.markdown("""
**How the model picks stocks.** Each week, the system computes features
for 50 S&P 500 large caps — momentum (1/4/12 weeks), realised volatility,
RSI, MACD, distance from the 52-week high/low, volume trend, VIX and the
10-year yield, Fama-French factors, and month dummies. A RandomForest
classifier (the course's 2026 best config) predicts each stock's
probability of landing in the top quintile of next week's returns. The
portfolio buys the top 5 by that probability, equal-weighted, holds for
one week, and rebalances on Monday.

**What the score means.** The score is P(next-week return in the top
20% of the cross-section). The baseline is **0.20**, not 0.50: a random
stock has a 20% chance of being in the top quintile. A score of 0.40
means the model thinks the stock is twice as likely as average to be a
top performer — that is already a strong signal. A score of 0.50 would
be extraordinary. The score says nothing about whether the market as a
whole goes up or down: it ranks stocks *against each other*.

**The regime filter.** A separate 200-day moving-average rule (Faber
2007) decides *whether* the long-only strategy is invested at all: in
when SPY closes above its 200-day MA on Friday, in cash when below.
Walk-forward testing shows it cuts the maximum drawdown from −46% to
−29% and lifts the Sharpe ratio from 0.91 to 1.09, at the cost of
missing some rebounds (2023, 2025) and occasional whipsaws.

**Honest caveats.** The ranking signal is real but modest: mean ROC AUC
across 20 walk-forward folds is ~0.58 for the RandomForest. The
single-split headline (+91.6% over 63 weeks) came from one favourable
window; the pooled walk-forward headline is a +28% CAGR with a −47%
maximum drawdown before the filter. The backtest has survivorship bias
(the universe is the *current* S&P 500 list), costs are approximate
(5 bps one-way), and 8 years of history is short. This dashboard shows
only the RandomForest for live picks — the XGBoost and LSTM comparisons
are in the README and notebooks, not recomputed here, to keep the app
fast. **This is a course project, not investment advice.**
""")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Stock Markets Analytics — Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Weekly Top-Quintile Stock Selection — Dashboard")
    st.caption("DataTalksClub Stock Markets Analytics Zoomcamp 2026 · "
               "RandomForest ranking + 200-day MA regime filter · "
               "Not investment advice.")

    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.refresh_key += 1
            st.rerun()
        st.caption("Refresh clears the cache and re-downloads from "
                   "yfinance (takes ~1 minute on first load).")
        universe = st.radio(
            "Universe",
            options=["market-cap", "name"],
            format_func=lambda u: UNIVERSE_LABELS[u],
            index=0,
        )
        st.divider()
        st.caption(
            "Macro source: "
            + ("FRED (DGS10, VIXCLS, T10Y2Y, DGS2)"
               if os.environ.get("FRED_API_KEY")
               else "yfinance ^VIX/^TNX proxies")
            + ". Set FRED_API_KEY as an environment variable to use "
              "FRED (free key from the St. Louis Fed)."
        )
        st.caption("Data refreshes automatically every 12 hours; "
                   "use Refresh for the latest prices.")

    refresh_key = st.session_state.refresh_key
    tabs = st.tabs([
        "Regime",
        "This week's picks",
        "Full ranking",
        "Strategies",
        "Walk-forward",
        "About",
    ])
    with tabs[0]:
        tab_regime(refresh_key)
    with tabs[1]:
        tab_picks(universe, refresh_key)
    with tabs[2]:
        tab_ranking(universe, refresh_key)
    with tabs[3]:
        tab_strategies(refresh_key)
    with tabs[4]:
        tab_walkforward()
    with tabs[5]:
        tab_about()


if __name__ == "__main__":
    main()
