"""Vector backtest simulation.

Weekly rebalance, long-only. Each week the model scores every stock; the
top-N by score are bought at that week's close and held for one week, so the
realised return of a position is the stock's *next-week* return.

Strategies
----------
A. Equal weight: top-N stocks, equal weights.
B. HRP weight: same top-N selection, weights from Hierarchical Risk Parity
   on the trailing 26 weeks of returns (see ``hrp.py``).
C. Long-short (Phase 2): long the top-N by score, short the bottom-N,
   equal weight within each leg, dollar-neutral (50% long / 50% short,
   100% gross). Phase 3 adds a configurable securities-borrow fee on the
   short leg (default 2% annualised on short notional, pro-rata weekly),
   so the long-short return can be reported with and without the fee.

Two simulation engines
----------------------
* ``simulate`` / ``simulate_long_short`` -- vector backtests: weekly
  portfolio returns are computed from weights x next-week returns, with
  costs on measured turnover.
* ``simulate_exact`` (Phase 2) -- row-by-row simulation with explicit
  dollar accounting: capital, positions and cash are tracked week by
  week, costs are deducted from capital in dollars, and gains are
  reinvested. It should land very close to the vector backtest for the
  same strategy; the residual gap is the cost-timing cross-term
  (costs reduce the capital that then earns the week's return) and is
  explained in the README.

Costs: 5 bps one-way on traded notional. Turnover is measured as half the
sum of absolute weight changes versus the previous week's holdings grown
by their realised returns (standard one-way turnover definition).

Phase 3 adds ``grouped_metrics`` (per-year / per-fold metrics from a
weekly returns frame) and ``benchmark_weekly`` (the SPY weekly return
series), which the walk-forward validation in ``walkforward.py`` uses to
report stability across calendar windows and folds.

Benchmark: SPY buy-and-hold over the same test weeks.

Metrics: total return, CAGR, Sharpe ratio (annualised, rf = 0), max
drawdown, average weekly turnover, win rate (fraction of weeks with a
positive portfolio return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hrp import hrp_weights_for_week

TRADING_COST_BPS = 5  # one-way, in basis points
WEEKS_PER_YEAR = 52

# Default annualised borrow fee on short notional (Phase 3). 2% is a
# typical rate for easy-to-borrow large caps; hard-to-borrow names can
# cost 10%+ -- the CLI flag --borrow-fee lets you stress-test that.
BORROW_FEE_ANNUAL = 0.02


def _metrics(weekly_ret: pd.Series, turnover: pd.Series) -> dict:
    """Performance metrics from a series of weekly portfolio returns."""
    weekly_ret = weekly_ret.dropna()
    n = len(weekly_ret)
    if n == 0:
        return {}
    total_ret = float((1.0 + weekly_ret).prod() - 1.0)
    cagr = float((1.0 + total_ret) ** (WEEKS_PER_YEAR / n) - 1.0)
    vol = float(weekly_ret.std(ddof=1))
    sharpe = float(weekly_ret.mean() / vol * np.sqrt(WEEKS_PER_YEAR)) if vol > 0 else 0.0
    equity = (1.0 + weekly_ret).cumprod()
    running_max = equity.cummax()
    max_dd = float(((equity - running_max) / running_max).min())
    return {
        "weeks": n,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "win_rate": float((weekly_ret > 0).mean()),
    }


def _week_portfolio_return(
    picks: pd.DataFrame, weights: pd.Series
) -> float:
    """Realised next-week return of one week's holdings."""
    w = weights.reindex(picks["ticker"]).fillna(0.0)
    return float((w.to_numpy() * picks["next_ret"].to_numpy()).sum())


def simulate(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = 5,
    strategy: str = "equal",
    score_threshold: float | None = None,
    cost_bps: float = TRADING_COST_BPS,
    hrp_lookback_weeks: int = 26,
) -> tuple[dict, pd.DataFrame]:
    """Run a vector backtest over the scored test panel.

    Parameters
    ----------
    panel : test panel with ``date, ticker, next_ret, score``.
    prices : long price frame (needed for HRP trailing returns).
    top_n : number of stocks held each week.
    strategy : "equal" or "hrp".
    score_threshold : if set, only stocks with ``score`` above it are
        eligible (e.g. 0.55 for the RandomForest probability rule). Weeks
        with no eligible stock hold cash (0 return that week).
    cost_bps : one-way transaction cost in basis points.
    hrp_lookback_weeks : trailing window for the HRP covariance.

    Returns
    -------
    (metrics dict, weekly DataFrame with date, ret, turnover, holdings).
    """
    if strategy not in {"equal", "hrp"}:
        raise ValueError("strategy must be 'equal' or 'hrp'")

    weekly_rows = []
    prev_weights: pd.Series | None = None
    prev_picks: pd.DataFrame | None = None

    for date, week in panel.groupby("date", sort=True):
        eligible = week
        if score_threshold is not None:
            eligible = week[week["score"] >= score_threshold]
        picks = eligible.nlargest(top_n, "score")
        if picks.empty:
            weekly_rows.append(
                {"date": date, "ret": 0.0, "turnover": 0.0,
                 "holdings": "", "n_holdings": 0}
            )
            prev_weights, prev_picks = None, None
            continue

        tickers = picks["ticker"].tolist()
        if strategy == "equal":
            weights = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            weights = hrp_weights_for_week(
                prices, tickers, pd.Timestamp(date),
                lookback_weeks=hrp_lookback_weeks,
            )

        gross_ret = _week_portfolio_return(picks, weights)

        # Turnover vs previous holdings grown by their realised returns.
        if prev_weights is None or prev_picks is None:
            turnover = 1.0  # first week: fully invested from cash
        else:
            grown = prev_weights * (1.0 + prev_picks.set_index("ticker")["next_ret"])
            grown = grown / grown.sum() if grown.sum() > 0 else grown * 0
            union = weights.index.union(grown.index)
            turnover = 0.5 * float(
                (weights.reindex(union, fill_value=0.0)
                 - grown.reindex(union, fill_value=0.0)).abs().sum()
            )
        cost = turnover * cost_bps / 10_000.0
        net_ret = gross_ret - cost

        weekly_rows.append({
            "date": date,
            "ret": net_ret,
            "gross_ret": gross_ret,
            "turnover": turnover,
            "holdings": ",".join(tickers),
            "n_holdings": len(tickers),
        })
        prev_weights, prev_picks = weights, picks

    weekly = pd.DataFrame(weekly_rows)
    metrics = _metrics(weekly["ret"], weekly["turnover"])
    metrics["strategy"] = strategy
    metrics["top_n"] = top_n
    return metrics, weekly


def benchmark_metrics(spy: pd.DataFrame, dates) -> dict:
    """SPY buy-and-hold metrics over the same test weeks.

    ``spy`` is the benchmark frame from ``data.download_benchmark``
    (``date, spy_close, spy_ret``). A week is included if its date is in
    ``dates``; the return used is the *next* week's SPY return, matching
    the strategy's holding period.
    """
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["next_spy_ret"] = spy["spy_ret"].shift(-1)
    mask = spy["date"].isin(pd.to_datetime(list(dates)))
    rets = spy.loc[mask, "next_spy_ret"].dropna()
    metrics = _metrics(rets, pd.Series(0.0, index=rets.index))
    metrics["strategy"] = "SPY buy-and-hold"
    return metrics


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Pretty-printable comparison table from {name: metrics}."""
    rows = []
    for name, m in results.items():
        rows.append({
            "strategy": name,
            "weeks": m.get("weeks"),
            "total_return": m.get("total_return"),
            "CAGR": m.get("cagr"),
            "Sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "avg_turnover": m.get("avg_turnover"),
            "win_rate": m.get("win_rate"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2: long-short strategy (vector backtest)
# ---------------------------------------------------------------------------
def simulate_long_short(
    panel: pd.DataFrame,
    top_n: int = 5,
    cost_bps: float = TRADING_COST_BPS,
    borrow_fee_annual: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    """Dollar-neutral long-short vector backtest.

    Each week: long the top-N stocks by score, short the bottom-N,
    equal weight within each leg, 50% of capital long / 50% short
    (100% gross, net zero). The weekly return is
    0.5 * (long-leg return) - 0.5 * (short-leg return), minus 5 bps
    one-way costs on the turnover of *both* legs.

    Borrow fee (Phase 3): shorting is not free -- the borrower pays the
    lender a fee on the short notional for as long as the position is
    open. ``borrow_fee_annual`` is that fee as an annual rate on short
    notional (e.g. 0.02 = 2%/year, typical for easy-to-borrow large
    caps); it is charged pro-rata weekly on the 50% short leg, i.e.
    each week's return is reduced by ``0.5 * borrow_fee_annual / 52``.
    The default 0.0 reproduces the Phase 2 behaviour (no borrow fee),
    so the fee's drag is visible by running the strategy twice.
    The metrics dict reports ``borrow_fee_drag`` (the sum of the weekly
    fee deductions, in return points) alongside the usual metrics.

    Turnover is measured on signed weights: last week's positions are
    grown by their realised returns and divided by (1 + last week's net
    portfolio return), which is the correct growth accounting for a
    portfolio whose weights sum to zero (the long-only renormalisation
    by the weight sum would divide by ~0 here).

    Returns (metrics dict, weekly DataFrame).
    """
    # Weekly borrow fee as a fraction of total capital: the short leg is
    # always ~50% of capital (rebalanced weekly), so the fee is a
    # constant weekly drag of 0.5 * annual_rate / 52.
    weekly_borrow_fee = 0.5 * borrow_fee_annual / WEEKS_PER_YEAR

    weekly_rows = []
    prev_weights: pd.Series | None = None
    prev_picks: pd.DataFrame | None = None
    prev_net_ret: float | None = None

    for date, week in panel.groupby("date", sort=True):
        longs = week.nlargest(top_n, "score")
        shorts = week.nsmallest(top_n, "score")
        long_w = pd.Series(0.5 / len(longs), index=longs["ticker"])
        short_w = pd.Series(-0.5 / len(shorts), index=shorts["ticker"])
        weights = pd.concat([long_w, short_w])
        picks = pd.concat([longs, shorts])

        gross_ret = _week_portfolio_return(picks, weights)

        if prev_weights is None or prev_picks is None or prev_net_ret is None:
            turnover = 1.0  # first week: establish 100% gross exposure
        else:
            grown = prev_weights * (
                1.0 + prev_picks.set_index("ticker")["next_ret"]
            )
            grown = grown / (1.0 + prev_net_ret)
            union = weights.index.union(grown.index)
            turnover = 0.5 * float(
                (weights.reindex(union, fill_value=0.0)
                 - grown.reindex(union, fill_value=0.0)).abs().sum()
            )
        cost = turnover * cost_bps / 10_000.0
        net_ret = gross_ret - cost - weekly_borrow_fee

        weekly_rows.append({
            "date": date,
            "ret": net_ret,
            "gross_ret": gross_ret,
            "turnover": turnover,
            "borrow_fee": weekly_borrow_fee,
            "longs": ",".join(longs["ticker"]),
            "shorts": ",".join(shorts["ticker"]),
            "n_holdings": len(longs) + len(shorts),
        })
        prev_weights, prev_picks, prev_net_ret = weights, picks, net_ret

    weekly = pd.DataFrame(weekly_rows)
    metrics = _metrics(weekly["ret"], weekly["turnover"])
    metrics["strategy"] = "long-short"
    metrics["top_n"] = top_n
    metrics["borrow_fee_annual"] = borrow_fee_annual
    # Total fee drag in return points (sum of the weekly deductions).
    # This is a slight overstatement of the total-return impact, because
    # fees also reduce the capital that compounds -- the exact drag is
    # the difference in total_return between the with-fee and no-fee
    # runs, which the pipeline reports side by side.
    metrics["borrow_fee_drag"] = float(weekly["borrow_fee"].sum())
    return metrics, weekly


# ---------------------------------------------------------------------------
# Phase 3: grouped metrics (per-year / per-fold) + benchmark weekly series
# ---------------------------------------------------------------------------
def grouped_metrics(weekly: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group metrics from a weekly returns frame.

    ``weekly`` must have ``date, ret, turnover`` and ``group_col``
    (e.g. calendar year, or a walk-forward fold label). Each group's
    metrics are computed from that group's weekly returns only.

    Two reading notes (see LEARNING_NOTES.md): (1) CAGR for a short
    group (e.g. a 13-week fold) is an annualised extrapolation of a
    short window -- treat it as a comparability device, not a forecast;
    (2) max drawdown resets at each group boundary, so a group's
    drawdown is the worst fall *within* that group, not from the
    strategy's all-time peak.
    """
    rows = []
    for group, g in weekly.groupby(group_col, sort=True):
        m = _metrics(g["ret"], g["turnover"])
        m[group_col] = group
        m["weeks"] = len(g)
        m["start"] = pd.to_datetime(g["date"]).min().date()
        m["end"] = pd.to_datetime(g["date"]).max().date()
        rows.append(m)
    cols = [group_col, "weeks", "start", "end", "total_return", "cagr",
            "sharpe", "max_drawdown", "avg_turnover", "win_rate"]
    return pd.DataFrame(rows)[cols]


def benchmark_weekly(spy: pd.DataFrame, dates) -> pd.DataFrame:
    """SPY buy-and-hold weekly return series over the given weeks.

    Same holding-period convention as ``benchmark_metrics`` (the return
    used for week ``t`` is the *next* week's SPY return), but returns
    the weekly series instead of aggregate metrics, so it can be sliced
    by year/fold with ``grouped_metrics``.
    """
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["next_spy_ret"] = spy["spy_ret"].shift(-1)
    mask = spy["date"].isin(pd.to_datetime(list(dates)))
    out = spy.loc[mask, ["date", "next_spy_ret"]].dropna().copy()
    out = out.rename(columns={"next_spy_ret": "ret"})
    out["turnover"] = 0.0
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2: exact row-by-row simulation with dollar accounting
# ---------------------------------------------------------------------------
def simulate_exact(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = 5,
    strategy: str = "equal",
    cost_bps: float = TRADING_COST_BPS,
    initial_capital: float = 100_000.0,
    hrp_lookback_weeks: int = 26,
) -> tuple[dict, pd.DataFrame]:
    """Exact week-by-week simulation with explicit capital accounting.

    Unlike the vector backtest (which computes weekly returns as
    weights x next-week returns and subtracts costs from the return),
    this engine tracks dollars: positions are dollar values, costs are
    deducted from capital in dollars at each rebalance, and gains are
    reinvested into the following week's positions.

    Timing convention (same as the vector backtest): at each week's
    close, positions chosen from that week's scores are held into the
    next week, so the return attributed to week ``t`` is the growth of
    the capital from the start of week ``t`` (pre-trade, after last
    week's positions have grown) to the start of week ``t+1``:
    ret_t = capital_{t+1}^{pre-trade} / capital_t^{pre-trade} - 1.
    Measuring from pre-trade to pre-trade is essential: it is the step
    that contains BOTH the cost deduction (pre -> post within week t)
    and the week's market growth. Measuring post-trade to pre-trade
    would silently drop the costs from the return series (the total
    return would then disagree with final_capital / initial_capital --
    a reconciliation check the tests rely on).

    Expected relationship to the vector backtest: the two should be
    very close. The residual gap is the cost-timing cross-term -- the
    exact engine charges costs *before* the week's return accrues, so
    ret_exact ~= (1 - cost) * (1 + gross) - 1, while the vector engine
    uses gross - cost. The gap is ~(gross x cost) per week, i.e. a few
    basis points over the whole test period.

    Returns (metrics dict, weekly DataFrame with the capital path).
    The metrics dict additionally carries ``final_capital`` and
    ``total_costs`` (dollars).
    """
    if strategy not in {"equal", "hrp"}:
        raise ValueError("simulate_exact: strategy must be 'equal' or 'hrp'")

    positions: pd.Series = pd.Series(dtype=float)  # ticker -> dollar value
    cash = float(initial_capital)
    prev_picks: pd.DataFrame | None = None
    rows: list[dict] = []
    total_costs = 0.0

    for date, week in panel.groupby("date", sort=True):
        # 1. Grow last week's positions by their realised returns.
        if prev_picks is not None and not positions.empty:
            rets = prev_picks.set_index("ticker")["next_ret"]
            growth = (1.0 + rets.reindex(positions.index).fillna(0.0))
            positions = positions * growth
        capital_pre = cash + float(positions.sum())

        # 2. Target weights from this week's scores.
        picks = week.nlargest(top_n, "score")
        tickers = picks["ticker"].tolist()
        if strategy == "equal":
            weights = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            weights = hrp_weights_for_week(
                prices, tickers, pd.Timestamp(date),
                lookback_weeks=hrp_lookback_weeks,
            )

        # 3. Turnover (one-way, fraction of capital) and dollar costs.
        # Cash is included as an asset: investing from cash (or
        # liquidating to cash) trades the full notional, so the first
        # week counts turnover 1.0, matching the vector backtest.
        target_values = weights * capital_pre
        current_cash = capital_pre - float(positions.sum())
        union = target_values.index.union(positions.index)
        stock_diff = float(
            (target_values.reindex(union, fill_value=0.0)
             - positions.reindex(union, fill_value=0.0)).abs().sum()
        )
        turnover = 0.5 * (stock_diff + abs(0.0 - current_cash)) / capital_pre
        cost_dollars = turnover * capital_pre * cost_bps / 10_000.0
        total_costs += cost_dollars
        capital_post = capital_pre - cost_dollars

        # 4. Rebalance on post-cost capital; residual stays in cash.
        positions = target_values * (capital_post / capital_pre)
        cash = capital_post - float(positions.sum())

        rows.append({
            "date": date,
            "capital_pre": capital_pre,
            "capital_post": capital_post,
            "turnover": turnover,
            "cost_dollars": cost_dollars,
            "holdings": ",".join(tickers),
            "n_holdings": len(tickers),
        })
        prev_picks = picks

    # Final leg: grow the last week's positions by their realised
    # next-week returns to close out the capital path.
    if prev_picks is not None and not positions.empty:
        rets = prev_picks.set_index("ticker")["next_ret"]
        growth = (1.0 + rets.reindex(positions.index).fillna(0.0))
        positions = positions * growth
    final_capital = cash + float(positions.sum())

    weekly = pd.DataFrame(rows)
    # Weekly returns from the capital path, measured pre-trade to
    # pre-trade so costs are included (see docstring). Reconciliation:
    # prod(1 + ret) must equal final_capital / initial_capital.
    caps_pre = weekly["capital_pre"].to_numpy()
    rets = np.empty(len(weekly))
    rets[:-1] = caps_pre[1:] / caps_pre[:-1] - 1.0
    rets[-1] = final_capital / caps_pre[-1] - 1.0
    weekly["ret"] = rets

    metrics = _metrics(weekly["ret"], weekly["turnover"])
    metrics["strategy"] = f"exact-{strategy}"
    metrics["top_n"] = top_n
    metrics["final_capital"] = final_capital
    metrics["total_costs"] = total_costs
    return metrics, weekly
