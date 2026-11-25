"""200-day moving-average regime filter (Phase 4).

A trend-following market-timing overlay in the spirit of Faber (2007),
"A Quantitative Approach to Tactical Asset Allocation": be invested
when the market is above its 200-day moving average, in cash when it is
below. The rule is applied to SPY (the market proxy) and gates the
long-only strategy (Strategy A) -- it does not change *which* stocks the
model picks, only *whether* the portfolio is invested at all.

Timing convention -- no lookahead (read this before changing the code)
----------------------------------------------------------------------
Weekly bars in this project are Monday-dated but contain the full week's
data (the close is Friday's close). The strategy rebalances at each
week's close: the picks for week ``t`` are bought at Friday's close of
week ``t`` and held into week ``t+1``, so the return attributed to week
``t`` is the next week's return.

The filter decision for week ``t`` must therefore use only data
available at Friday's close of week ``t``:

* the SPY close is the close of the **last trading day of week ``t``**
  (normally Friday; Thursday if Friday is a holiday), and
* the 200-day MA is the mean of the **200 daily closes ending on that
  same day** (inclusive).

Both are known at the rebalance moment, so the filter never peeks into
the future. The alignment is done with ``merge_asof`` (backward) from
each week's Friday onto the daily series, which also handles holiday
Fridays correctly. The exposure for week ``t`` gates the return earned
over week ``t+1`` -- exactly the week the strategy would have been
invested in.

Cash convention: out-of-market weeks earn 0% (no T-bill yield is
modelled -- see README). Liquidating into cash and re-entering both
count as one-way turnover of 1.0 in the simulation (see
``simulate.simulate``), so the filter's switching costs are charged,
not ignored.

Band variant (optional hysteresis): with ``band > 0``, the filter only
switches state when the close crosses the MA by more than the band --
out when close < MA * (1 - band), in when close > MA * (1 + band) --
which suppresses whipsaws around the average at the cost of slightly
later entries and exits.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data import _cache_is_fresh, _resolve_data_dir, _write_meta

# Default MA window (trading days). 200 days is the Faber (2007)
# convention and the industry-standard long-term trend length.
MA_WINDOW_DAYS = 200

# Extra years of daily history fetched beyond the requested window, so
# the 200-day MA is already valid at the first panel week (200 trading
# days is ~10 calendar months).
BUFFER_YEARS = 1


def download_spy_daily(
    years: int = 8,
    data_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily SPY closes (cached to ``data/spy_daily.parquet``).

    Fetches ``years + 1`` years so the 200-day MA is valid from the
    first panel week. Uses auto-adjusted closes, consistent with the
    weekly prices in ``data.download_prices`` (both the close and the
    MA are adjusted the same way, so the comparison is unaffected).

    Returns a DataFrame with columns ``date, close`` (one row per
    trading day).
    """
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "spy_daily.parquet"
    meta_path = data_dir / "spy_daily.meta.json"

    if (
        not force_refresh
        and cache_path.exists()
        and _cache_is_fresh(meta_path, ["SPY"], years)
    ):
        print(f"Regime: using daily SPY cache {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Regime: downloading SPY daily ({years + BUFFER_YEARS}y, "
          f"with {MA_WINDOW_DAYS}-day MA buffer) ...")
    raw = yf.download(
        "SPY", period=f"{years + BUFFER_YEARS}y", interval="1d",
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily SPY data.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        # yfinance >= 0.2.40 returns MultiIndex columns even for one
        # ticker, so Close is a one-column DataFrame; take the series.
        close = close.iloc[:, 0]
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index),
        "close": close.to_numpy(dtype=float),
    })
    out = out.dropna().sort_values("date").reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    _write_meta(meta_path, ["SPY"], years)
    print(f"Regime: {len(out)} daily closes, "
          f"{out['date'].min().date()} -> {out['date'].max().date()}")
    return out


def moving_average_regime(
    daily: pd.DataFrame,
    weekly_dates,
    window: int = MA_WINDOW_DAYS,
    band: float = 0.0,
) -> pd.DataFrame:
    """Weekly regime-filter exposure aligned to weekly Monday dates.

    Parameters
    ----------
    daily : daily SPY frame from ``download_spy_daily``
        (``date, close``).
    weekly_dates : the weekly bar dates to align to (Monday-dated,
        e.g. ``panel["date"].unique()``).
    window : MA length in trading days (default 200).
    band : hysteresis band as a fraction (default 0.0 = plain rule).
        With ``band > 0``, the state only flips when the close crosses
        the MA by more than the band (see module docstring).

    Returns a DataFrame with columns ``date`` (weekly Monday),
    ``spy_close`` (last trading day's close of that week),
    ``ma`` (the window-day MA as of that day), and ``exposure``
    (1.0 = invested, 0.0 = cash). Weeks before the first valid MA
    value are dropped.
    """
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    # Normalise datetime resolution: merge_asof requires identical
    # dtypes, and pandas 3.x keeps yfinance/parquet dates at [s] while
    # date_range produces [us]. Cast both sides to [ns].
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    daily["ma"] = daily["close"].rolling(window, min_periods=window).mean()

    fridays = pd.DataFrame({
        "date": pd.to_datetime(list(weekly_dates)).astype("datetime64[ns]"),
    })
    fridays["friday"] = fridays["date"] + pd.Timedelta(days=4)
    fridays = fridays.sort_values("friday")

    # Last daily observation on or before each week's Friday. This is
    # the no-lookahead step: the Friday close and the MA as of that
    # Friday are both known at the rebalance moment. The 7-day
    # tolerance guards against a (never-observed) full holiday week.
    merged = pd.merge_asof(
        fridays,
        daily[["date", "close", "ma"]],
        left_on="friday",
        right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
        suffixes=("", "_daily"),
    )
    merged = merged.rename(columns={"close": "spy_close"})[
        ["date", "spy_close", "ma"]].dropna(subset=["ma"])
    merged = merged.sort_values("date").reset_index(drop=True)

    if band <= 0.0:
        merged["exposure"] = (merged["spy_close"] > merged["ma"]).astype(float)
    else:
        # Hysteresis: keep the previous state inside the band.
        exposures: list[float] = []
        state = 1.0 if merged["spy_close"].iloc[0] > merged["ma"].iloc[0] else 0.0
        for _, row in merged.iterrows():
            if state == 1.0 and row["spy_close"] < row["ma"] * (1.0 - band):
                state = 0.0
            elif state == 0.0 and row["spy_close"] > row["ma"] * (1.0 + band):
                state = 1.0
            exposures.append(state)
        merged["exposure"] = exposures
    return merged


def current_state(
    daily: pd.DataFrame,
    window: int = MA_WINDOW_DAYS,
) -> dict:
    """Current regime state from the latest daily data.

    Returns a dict with the as-of date, the latest SPY close, the
    200-day MA as of that day, whether the filter says in (``True``)
    or out (``False``), and the close's distance from the MA in
    percent. The latest row is the last *trading* day in the data --
    if today's bar has not settled yet, it is yesterday's close, and
    the as-of date says so.
    """
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    daily["ma"] = daily["close"].rolling(window, min_periods=window).mean()
    last = daily.dropna(subset=["ma"]).iloc[-1]
    close = float(last["close"])
    ma = float(last["ma"])
    return {
        "date": pd.Timestamp(last["date"]).date(),
        "close": close,
        "ma": ma,
        "in_market": bool(close > ma),
        "distance_pct": (close / ma - 1.0) * 100.0,
    }


def describe_current_state(daily: pd.DataFrame, window: int = MA_WINDOW_DAYS) -> str:
    """One-line human-readable current state (for CLI output)."""
    s = current_state(daily, window)
    verdict = "IN (invested)" if s["in_market"] else "OUT (cash)"
    return (
        f"Regime filter (200-day MA): {verdict} -- SPY "
        f"${s['close']:,.2f} vs 200-day MA ${s['ma']:,.2f} "
        f"({s['distance_pct']:+.1f}%), as of {s['date']}"
    )
