"""Feature engineering -- all features are point-in-time correct.

Every feature for week ``t`` uses only information available at the end of
week ``t`` (Friday close). No future data enters any feature: returns,
rolling windows and EMAs are all causal (``pct_change``, ``rolling`` and
``ewm`` only look backwards). The target (next week's return) is built
separately in ``target.py`` and is never used as a feature.

Feature sets follow the course convention:
    TO_PREDICT -- the target column.
    NUMERIC    -- continuous features.
    DUMMIES    -- month dummy variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TO_PREDICT = ["target"]

NUMERIC = [
    # Momentum (trend-continuation hypothesis).
    "ret_1w", "ret_4w", "ret_12w",
    # Realised volatility (annualised, from weekly log returns).
    "vol_4w", "vol_12w",
    # Technical indicators (implemented from scratch, no TA-Lib needed).
    "rsi_14", "macd", "macd_signal", "macd_hist",
    # Distance from 52-week extremes (mean-reversion / breakout signal).
    "dist_52w_high", "dist_52w_low",
    # Volume trend.
    "vol_ratio_4w",
    # Macro (yfinance proxies -- the fallback when FRED_API_KEY is not set).
    "vix", "vix_chg_1w", "tnx", "tnx_chg_1w",
    # FRED macro series (Phase 3 -- primary macro source when
    # FRED_API_KEY is set; the yfinance proxies above are the fallback).
    # Yields/spread changes are in percentage points (diff); VIX change
    # is a percent change, mirroring the proxy convention.
    "dgs10", "dgs10_chg_1w", "vixcls", "vixcls_chg_1w",
    "t10y2y", "t10y2y_chg_1w", "dgs2", "dgs2_chg_1w",
    # Fama-French weekly factors (same-week, known at Friday close).
    "mkt_rf", "smb", "hml", "rf",
]

DUMMIES = [f"month_{m}" for m in range(2, 13)]  # month_1 is the reference


# ---------------------------------------------------------------------------
# Technical indicators (from scratch)
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI with exponential smoothing (alpha = 1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram (weekly bars)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


# ---------------------------------------------------------------------------
# Per-ticker features
# ---------------------------------------------------------------------------
def _ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all price/volume-derived features for one ticker's history.

    ``g`` must be sorted by date and contain ``close`` and ``volume``.
    """
    g = g.copy()
    close = g["close"]
    log_ret = np.log(close / close.shift(1))

    # Momentum: trailing returns over 1 / 4 / 12 weeks.
    g["ret_1w"] = close.pct_change(1)
    g["ret_4w"] = close.pct_change(4)
    g["ret_12w"] = close.pct_change(12)

    # Realised volatility: rolling std of weekly log returns, annualised.
    g["vol_4w"] = log_ret.rolling(4).std() * np.sqrt(52)
    g["vol_12w"] = log_ret.rolling(12).std() * np.sqrt(52)

    # Technical indicators.
    g["rsi_14"] = _rsi(close, 14)
    macd, signal, hist = _macd(close)
    g["macd"] = macd
    g["macd_signal"] = signal
    g["macd_hist"] = hist

    # Distance from 52-week high / low (as a fraction of the extreme).
    roll_max = close.rolling(52).max()
    roll_min = close.rolling(52).min()
    g["dist_52w_high"] = close / roll_max - 1.0
    g["dist_52w_low"] = close / roll_min - 1.0

    # Volume trend: this week's volume vs its 4-week average.
    g["vol_ratio_4w"] = g["volume"] / g["volume"].rolling(4).mean() - 1.0
    return g


# ---------------------------------------------------------------------------
# Full panel
# ---------------------------------------------------------------------------
def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    french: pd.DataFrame | None = None,
    fred: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the modelling panel from prices (+ macro + French factors).

    Parameters
    ----------
    prices : long DataFrame with ``date, ticker, open, high, low, close,
        volume`` (from ``data.download_prices``).
    macro : DataFrame with ``date, vix, tnx`` (weekly, Monday-dated).
    french : DataFrame with ``date, mkt_rf, smb, hml, rf`` (Friday-dated).
    fred : DataFrame with ``date, dgs10, vixcls, t10y2y, dgs2``
        (Friday-dated, weekly end-of-period). When supplied, FRED is the
        macro source and ``macro`` should be None (the two would
        duplicate each other -- FRED's VIXCLS *is* the VIX). When None,
        the yfinance proxies in ``macro`` are used instead.

    Returns
    -------
    Long DataFrame with one row per (ticker, week) and all NUMERIC +
    DUMMIES columns. Rows whose 52-week features are not yet available
    (the first year) are dropped.
    """
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Normalise datetime resolution: parquet round-trips can come back as
    # datetime64[ms] while fresh parses are [us]/[ns]; merge_asof requires
    # identical key dtypes.
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    # Per-ticker loop (not groupby.apply: pandas 3 drops the grouping
    # column from apply results, and the loop keeps the code explicit).
    frames = []
    for ticker, g in prices.groupby("ticker", sort=False):
        out = _ticker_features(g)
        out["ticker"] = ticker
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)

    # --- Macro (merge on the Monday bar date; both are weekly) ---
    if macro is not None:
        m = macro.copy()
        m["date"] = pd.to_datetime(m["date"]).astype("datetime64[ns]")
        m["vix_chg_1w"] = m["vix"].pct_change(1)
        m["tnx_chg_1w"] = m["tnx"].diff(1)
        df = df.merge(m, on="date", how="left")

    # --- Fama-French factors: Friday-dated; merge_asof forward so each
    # Monday bar picks up the factors of the week ending that Friday.
    # The bar's close is Friday's close, so this is point-in-time correct.
    if french is not None:
        f = french.sort_values("date").reset_index(drop=True)
        f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df, f, on="date", direction="forward",
            tolerance=pd.Timedelta(days=7),
        )
        # The French library lags real time by ~1-2 months, so the most
        # recent weeks have no factor row. Forward-fill the last available
        # weekly factors (they are persistent enough for this purpose) so
        # the latest weeks -- including the live-trade week -- keep their
        # features. Only the tail is affected; the backtest body is exact.
        # (df is date-sorted here and every ticker in a week shares the
        # same factor values, so a plain ffill is point-in-time correct.)
        ff_cols = ["mkt_rf", "smb", "hml", "rf"]
        df[ff_cols] = df[ff_cols].ffill()

    # --- FRED macro series (Phase 3): Friday-dated weekly end-of-period
    # values; merge_asof forward so each Monday bar picks up the FRED
    # values of the week ending that Friday (the bar's close is Friday's
    # close, so this is point-in-time correct -- same logic as French).
    # Changes are computed on the WEEKLY frame before merging: diffing
    # after the merge would compare rows of the same week against each
    # other (every ticker shares the week's FRED values) and return ~0.
    # FRED publishes with ~1 day lag, so the tail is normally covered;
    # ffill anyway so the latest (live-trade) week never loses its macro
    # features to a publication delay.
    if fred is not None:
        fr = fred.sort_values("date").reset_index(drop=True)
        fr["date"] = pd.to_datetime(fr["date"]).astype("datetime64[ns]")
        # 1-week changes: yields and the spread move in percentage points
        # (diff); the VIX is an index level (percent change).
        fr["dgs10_chg_1w"] = fr["dgs10"].diff(1)
        fr["dgs2_chg_1w"] = fr["dgs2"].diff(1)
        fr["t10y2y_chg_1w"] = fr["t10y2y"].diff(1)
        fr["vixcls_chg_1w"] = fr["vixcls"].pct_change(1)
        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df, fr, on="date", direction="forward",
            tolerance=pd.Timedelta(days=7),
        )
        fred_cols = ["dgs10", "vixcls", "t10y2y", "dgs2",
                     "dgs10_chg_1w", "dgs2_chg_1w",
                     "t10y2y_chg_1w", "vixcls_chg_1w"]
        df[fred_cols] = df[fred_cols].ffill()

    # --- Month dummies (calendar / seasonality) ---
    df["month"] = pd.to_datetime(df["date"]).dt.month
    for m in range(2, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    df = df.drop(columns=["month"])

    # Drop rows with incomplete lookbacks (first 52 weeks per ticker, plus
    # any week missing macro/factor data).
    needed = [c for c in NUMERIC if c in df.columns]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"Features: {len(df):,} rows after dropping {before - len(df):,} "
          f"incomplete rows; {df['ticker'].nunique()} tickers, "
          f"{df['date'].nunique()} weeks.")
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, dummy) columns actually present in ``df``."""
    numeric = [c for c in NUMERIC if c in df.columns]
    dummies = [c for c in DUMMIES if c in df.columns]
    return numeric, dummies
