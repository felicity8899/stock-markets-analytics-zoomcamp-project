"""Target construction.

Target: top-quintile next-week return (binary).

For each week, stocks are ranked by their realised next-week return; the
target is 1 for stocks in the top 20% cross-sectionally that week, else 0.
This matches the trading decision directly ("buy the best N stocks") better
than a plain up/down label, where +0.1% and +5% count the same.

The target uses next week's return, so it is built *after* the features and
is never itself a feature. The final week of each ticker has no next-week
return and is dropped.
"""

from __future__ import annotations

import pandas as pd


def make_target(df: pd.DataFrame, quantile: float = 0.8) -> pd.DataFrame:
    """Add ``next_ret`` and binary ``target`` columns to the panel.

    Parameters
    ----------
    df : panel with ``date, ticker, close`` (output of
        ``features.build_features``).
    quantile : cross-sectional rank threshold; 0.8 = top quintile.

    Returns
    -------
    Panel with ``next_ret`` (realised next-week return) and ``target``
    (1 if next_ret is in the top ``(1 - quantile)`` of that week's
    cross-section). Rows without a next week are dropped.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["next_ret"] = (
        df.groupby("ticker")["close"].transform(lambda s: s.shift(-1) / s - 1.0)
    )
    df = df.dropna(subset=["next_ret"]).reset_index(drop=True)

    # Cross-sectional rank of next week's return within each week.
    # Strict '>' gives exactly the top 20% when there are no ties
    # (with 30 stocks, pct ranks are k/30, so >= 0.8 would take 7).
    df["next_ret_rank"] = df.groupby("date")["next_ret"].rank(pct=True)
    df["target"] = (df["next_ret_rank"] > quantile).astype(int)
    df = df.drop(columns=["next_ret_rank"])

    pos_rate = df["target"].mean()
    print(f"Target: {len(df):,} rows, positive rate {pos_rate:.3f} "
          f"(expected ~{1 - quantile:.2f}).")
    return df


def time_split(
    df: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the panel by time (never shuffle time series).

    The first ``train_frac`` of distinct weeks go to train, the rest to
    test. Returns (train, test, cutoff_date).
    """
    weeks = sorted(df["date"].unique())
    cutoff_idx = int(len(weeks) * train_frac)
    cutoff = pd.Timestamp(weeks[cutoff_idx])
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)
    print(f"Time split: train {train['date'].min().date()} -> "
          f"{train['date'].max().date()} ({len(train):,} rows); "
          f"test {test['date'].min().date()} -> "
          f"{test['date'].max().date()} ({len(test):,} rows).")
    return train, test, cutoff
