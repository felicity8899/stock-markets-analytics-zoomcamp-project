"""Data sourcing for the Stock Markets Analytics Zoomcamp project.

Sources
-------
* Prices: yfinance weekly bars (5y by default), cached to ``data/`` as parquet
  so reruns do not re-download. Course rule: no data files are committed to
  GitHub -- ``data/`` is in ``.gitignore``.
* Macro proxies: ^VIX and ^TNX from yfinance. A FRED source is the
  primary macro source when the ``FRED_API_KEY`` environment variable is
  set (see ``download_fred``): DGS10, VIXCLS, T10Y2Y and DGS2, weekly
  end-of-period values from api.stlouisfed.org. Without a key the
  yfinance proxies are used instead (see ``main.py``).
* Fama-French weekly factors: Ken French's data library
  (mba.tuck.dartmouth.edu). The file has a messy format (header text rows
  and a copyright footer) which ``download_french_weekly`` handles.

Point-in-time note
------------------
yfinance weekly bars are timestamped Monday but contain the full week's data
(the close is Friday's close). All features built from a bar are therefore
"as of Friday close" and the target is the *next* week's return. Fama-French
weekly factors are dated Friday of the same week, so they align with the bar
that contains that week's data -- no lookahead.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Fallback universe: 30 mega/large-cap names, used only if the Wikipedia
# S&P 500 list cannot be fetched (no internet to Wikipedia, layout change).
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO",
    "LLY", "JPM", "V", "MA", "XOM", "UNH", "HD", "COST", "PG", "NFLX",
    "CRM", "ORCL", "WMT", "BAC", "ABBV", "KO", "AMD", "ADBE", "PEP",
    "CSCO", "TMO", "ACN",
]

FRENCH_WEEKLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
)

# FRED macro series (Phase 3). Direct REST calls -- no extra dependency.
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = ["DGS10", "VIXCLS", "T10Y2Y", "DGS2"]

CACHE_MAX_AGE_DAYS = 7

# Cache location, resolved against the project root (not the caller's cwd):
# notebooks run with cwd=notebooks/, main.py with cwd=project root -- both
# must share ONE cache, otherwise results silently diverge between runs.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Default data dir (project-root ``data/``) when none is given."""
    return DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
MARKET_CAP_CACHE_DAYS = 30


def get_universe(
    n: int = 50,
    data_dir: str | Path | None = None,
    order: str = "name",
) -> list[str]:
    """Return the first ``n`` S&P 500 tickers.

    Tries the Wikipedia S&P 500 list first; falls back to a hardcoded top-30
    large-cap list if that fails. With ``order="name"`` (default) the
    Wikipedia table is ordered by company name, so "first n" is a
    name-ordered subset, not the n largest companies. With
    ``order="market-cap"`` the list is ordered by market capitalisation
    (see ``get_universe_market_cap``).
    """
    if order == "market-cap":
        return get_universe_market_cap(n=n, data_dir=data_dir)
    if order != "name":
        raise ValueError("order must be 'name' or 'market-cap'")
    try:
        # Wikipedia blocks pandas' default user agent (403); fetch the page
        # with a browser UA and parse the HTML string instead.
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/120.0 Safari/537.36"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        symbols = tables[0]["Symbol"].astype(str).tolist()
        # Yahoo uses '-' where Wikipedia uses '.' (BRK.B -> BRK-B).
        tickers = [s.replace(".", "-").strip() for s in symbols]
        tickers = [t for t in tickers if t and t != "nan"]
        print(f"Universe: {len(tickers)} tickers from Wikipedia S&P 500 list.")
        return tickers[:n]
    except Exception as exc:  # noqa: BLE001 - any fetch/parse failure
        print(f"Wikipedia S&P 500 fetch failed ({exc}); using fallback list.")
        return FALLBACK_TICKERS[:n]


def _fetch_one_market_cap(ticker: str) -> tuple[str, float | None]:
    """Market cap for one ticker via yfinance fast_info (light quote call)."""
    try:
        info = yf.Ticker(ticker).fast_info
        # yfinance 1.8 uses camelCase ('marketCap'); older versions used
        # snake_case ('market_cap'). Accept either.
        cap = info.get("marketCap", info.get("market_cap"))
        return ticker, (float(cap) if cap else None)
    except Exception:  # noqa: BLE001 - per-ticker failure, skip it
        return ticker, None


def get_market_caps(
    tickers: list[str],
    data_dir: str | Path | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Market capitalisation for ``tickers`` (cached 30 days).

    One light quote call per ticker, fetched concurrently. Results are
    cached to ``data/market_caps.parquet`` so the ~500 calls happen at
    most once a month -- do not call this in a tight loop. Tickers whose
    quote fails are dropped (with a count in the log line).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "market_caps.parquet"
    meta_path = data_dir / "market_caps.meta.json"

    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            saved = datetime.fromisoformat(meta["saved_at"])
            cached = pd.read_parquet(cache_path)
            have = set(cached["ticker"])
            if (datetime.now() - saved < timedelta(days=MARKET_CAP_CACHE_DAYS)
                    and set(tickers) <= have):
                print(f"Market caps: using cache {cache_path} "
                      f"({len(cached)} tickers, saved {meta['saved_at'][:10]})")
                return cached[cached["ticker"].isin(tickers)]
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # corrupt cache -> refetch below

    print(f"Market caps: fetching {len(tickers)} quotes "
          f"({max_workers} workers, one light call each) ...")
    results: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_market_cap, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker, cap = fut.result()
            if cap is not None and cap > 0:
                results[ticker] = cap
    df = pd.DataFrame(
        sorted(results.items()), columns=["ticker", "market_cap"])
    failed = len(tickers) - len(df)
    print(f"Market caps: got {len(df)} / {len(tickers)} "
          f"({failed} failed, dropped).")
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(
        {"tickers": sorted(tickers),
         "saved_at": datetime.now().isoformat()}))
    return df


def get_universe_market_cap(
    n: int = 50,
    data_dir: str | Path | None = None,
) -> list[str]:
    """Return the ``n`` largest S&P 500 stocks by market cap.

    Fetches the full Wikipedia list, then orders it by market cap
    (cached -- see ``get_market_caps``). Falls back to the name-ordered
    universe if the market-cap fetch returns too few names.
    """
    all_tickers = get_universe(n=10_000, data_dir=data_dir)
    caps = get_market_caps(all_tickers, data_dir=data_dir)
    if len(caps) < n:
        print(f"Market-cap ordering: only {len(caps)} caps available, "
              f"falling back to name order.")
        return all_tickers[:n]
    ordered = caps.sort_values("market_cap", ascending=False)["ticker"].tolist()
    print(f"Universe (market-cap order): top {n} of {len(ordered)} "
          f"({ordered[0]}, {ordered[1]}, {ordered[2]}, ...).")
    return ordered[:n]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_is_fresh(meta_path: Path, tickers: list[str], years: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("tickers") != sorted(tickers) or meta.get("years") != years:
        return False
    saved = datetime.fromisoformat(meta["saved_at"])
    return datetime.now() - saved < timedelta(days=CACHE_MAX_AGE_DAYS)


def _write_meta(meta_path: Path, tickers: list[str], years: int) -> None:
    meta_path.write_text(json.dumps({
        "tickers": sorted(tickers),
        "years": years,
        "saved_at": datetime.now().isoformat(),
    }))


# ---------------------------------------------------------------------------
# Prices (yfinance)
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str],
    years: int = 5,
    data_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download weekly OHLCV bars from yfinance (cached as parquet).

    Returns a long DataFrame with columns:
    ``date, ticker, open, high, low, close, volume``.
    """
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "prices_weekly.parquet"
    meta_path = data_dir / "prices_weekly.meta.json"

    if (
        not force_refresh
        and cache_path.exists()
        and _cache_is_fresh(meta_path, tickers, years)
    ):
        print(f"Prices: using cache {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Prices: downloading {len(tickers)} tickers, {years}y weekly ...")
    raw = yf.download(
        tickers,
        period=f"{years}y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no price data.")

    # yfinance returns MultiIndex columns (field, ticker) for >1 ticker and
    # flat columns for a single ticker. Normalise to long format.
    if isinstance(raw.columns, pd.MultiIndex):
        stacked = raw.stack(level=1)  # index: date, ticker
    else:
        stacked = raw.copy()
        stacked["ticker"] = tickers[0]
        stacked = stacked.set_index("ticker", append=True)
    stacked.index.names = ["date", "ticker"]
    df = stacked.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop rows with no trade (all-NaN bars, e.g. IPO week gaps).
    df = df.dropna(subset=["close"])

    df.to_parquet(cache_path, index=False)
    _write_meta(meta_path, tickers, years)
    print(f"Prices: {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Macro data: yfinance proxies (fallback) + FRED (primary, Phase 3)
# ---------------------------------------------------------------------------
def download_macro(
    data_dir: str | Path | None = None,
    years: int = 5,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download weekly ^VIX and ^TNX closes (macro proxies, cached).

    Returns a DataFrame with columns ``date, vix, tnx``.
    """
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "macro_weekly.parquet"
    meta_path = data_dir / "macro_weekly.meta.json"

    if (
        not force_refresh
        and cache_path.exists()
        and _cache_is_fresh(meta_path, ["^VIX", "^TNX"], years)
    ):
        print(f"Macro: using cache {cache_path}")
        return pd.read_parquet(cache_path)

    print("Macro: downloading ^VIX and ^TNX weekly ...")
    raw = yf.download(
        ["^VIX", "^TNX"],
        period=f"{years}y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no macro data.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index),
        "vix": close["^VIX"].to_numpy(dtype=float),
        "tnx": close["^TNX"].to_numpy(dtype=float),
    })
    out = out.dropna().reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    _write_meta(meta_path, ["^VIX", "^TNX"], years)
    print(f"Macro: {len(out)} weeks, {out['date'].min().date()} -> "
          f"{out['date'].max().date()}")
    return out


def download_fred(
    series: list[str] | None = None,
    years: int = 5,
    data_dir: str | Path | None = None,
    force_refresh: bool = False,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download weekly FRED macro series (cached).

    Series (Phase 3 -- the primary macro source when a key is available):

    * ``DGS10`` -- 10-year Treasury constant-maturity yield (%).
    * ``VIXCLS`` -- CBOE Volatility Index, close (index level).
    * ``T10Y2Y`` -- 10Y minus 2Y Treasury spread (percentage points);
      the classic yield-curve / recession signal.
    * ``DGS2`` -- 2-year Treasury constant-maturity yield (%).

    The key is read from the ``FRED_API_KEY`` environment variable
    (free key from fred.stlouisfed.org) or the ``api_key`` argument.
    **The key is never written to any file or log by this function.**
    Values are weekly end-of-period (Friday) observations, so they align
    with the Friday-close information in each weekly price bar.

    Security note: the key travels as a query parameter in the request
    URL, and ``requests`` puts the full URL into HTTP error messages.
    Any request failure is therefore re-raised as a sanitised error
    carrying only the series id and the HTTP status -- never the URL.

    Returns a DataFrame with columns ``date, dgs10, vixcls, t10y2y,
    dgs2`` (lowercase series ids), cached to ``data/fred_weekly.parquet``.
    """
    import os

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED download needs an API key: set the FRED_API_KEY "
            "environment variable (free key from fred.stlouisfed.org). "
            "Without a key, use download_macro() "
            "(yfinance ^VIX/^TNX proxies)."
        )
    series = list(series or FRED_SERIES)
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "fred_weekly.parquet"
    meta_path = data_dir / "fred_weekly.meta.json"

    if (
        not force_refresh
        and cache_path.exists()
        and _cache_is_fresh(meta_path, series, years)
    ):
        print(f"FRED: using cache {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"FRED: downloading {', '.join(series)} weekly ...")
    # One extra year of buffer so the first usable week is fully covered
    # after the feature lookbacks are applied downstream.
    start = (datetime.now() - timedelta(days=365 * (years + 1))).strftime(
        "%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    for sid in series:
        try:
            resp = requests.get(
                FRED_API_BASE,
                params={
                    "series_id": sid,
                    "api_key": key,
                    "file_type": "json",
                    "frequency": "w",
                    "aggregation_method": "eop",
                    "observation_start": start,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            # Sanitised: never leak the request URL (it carries the key).
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            raise RuntimeError(
                f"FRED request for series {sid} failed (HTTP {status}). "
                f"Check that FRED_API_KEY is valid."
            ) from None
        obs = resp.json().get("observations", [])
        if not obs:
            raise RuntimeError(
                f"FRED returned no observations for series {sid}.")
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        # "." marks missing values in FRED responses.
        df[sid.lower()] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df[["date", sid.lower()]])
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    _write_meta(meta_path, series, years)
    print(f"FRED: {len(out)} weeks, {out['date'].min().date()} -> "
          f"{out['date'].max().date()} "
          f"(missing values per series: "
          f"{ {c: int(out[c].isna().sum()) for c in out.columns if c != 'date'} })")
    return out


# ---------------------------------------------------------------------------
# Fama-French weekly factors
# ---------------------------------------------------------------------------
def download_french_weekly(
    data_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download Fama-French weekly research factors (Mkt-RF, SMB, HML, RF).

    Source: Ken French data library. The CSV inside the zip has header text
    rows and a copyright footer; only lines starting with an 8-digit date
    are data. Values are in percent and are converted to decimals.

    Returns a DataFrame with columns ``date, mkt_rf, smb, hml, rf`` where
    ``date`` is the Friday of each week.
    """
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "french_weekly.parquet"

    if not force_refresh and cache_path.exists():
        print(f"French factors: using cache {cache_path}")
        return pd.read_parquet(cache_path)

    print("French factors: downloading from Ken French data library ...")
    resp = requests.get(FRENCH_WEEKLY_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Member name case varies ('.csv' vs '.CSV'); match case-insensitively.
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(
                f"No CSV member in the French factors zip (members: "
                f"{zf.namelist()}).")
        text = zf.read(names[0]).decode("utf-8", errors="ignore")

    # Data lines look like: 20260702,   0.50,  -0.12,   0.33,   0.08
    # Header/footer lines do not start with 8 digits + comma.
    rows = []
    for line in text.splitlines():
        if re.match(r"^\d{8},", line.strip()):
            parts = [p.strip() for p in line.strip().split(",")]
            rows.append(parts)
    if not rows:
        raise RuntimeError("No data rows found in the French factors file "
                           "(format may have changed).")
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for col in ["mkt_rf", "smb", "hml", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # -99.99 marks missing values in French files.
    df = df.replace(-99.99, np.nan).dropna().reset_index(drop=True)
    # Percent -> decimal.
    for col in ["mkt_rf", "smb", "hml", "rf"]:
        df[col] = df[col] / 100.0
    df.to_parquet(cache_path, index=False)
    print(f"French factors: {len(df)} weeks, {df['date'].min().date()} -> "
          f"{df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Benchmark (SPY buy-and-hold)
# ---------------------------------------------------------------------------
def download_benchmark(
    years: int = 5,
    data_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download weekly SPY closes (benchmark), cached.

    Returns a DataFrame with columns ``date, spy_close, spy_ret`` where
    ``spy_ret`` is the week-over-week return.
    """
    data_dir = _resolve_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "spy_weekly.parquet"
    meta_path = data_dir / "spy_weekly.meta.json"

    if (
        not force_refresh
        and cache_path.exists()
        and _cache_is_fresh(meta_path, ["SPY"], years)
    ):
        print(f"Benchmark: using cache {cache_path}")
        return pd.read_parquet(cache_path)

    print("Benchmark: downloading SPY weekly ...")
    raw = yf.download(
        "SPY", period=f"{years}y", interval="1wk",
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no SPY data.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        # yfinance >= 0.2.40 returns MultiIndex columns even for one ticker,
        # so Close is a one-column DataFrame; take the series.
        close = close.iloc[:, 0]
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index),
        "spy_close": close.to_numpy(dtype=float),
    })
    out["spy_ret"] = out["spy_close"].pct_change()
    out = out.dropna().reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    _write_meta(meta_path, ["SPY"], years)
    return out
