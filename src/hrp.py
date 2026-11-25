"""HRP -- Hierarchical Risk Parity (Lopez de Prado, 2016).

"Building Diversified Portfolios that Outperform Out of Sample"
(Journal of Portfolio Management, 2016).

HRP is a *portfolio construction* method, not a prediction model: it answers
"how much weight should each selected stock get?", not "which stocks will go
up?". It needs only the covariance matrix (no expected-return estimates) and
never inverts it, which is why its weights are far more stable out-of-sample
than Markowitz mean-variance weights.

Algorithm (implemented from scratch with scipy):
  1. Cluster: hierarchical clustering on the correlation distance
     d = sqrt(0.5 * (1 - rho)).
  2. Quasi-diagonalisation: reorder the covariance matrix by the dendrogram
     leaf order so similar assets sit next to each other.
  3. Recursive bisection: split the ordered assets in two, allocate between
     the halves in inverse proportion to their cluster variances, recurse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix -> distance matrix, d = sqrt(0.5 * (1 - rho))."""
    corr = corr.clip(-1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order of the dendrogram (seriation)."""
    return dendrogram(link, no_plot=True)["leaves"]


def _cluster_variance(cov: pd.DataFrame, tickers: list) -> float:
    """Variance of the inverse-variance portfolio of a cluster."""
    sub = cov.loc[tickers, tickers]
    ivp = 1.0 / np.diag(sub.values)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub.values @ ivp)


def _inverse_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Inverse-variance portfolio weights for the assets in ``cov``."""
    ivp = 1.0 / np.diag(cov.values)
    ivp = ivp / ivp.sum()
    return pd.Series(ivp, index=cov.index)


def _recursive_bisection(
    ordered_cov: pd.DataFrame, ordered_tickers: list
) -> pd.Series:
    """Allocate weights by recursive bisection on the ordered assets."""
    weights = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]  # clusters still to split
    while clusters:
        # Split each cluster in half (on the quasi-diagonal order).
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            # Allocate between the two halves by inverse cluster variance.
            var_left = _cluster_variance(ordered_cov, left)
            var_right = _cluster_variance(ordered_cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return weights / weights.sum()


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP weights from a trailing returns panel.

    Parameters
    ----------
    returns : DataFrame of asset returns, rows = weeks, columns = tickers
        (e.g. the trailing 26 weeks of weekly returns for the selected
        stocks).

    Returns
    -------
    Series of weights indexed by ticker, summing to 1, all >= 0.
    """
    returns = returns.dropna(axis=1, how="all")
    tickers = list(returns.columns)
    if len(tickers) == 0:
        raise ValueError("hrp_weights: no tickers with return data.")
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)

    cov = returns.cov()
    corr = returns.corr()
    # Guard against NaN / constant series (e.g. a halted stock).
    bad = cov.isna().any() | (np.diag(cov.values) <= 0)
    if bad.any():
        keep = [t for t, b in zip(tickers, bad) if not b]
        if not keep:
            return pd.Series(1.0 / len(tickers), index=tickers)
        returns = returns[keep]
        tickers = keep
        cov = returns.cov()
        corr = returns.corr()
        if len(tickers) == 1:
            return pd.Series([1.0], index=tickers)

    dist = _corr_to_distance(corr)
    # Single linkage, as in Lopez de Prado's original paper.
    link = linkage(squareform(dist.values, checks=False), method="single")
    order = _quasi_diagonal_order(link)
    ordered_tickers = [tickers[i] for i in order]
    ordered_cov = cov.loc[ordered_tickers, ordered_tickers]

    weights = _recursive_bisection(ordered_cov, ordered_tickers)
    # Numerical safety: clip tiny negatives, renormalise.
    weights = weights.clip(lower=0.0)
    return weights / weights.sum()


def hrp_weights_for_week(
    prices: pd.DataFrame,
    tickers: list[str],
    asof: pd.Timestamp,
    lookback_weeks: int = 26,
) -> pd.Series:
    """Convenience wrapper: HRP weights for ``tickers`` as of week ``asof``.

    Uses only returns up to and including ``asof`` (point-in-time correct).
    Falls back to equal weights if fewer than 8 weeks of history exist.
    """
    hist = prices[(prices["date"] <= asof) & (prices["ticker"].isin(tickers))]
    piv = hist.pivot(index="date", columns="ticker", values="close").sort_index()
    piv = piv.tail(lookback_weeks + 1)
    rets = np.log(piv / piv.shift(1)).dropna(how="all")
    if len(rets) < 8 or rets.shape[1] == 0:
        return pd.Series(1.0 / len(tickers), index=tickers)
    present = [t for t in tickers if t in rets.columns]
    w = hrp_weights(rets[present])
    # Any ticker missing from returns gets zero weight; renormalise.
    return w.reindex(tickers, fill_value=0.0).pipe(lambda s: s / s.sum())
