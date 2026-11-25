"""Generate the 5 Phase 2 notebooks (see notebooks/EDA_outline.md).

Usage: .venv/bin/python scripts/build_notebooks.py

Each notebook runs against the cached data and the src modules (same code
path as main.py). Execute with:
  .venv/bin/jupyter nbconvert --execute --to notebook --inplace \
      --ExecutePreprocessor.timeout=900 notebooks/*.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src import features as feat_mod
from src import models as models_mod
from src import simulate as sim_mod
from src.target import make_target, time_split

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
plt.rcParams["figure.figsize"] = (10, 5)
"""

LOAD_50_5Y = """\
# Same parameters as the fuller main.py run (50 tickers, 5 years).
# Downloads are cached under data/ (7-day freshness), so re-running is fast.
tickers = data_mod.get_universe(n=50)
prices = data_mod.download_prices(tickers, years=5)
macro = data_mod.download_macro(years=5)
french = data_mod.download_french_weekly()
spy = data_mod.download_benchmark(years=5)
"""


def nb(cells, title: str) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": title,
    }
    return notebook


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------------
# 01 - Data overview
# ---------------------------------------------------------------------------
def build_01() -> nbf.NotebookNode:
    return nb([
        md("""\
# 01 — Data Overview

Universe coverage, missing data, price/volume sanity checks, weekly return
distribution, and the universe vs SPY comparison. All data comes from the
same `src/data.py` code path as `main.py` (yfinance weekly bars, 50
tickers, 5 years; cached under `data/`).
"""),
        code(SETUP + "\n" + LOAD_50_5Y),
        md("## Universe coverage"),
        code("""\
print(f"Tickers: {prices['ticker'].nunique()}")
print(f"Weeks: {prices['date'].nunique()} "
      f"({prices['date'].min().date()} -> {prices['date'].max().date()})")
print(f"Rows: {len(prices):,}")
rows_per_ticker = prices.groupby('ticker').size()
print(rows_per_ticker.describe())
print("\\nTickers with the fewest weeks (recent IPOs / gaps):")
print(rows_per_ticker.nsmallest(5))
"""),
        md("""\
## Missing data

A week is "missing" for a ticker if it has no bar in that week. Recent
IPOs legitimately start late; anything else is a data gap worth knowing
about before modelling.
"""),
        code("""\
piv = prices.pivot(index='date', columns='ticker', values='close')
missing = piv.isna()
print(f"Missing cells: {missing.sum().sum():,} of {missing.size:,} "
      f"({100 * missing.mean().mean():.2f}%)")
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(missing.T.values, aspect='auto', cmap='Greys', interpolation='nearest')
ax.set_title('Missing weekly bars (white = missing)')
ax.set_xlabel('week index')
ax.set_ylabel('ticker index')
ax.set_yticks([])
plt.tight_layout()
plt.show()
"""),
        md("## Price / volume sanity checks"),
        code("""\
print(prices[['open', 'high', 'low', 'close', 'volume']].describe().T)
# Suspicious bars: weekly moves over +-40% are usually splits or bad ticks
# (yfinance auto-adjusts for splits, so survivors are genuine jumps).
prices = prices.sort_values(['ticker', 'date'])
prices['wk_ret'] = prices.groupby('ticker')['close'].pct_change()
big = prices[prices['wk_ret'].abs() > 0.40].sort_values('wk_ret')
print(f"\\nWeeks with |return| > 40%: {len(big)}")
print(big[['date', 'ticker', 'close', 'wk_ret']].head(10).to_string(index=False))
"""),
        md("## Weekly return distribution — fat tails?"),
        code("""\
rets = prices['wk_ret'].dropna()
print(f"mean {rets.mean():.4%}, std {rets.std():.4%}, "
      f"skew {rets.skew():.2f}, kurtosis {rets.kurtosis():.2f}")
print(f"1st pct {rets.quantile(0.01):.2%}, 99th pct {rets.quantile(0.99):.2%}")
fig, ax = plt.subplots()
ax.hist(rets.clip(-0.3, 0.3), bins=80)
ax.set_title('Weekly returns, all tickers (clipped to +-30%)')
ax.set_xlabel('weekly return')
ax.set_ylabel('count')
plt.tight_layout()
plt.show()
"""),
        md("## Universe vs SPY"),
        code("""\
uni_weekly = prices.groupby('date')['wk_ret'].mean().rename('universe_mean')
cmp = pd.DataFrame({'universe_mean': uni_weekly}).join(
    spy.set_index('date')['spy_ret'])
cmp = cmp.dropna()
print(cmp.describe().T)
print(f"\\nCorrelation(universe mean weekly return, SPY): "
      f"{cmp['universe_mean'].corr(cmp['spy_ret']):.3f}")
print(f"Universe mean weekly return: {cmp['universe_mean'].mean():.4%}")
print(f"SPY mean weekly return:      {cmp['spy_ret'].mean():.4%}")
"""),
    ], "01 — Data Overview")


# ---------------------------------------------------------------------------
# 02 - Features EDA
# ---------------------------------------------------------------------------
def build_02() -> nbf.NotebookNode:
    return nb([
        md("""\
# 02 — Features EDA

Distributions of the NUMERIC features, their correlation with the target
and with next-week return, multicollinearity between features, and
decile analysis for momentum and volatility (do high-momentum stocks
actually outperform next week?).
"""),
        code(SETUP + "\n" + LOAD_50_5Y + """
feat = feat_mod.build_features(prices, macro=macro, french=french)
panel = make_target(feat)
numeric, dummies = feat_mod.feature_columns(panel)
print(f"Panel: {len(panel):,} rows, {len(numeric)} numeric + "
      f"{len(dummies)} dummy features")
"""),
        md("## Feature distributions"),
        code("""\
print(panel[numeric].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3))
"""),
        md("""\
## Correlation with the target and with next-week return

Point-biserial correlation with the binary target (= Pearson on a 0/1
variable) and Pearson correlation with realised next-week return. Weekly
stock returns are noisy, so expect small numbers -- anything above ~0.05
in absolute value is already interesting.
"""),
        code("""\
corr_t = panel[numeric + ['target']].corr()['target'].drop('target')
corr_r = panel[numeric + ['next_ret']].corr()['next_ret'].drop('next_ret')
ct = pd.DataFrame({'corr_target': corr_t, 'corr_next_ret': corr_r})
ct = ct.reindex(ct['corr_target'].abs().sort_values(ascending=False).index)
print(ct.round(3))
fig, ax = plt.subplots(figsize=(10, 6))
ct['corr_target'].sort_values().plot.barh(ax=ax)
ax.set_title('Correlation of each feature with the top-quintile target')
ax.set_xlabel('point-biserial correlation')
plt.tight_layout()
plt.show()
"""),
        md("""\
## Multicollinearity between features

Highly correlated features (e.g. ret_4w vs ret_12w, vol_4w vs vol_12w)
are expected -- they are nested windows of the same series. The Ridge
factor regression handles this with regularisation; tree models are
largely unaffected.
"""),
        code("""\
c = panel[numeric].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(c.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric)))
ax.set_xticklabels(numeric, rotation=90)
ax.set_yticks(range(len(numeric)))
ax.set_yticklabels(numeric)
ax.set_title('Feature correlation matrix')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
# Strongest off-diagonal pairs.
pairs = c.where(np.triu(np.ones_like(c, dtype=bool), k=1)).stack()
print(pairs.abs().sort_values(ascending=False).head(8).round(3))
"""),
        md("""\
## Decile analysis: momentum and volatility

Split all rows into deciles by a feature and look at the mean next-week
return (and top-quintile hit rate) per decile. A monotone pattern means
the feature carries signal; a flat pattern means it does not.
"""),
        code("""\
def decile_table(col: str) -> pd.DataFrame:
    d = panel[['date', 'ticker', col, 'next_ret', 'target']].copy()
    d['decile'] = pd.qcut(d[col], 10, labels=False, duplicates='drop') + 1
    return d.groupby('decile').agg(
        mean_next_ret=('next_ret', 'mean'),
        hit_rate=('target', 'mean'),
        n=('next_ret', 'size')).round(4)

for col in ['ret_12w', 'vol_12w', 'rsi_14']:
    t = decile_table(col)
    print(f"\\n=== {col} deciles ===")
    print(t)
    fig, ax = plt.subplots()
    t['mean_next_ret'].plot.bar(ax=ax)
    ax.set_title(f'Mean next-week return by {col} decile')
    ax.set_xlabel('decile (1 = lowest)')
    ax.set_ylabel('mean next-week return')
    plt.tight_layout()
    plt.show()
"""),
    ], "02 — Features EDA")


# ---------------------------------------------------------------------------
# 03 - Target analysis
# ---------------------------------------------------------------------------
def build_03() -> nbf.NotebookNode:
    return nb([
        md("""\
# 03 — Target Analysis

Base rate of the top-quintile label by year and by month (seasonality?),
persistence of winners (does last week's top quintile stay top?), sector
concentration of the picks, and the return distribution of top-quintile
stocks vs the rest.
"""),
        code(SETUP + "\n" + LOAD_50_5Y + """
feat = feat_mod.build_features(prices, macro=macro, french=french)
panel = make_target(feat)
panel['year'] = pd.to_datetime(panel['date']).dt.year
panel['month'] = pd.to_datetime(panel['date']).dt.month
print(f"Panel: {len(panel):,} rows, base rate {panel['target'].mean():.3f}")
"""),
        md("## Base rate by year and by month"),
        code("""\
by_year = panel.groupby('year')['target'].agg(['mean', 'size']).round(3)
by_year.columns = ['top_quintile_rate', 'n']
print(by_year)
fig, ax = plt.subplots()
by_year['top_quintile_rate'].plot.bar(ax=ax)
ax.axhline(0.20, color='red', linestyle='--', label='expected 0.20')
ax.set_title('Top-quintile base rate by year')
ax.legend()
plt.tight_layout()
plt.show()
by_month = panel.groupby('month')['target'].agg(['mean', 'size']).round(3)
by_month.columns = ['top_quintile_rate', 'n']
print(by_month)
fig, ax = plt.subplots()
by_month['top_quintile_rate'].plot.bar(ax=ax)
ax.axhline(0.20, color='red', linestyle='--', label='expected 0.20')
ax.set_title('Top-quintile base rate by month (seasonality check)')
ax.set_xlabel('month')
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""\
## Persistence of winners

If last week's winners tend to stay winners, momentum is real in this
universe and the target is partly predictable from last week's outcome
alone. Compare against the 20% base rate.
"""),
        code("""\
p = panel.sort_values(['ticker', 'date']).copy()
p['prev_target'] = p.groupby('ticker')['target'].shift(1)
p = p.dropna(subset=['prev_target'])
stay = p[p['prev_target'] == 1]['target'].mean()
leave = p[p['prev_target'] == 0]['target'].mean()
print(f"P(top this week | top last week)     = {stay:.3f}")
print(f"P(top this week | not top last week) = {leave:.3f}")
print(f"Base rate                            = 0.200")
p['prev2_target'] = p.groupby('ticker')['target'].shift(2)
p2 = p.dropna(subset=['prev2_target'])
stay2 = p2[(p2['prev_target'] == 1) & (p2['prev2_target'] == 1)]['target'].mean()
print(f"P(top this week | top last 2 weeks)  = {stay2:.3f}")
"""),
        md("""\
## Sector concentration of the picks

Sectors come from the Wikipedia S&P 500 table (GICS Sector column).
If the top-quintile picks concentrate in one or two sectors, the
strategy is implicitly making a sector bet -- worth knowing before
trading it.
"""),
        code("""\
import io
import requests

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0 Safari/537.36"}
resp = requests.get(data_mod.WIKI_SP500_URL, headers=headers, timeout=30)
wiki = pd.read_html(io.StringIO(resp.text))[0]
wiki['Symbol'] = wiki['Symbol'].str.replace('.', '-', regex=False)
sector = dict(zip(wiki['Symbol'], wiki['GICS Sector']))
panel['sector'] = panel['ticker'].map(sector)
picks = panel[panel['target'] == 1]
uni_share = panel['sector'].value_counts(normalize=True)
pick_share = picks['sector'].value_counts(normalize=True)
cmp = pd.DataFrame({'universe_share': uni_share,
                    'pick_share': pick_share}).fillna(0).round(3)
cmp['lift'] = (cmp['pick_share'] / cmp['universe_share']).round(2)
print(cmp.sort_values('pick_share', ascending=False))
"""),
        md("## Return distribution: top quintile vs the rest"),
        code("""\
top = panel[panel['target'] == 1]['next_ret']
rest = panel[panel['target'] == 0]['next_ret']
print(f"Top quintile next-week return: mean {top.mean():.2%}, "
      f"median {top.median():.2%}")
print(f"Rest next-week return:         mean {rest.mean():.2%}, "
      f"median {rest.median():.2%}")
fig, ax = plt.subplots()
ax.hist(rest.clip(-0.2, 0.2), bins=60, alpha=0.6, label='rest')
ax.hist(top.clip(-0.2, 0.2), bins=60, alpha=0.6, label='top quintile')
ax.set_title('Next-week return: top quintile vs rest (clipped to +-20%)')
ax.set_xlabel('next-week return')
ax.legend()
plt.tight_layout()
plt.show()
"""),
    ], "03 — Target Analysis")


# ---------------------------------------------------------------------------
# 04 - Model comparison (incl. LSTM)
# ---------------------------------------------------------------------------
def build_04() -> nbf.NotebookNode:
    return nb([
        md("""\
# 04 — Model Comparison

ROC AUC / precision / recall per model on the same time-based test split
as `main.py`, feature importances, probability calibration, and error
analysis by ticker and by month. The LSTM runs only if TensorFlow is
installed (it is an optional dependency -- see README).
"""),
        code(SETUP + """
from src import lstm_model as lstm_mod
""" + "\n" + LOAD_50_5Y + """
feat = feat_mod.build_features(prices, macro=macro, french=french)
panel = make_target(feat)
numeric, dummies = feat_mod.feature_columns(panel)
feature_cols = numeric + dummies
train, test, cutoff = time_split(panel, train_frac=0.7)
print(f"Cutoff: {cutoff.date()}")
"""),
        md("## Train the models"),
        code("""\
from sklearn.metrics import roc_auc_score

scores = {}   # name -> test scores
rf = models_mod.train_random_forest(train[feature_cols], train['target'])
scores['RandomForest'] = models_mod.score_random_forest(rf, test[feature_cols])
ff = models_mod.train_factor_regression(train[feature_cols], train['next_ret'])
scores['FF-regression'] = models_mod.score_factor_regression(ff, test[feature_cols])
if models_mod.xgboost_available():
    xgb = models_mod.train_xgboost(train[feature_cols], train['target'])
    scores['XGBoost'] = models_mod.score_xgboost(xgb, test[feature_cols])
else:
    print("XGBoost not installed, skipping.")

# LSTM (optional): sequences from the full panel, same cutoff split.
lstm_meta = None
if lstm_mod.tensorflow_available():
    X_seq, y_seq, meta = lstm_mod.build_sequences(panel, feature_cols)
    te = (pd.to_datetime(meta['date']) >= cutoff).to_numpy()
    tr = ~te
    lstm, scaler, _, _ = lstm_mod.train_lstm(
        X_seq[tr], y_seq[tr], meta.loc[tr, 'date'])
    lstm_meta = meta[te].copy()
    lstm_meta['score'] = lstm_mod.score_lstm(lstm, scaler, X_seq[te])
    scores['LSTM'] = None  # scored on meta rows, handled separately below
    print(f"LSTM test rows: {len(lstm_meta)}")
else:
    print("TensorFlow not installed -- LSTM skipped (see README).")
"""),
        md("## ROC AUC / precision / recall"),
        code("""\
from sklearn.metrics import precision_score, recall_score

rows = []
for name, s in scores.items():
    if s is None:
        continue
    y_pred = (s >= 0.5).astype(int)
    rows.append({'model': name,
                 'roc_auc': roc_auc_score(test['target'], s),
                 'precision@0.5': precision_score(test['target'], y_pred, zero_division=0),
                 'recall@0.5': recall_score(test['target'], y_pred, zero_division=0)})
if lstm_meta is not None:
    s = lstm_meta['score'].to_numpy()
    y_pred = (s >= 0.5).astype(int)
    rows.append({'model': 'LSTM',
                 'roc_auc': roc_auc_score(lstm_meta['target'], s),
                 'precision@0.5': precision_score(lstm_meta['target'], y_pred, zero_division=0),
                 'recall@0.5': recall_score(lstm_meta['target'], y_pred, zero_division=0)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
"""),
        md("## Feature importances (RF + XGBoost)"),
        code("""\
imp = pd.Series(rf.feature_importances_, index=feature_cols)
print("RandomForest top 15:")
print(imp.sort_values(ascending=False).head(15).round(3))
fig, ax = plt.subplots()
imp.sort_values(ascending=False).head(15).sort_values().plot.barh(ax=ax)
ax.set_title('RandomForest feature importances (top 15)')
plt.tight_layout()
plt.show()
if models_mod.xgboost_available():
    imp_x = pd.Series(xgb.feature_importances_, index=feature_cols)
    fig, ax = plt.subplots()
    imp_x.sort_values(ascending=False).head(15).sort_values().plot.barh(ax=ax)
    ax.set_title('XGBoost feature importances (top 15)')
    plt.tight_layout()
    plt.show()
"""),
        md("""\
## Calibration of predicted probabilities

Bin the test rows by predicted probability decile and compare the mean
predicted probability with the actual positive rate in each bin. A
well-calibrated model tracks the diagonal; systematic gaps mean the
probabilities should not be read literally (ranking still works).
"""),
        code("""\
cal = test[['target']].copy()
cal['score'] = scores['RandomForest']
cal['bin'] = pd.qcut(cal['score'], 10, labels=False, duplicates='drop')
g = cal.groupby('bin').agg(mean_pred=('score', 'mean'),
                           actual_rate=('target', 'mean'),
                           n=('target', 'size'))
print(g.round(3))
fig, ax = plt.subplots()
ax.plot(g['mean_pred'], g['actual_rate'], marker='o', label='RandomForest')
ax.plot([0, 1], [0, 1], '--', color='grey', label='perfect calibration')
ax.set_xlabel('mean predicted P(top-quintile)')
ax.set_ylabel('actual top-quintile rate')
ax.set_title('Calibration (RandomForest, test period)')
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("## Error analysis: by ticker and by month"),
        code("""\
test_rf = test.copy()
test_rf['score'] = scores['RandomForest']
# Per-ticker AUC (test rows only, min 15 rows to be meaningful).
aucs = []
for t, g in test_rf.groupby('ticker'):
    if len(g) >= 15 and g['target'].nunique() > 1:
        aucs.append((t, len(g), roc_auc_score(g['target'], g['score'])))
aucs = pd.DataFrame(aucs, columns=['ticker', 'n', 'auc']).sort_values('auc')
print("Worst 5 tickers by test AUC:")
print(aucs.head(5).round(3).to_string(index=False))
print("\\nBest 5 tickers by test AUC:")
print(aucs.tail(5).round(3).to_string(index=False))
test_rf['month'] = pd.to_datetime(test_rf['date']).dt.month
bym = test_rf.groupby('month').apply(
    lambda g: roc_auc_score(g['target'], g['score'])
    if g['target'].nunique() > 1 else np.nan, include_groups=False)
print("\\nTest AUC by calendar month:")
print(bym.round(3))
"""),
    ], "04 — Model Comparison")


# ---------------------------------------------------------------------------
# 05 - Strategy deep dive
# ---------------------------------------------------------------------------
def build_05() -> nbf.NotebookNode:
    return nb([
        md("""\
# 05 — Strategy Deep Dive

Equity curves vs SPY, rolling Sharpe and rolling returns, drawdown
periods (what happened in the worst weeks?), HRP vs equal-weight weight
evolution, turnover and cost decomposition, and the long-short leg
breakdown.
"""),
        code(SETUP + "\n" + LOAD_50_5Y + """
feat = feat_mod.build_features(prices, macro=macro, french=french)
panel = make_target(feat)
numeric, dummies = feat_mod.feature_columns(panel)
feature_cols = numeric + dummies
train, test, cutoff = time_split(panel, train_frac=0.7)
rf = models_mod.train_random_forest(train[feature_cols], train['target'])
test_rf = models_mod.add_scores(
    test, rf, models_mod.score_random_forest, feature_cols)
"""),
        md("## Run the strategies"),
        code("""\
m_a, w_a = sim_mod.simulate(test_rf, prices, top_n=5, strategy='equal')
m_b, w_b = sim_mod.simulate(test_rf, prices, top_n=5, strategy='hrp')
m_ls, w_ls = sim_mod.simulate_long_short(test_rf, top_n=5)
m_spy = sim_mod.benchmark_metrics(spy, test['date'].unique())
m_exact, w_exact = sim_mod.simulate_exact(
    test_rf, prices, top_n=5, strategy='equal')
results = {
    'A: RF equal-weight': m_a,
    'B: RF HRP-weight': m_b,
    'F: RF long-short': m_ls,
    'A (exact sim)': m_exact,
    'Benchmark: SPY': m_spy,
}
print(sim_mod.metrics_table(results).round(4).to_string(index=False))
print(f"\\nExact vs vector gap (Strategy A): "
      f"{m_exact['total_return'] - m_a['total_return']:+.4%}")
"""),
        md("## Equity curves vs SPY"),
        code("""\
def equity(weekly: pd.DataFrame) -> pd.Series:
    eq = (1.0 + weekly.set_index('date')['ret']).cumprod()
    eq.index = pd.to_datetime(eq.index)
    return eq

fig, ax = plt.subplots()
for name, w in [('A: RF equal', w_a), ('B: RF HRP', w_b),
                ('F: long-short', w_ls)]:
    equity(w).plot(ax=ax, label=name)
# SPY equity over the same weeks.
spy_w = spy.sort_values('date').copy()
spy_w['next_spy_ret'] = spy_w['spy_ret'].shift(-1)
mask = spy_w['date'].isin(pd.to_datetime(test['date'].unique()))
spy_eq = (1.0 + spy_w.loc[mask, 'next_spy_ret'].dropna()).cumprod()
spy_eq.index = pd.to_datetime(spy_w.loc[mask, 'date'].iloc[:len(spy_eq)])
spy_eq.plot(ax=ax, label='SPY buy-and-hold', linestyle='--')
ax.set_title('Equity curves (test period, $1 invested)')
ax.set_ylabel('equity')
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("## Rolling Sharpe and rolling returns (Strategy A)"),
        code("""\
r = w_a.set_index('date')['ret']
roll_sharpe = r.rolling(12).mean() / r.rolling(12).std() * np.sqrt(52)
roll_ret4 = (1.0 + r).rolling(4).apply(lambda x: x.prod() - 1.0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
roll_sharpe.plot(ax=ax1)
ax1.axhline(0, color='grey', linestyle='--')
ax1.set_title('Rolling 12-week Sharpe (Strategy A)')
roll_ret4.plot(ax=ax2)
ax2.axhline(0, color='grey', linestyle='--')
ax2.set_title('Rolling 4-week return (Strategy A)')
plt.tight_layout()
plt.show()
"""),
        md("""\
## Drawdown periods: what happened in the worst weeks?

The five worst weeks for Strategy A, with the holdings and the SPY
return that week -- was it a stock-picking failure or a market-wide
drop the strategy could not avoid?
"""),
        code("""\
worst = w_a.nsmallest(5, 'ret').copy()
spy_map = dict(zip(pd.to_datetime(spy['date']), spy['spy_ret'].shift(-1)))
worst['spy_next_ret'] = pd.to_datetime(worst['date']).map(spy_map)
print(worst[['date', 'ret', 'gross_ret', 'turnover',
             'spy_next_ret', 'holdings']].to_string(index=False))
eq = equity(w_a)
dd = (eq - eq.cummax()) / eq.cummax()
print(f"\\nDeepest drawdown: {dd.min():.2%} in week of {dd.idxmin().date()}")
"""),
        md("""\
## HRP vs equal-weight: weight evolution

HRP weights for the same top-5 picks, sampled across the test period,
vs the flat 20% equal weight. The effective number of holdings
(1 / sum(w^2)) shows how concentrated HRP gets.
"""),
        code("""\
from src.hrp import hrp_weights_for_week

weeks = sorted(test['date'].unique())
sample_weeks = weeks[:: max(1, len(weeks) // 6)][:6]
for d in sample_weeks:
    wk = test_rf[test_rf['date'] == d]
    picks = wk.nlargest(5, 'score')['ticker'].tolist()
    w = hrp_weights_for_week(prices, picks, pd.Timestamp(d))
    eff_n = 1.0 / float((w ** 2).sum())
    print(f"{pd.Timestamp(d).date()}: eff. N = {eff_n:.2f} | "
          + ", ".join(f"{t} {v:.2f}" for t, v in w.items()))
"""),
        md("## Turnover and cost decomposition (Strategy A)"),
        code("""\
gross = (1.0 + w_a['gross_ret']).prod() - 1.0
net = m_a['total_return']
print(f"Gross total return (no costs): {gross:+.2%}")
print(f"Net total return:              {net:+.2%}")
print(f"Cost drag:                     {gross - net:+.2%} "
      f"over {len(w_a)} weeks")
print(f"Average weekly turnover:       {w_a['turnover'].mean():.2f}")
print(f"Exact-sim total costs paid:    ${m_exact['total_costs']:,.0f} "
      f"on $100,000 capital")
fig, ax = plt.subplots()
w_a.set_index('date')['turnover'].plot(ax=ax)
ax.set_title('Weekly turnover (Strategy A)')
ax.set_ylabel('one-way turnover')
plt.tight_layout()
plt.show()
"""),
        md("""\
## Long-short leg decomposition

Weekly returns of the long leg (top 5) and the short leg (bottom 5)
separately. A good long-short strategy makes money on both legs; if
only the long leg contributes, the shorts are mostly a hedge (and a
cost).
"""),
        code("""\
legs = []
for date, wk in test_rf.groupby('date', sort=True):
    longs = wk.nlargest(5, 'score')['next_ret'].mean()
    shorts = wk.nsmallest(5, 'score')['next_ret'].mean()
    legs.append({'date': date, 'long_leg': longs, 'short_leg': -shorts})
legs = pd.DataFrame(legs).set_index('date')
legs.index = pd.to_datetime(legs.index)
print(f"Long leg mean weekly return:  {legs['long_leg'].mean():+.4%}")
print(f"Short leg mean weekly return: {legs['short_leg'].mean():+.4%} "
      f"(profit = -return of the worst stocks)")
print(f"Long leg win rate:  {(legs['long_leg'] > 0).mean():.2%}")
print(f"Short leg win rate: {(legs['short_leg'] > 0).mean():.2%}")
fig, ax = plt.subplots()
((1.0 + 0.5 * legs).cumprod()).plot(ax=ax)
ax.set_title('Cumulative contribution by leg (50% weight each, gross of costs)')
ax.set_ylabel('cumulative factor')
ax.legend(['long leg', 'short leg'])
plt.tight_layout()
plt.show()
"""),
    ], "05 — Strategy Deep Dive")


if __name__ == "__main__":
    builders = {
        "01_data_overview": build_01,
        "02_features_eda": build_02,
        "03_target_analysis": build_03,
        "04_model_comparison": build_04,
        "05_strategy_deep_dive": build_05,
    }
    for name, fn in builders.items():
        path = NB_DIR / f"{name}.ipynb"
        nbf.write(fn(), path)
        print(f"Wrote {path}")
