# Algo-Trading Strategy: Weekly Top-Quintile Stock Selection

Course project for the **DataTalksClub Stock Markets Analytics Zoomcamp 2026**
— an end-to-end machine learning trading system: data → features → models →
strategy simulation → automation.

## Problem statement

Each week, select the 5 S&P 500 large-cap stocks most likely to be among the
best performers next week, hold them for one week, and rebalance. The system
must be fully automated: given the latest market data it downloads prices,
rebuilds features, retrains (or reloads) models, scores every stock, and
outputs the new trades — plus a backtest of how the strategy would have
performed.

**Why this framing?** Predicting the *direction* of a single stock (up/down)
treats +0.1% and +5% as the same outcome. Ranking stocks cross-sectionally
and buying the top quintile matches the actual trading decision ("buy the
best N stocks") and is the standard setup for cross-sectional equity
strategies in industry.

## Design

| Decision | Choice | Why |
|---|---|---|
| Universe | S&P 500 large caps (top 30 in the test run, 50–100 full run) | Liquid, clean yfinance data, realistic to trade |
| Horizon | Weekly, rebalance Monday | Low noise, low turnover and costs vs daily |
| Target | Top-quintile next-week return (binary) | Directly matches the buy decision |
| Features | Momentum (1/4/12w), realised vol (4/12w), RSI, MACD, distance from 52w high/low, volume trend, VIX, 10Y yield, Fama-French factors, month dummies | Each group encodes a testable hypothesis (trend continuation, mean reversion, volatility, macro regime) |
| Models | RandomForest baseline, Fama-French-style factor regression (Ridge), XGBoost, LSTM (Phase 2) | RF is the 2026 course best config; the factor regression is the interpretable industry-style baseline; XGBoost is the gradient-boosting comparison; the LSTM is the deep-learning comparison on panel sequences |
| Allocation | Equal weight vs Hierarchical Risk Parity (HRP) | HRP (López de Prado 2016) needs no expected returns and no covariance inversion, so weights are stable out-of-sample |
| Strategies | Long-only (equal / HRP) + long-short (Phase 2) | Long-short (long top 5 / short bottom 5, dollar-neutral) tests whether the model identifies losers as well as winners, and hedges market risk |
| Risk overlay | 200-day MA regime filter (Phase 4, Faber 2007) | Invested when SPY is above its 200-day MA, cash when below — gates *whether* the long-only strategy is invested, not which stocks it picks; cuts the 2022 drawdown by more than half |
| Simulation | Vector backtest + exact row-by-row simulation (Phase 2) | The exact engine tracks capital/positions/cash in dollars and reinvests gains; it must reconcile with the vector backtest (gap < 0.05 pp) |
| Benchmark | SPY buy-and-hold | The realistic alternative for the same capital |
| Costs | 5 bps one-way on traded notional + 2%/yr borrow fee on the short leg (Phase 3) | Keeps the backtest honest; the fee is reported side by side with the no-fee run so its drag is visible |
| Automation | `main.py` + Docker + weekly cron (Phase 2) | One command reproduces the whole pipeline; the cron job refreshes data and outputs the new week's trades |

## Repository layout

```
├── README.md
├── LEARNING_NOTES.md        # Phase 3-4 learning write-up (walk-forward, borrow fees, HRP, stability, trend following)
├── app.py                   # Streamlit observation dashboard (tabs: regime, picks, ranking, strategies, walk-forward, about)
├── requirements-app.txt     # slim dashboard requirements (Streamlit + pipeline deps, no TF/XGBoost)
├── dashboard_data/
│   └── walkforward_results.json  # precomputed walk-forward results for the dashboard (see scripts/precompute_walkforward.py)
├── requirements.txt
├── Dockerfile               # containerised weekly run (build-untested: no Docker in the dev VM)
├── .dockerignore
├── main.py                  # end-to-end: download -> features -> train -> predict -> simulate
├── src/
│   ├── data.py              # universe (name or market-cap order), yfinance prices/macro, FRED macro (Phase 3), Fama-French factors, SPY
│   ├── features.py          # point-in-time-correct features; TO_PREDICT / NUMERIC / DUMMIES; FRED or proxy macro features
│   ├── target.py            # top-quintile target + time-based split (never shuffled)
│   ├── models.py            # RandomForest, FF factor regression, XGBoost
│   ├── lstm_model.py        # LSTM comparison model (Phase 2; needs tensorflow)
│   ├── hrp.py               # Hierarchical Risk Parity (from scratch, scipy)
│   ├── walkforward.py       # Phase 3 walk-forward validation (expanding window, per-fold AUC + strategy metrics); Phase 4 filtered comparison
│   ├── regime.py            # Phase 4 200-day MA regime filter (Faber 2007): daily SPY, weekly exposure, current state
│   └── simulate.py          # vector backtest, long-short (+ borrow fee, Phase 3), exact simulation, benchmark, grouped metrics
├── scripts/
│   ├── build_notebooks.py   # generates the notebooks in notebooks/
│   ├── precompute_walkforward.py  # regenerates dashboard_data/walkforward_results.json
│   ├── weekly_run.sh        # weekly cron entry point (refresh + retrain + predict)
│   └── crontab.example      # documented crontab line (no live job installed)
├── notebooks/
│   ├── EDA_outline.md       # EDA plan
│   ├── 01_data_overview.ipynb
│   ├── 02_features_eda.ipynb
│   ├── 03_target_analysis.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_strategy_deep_dive.ipynb
└── data/                    # cached downloads (gitignored -- never committed)
```

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                        # test run: 30 tickers, 3 years
python main.py --tickers 50 --years 5 # fuller run
python main.py --universe market-cap  # largest stocks first (not name order)
python main.py --tickers 50 --years 8 --walk-forward  # Phase 3 walk-forward validation
python main.py --borrow-fee 0.05      # stress-test the short leg at 5%/yr borrow fee
python main.py --tickers 50 --years 8 --walk-forward --regime-filter  # Phase 4: 200-day MA
                                                             # regime filter on Strategy A
                                                             # (filtered vs unfiltered + 1% band)
```

Data is downloaded from yfinance and the Ken French data library on first
run and cached under `data/` (parquet, 7-day freshness). Re-runs reuse the
cache. No data files are committed to GitHub (course rule).

**FRED macro data (Phase 3).** The pipeline uses FRED as its primary macro
source when the `FRED_API_KEY` environment variable is set, and falls back
to the yfinance ^VIX/^TNX proxies otherwise (with a printed notice). The
key is free from the St. Louis Fed (fred.stlouisfed.org, "My Account" →
"API Keys") and is read from the environment only — never hardcoded, never
written to any file:

```bash
export FRED_API_KEY=...   # your free key; keep it out of files and Git
python main.py --tickers 50 --years 5
```

Series used: **DGS10** (10Y Treasury yield), **VIXCLS** (VIX), **T10Y2Y**
(10Y–2Y spread), **DGS2** (2Y Treasury yield) — weekly end-of-period
values, each as a level and a 1-week change (8 features, replacing the 4
yfinance proxy features). If the FRED download fails for any reason, the
pipeline falls back to the proxies with a warning instead of crashing.
Note the macro source changes the feature set, which changes the rankings:
with FRED enabled the same 5-year run gives RF AUC 0.609 (vs 0.604 with
proxies) but Strategy A +48.3% (vs +91.6%) — see LEARNING_NOTES.md §5 for
why small feature changes move strategy returns far more than AUC.

**Install note (small `/tmp`).** `tensorflow-cpu` is a ~274 MB wheel and
pip stages downloads in the system temp dir. On systems where `/tmp` is a
small tmpfs (the dev VM has 512 MB), the install dies at ~268 MB with
`No space left on device`. Fix: point pip at a big disk first:

```bash
mkdir -p ~/workspace/.tmp
TMPDIR=~/workspace/.tmp pip install -r requirements.txt
```

TensorFlow is optional: without it the pipeline prints a skip message for
the LSTM and runs everything else. The same applies to XGBoost.

**Notebooks.** Regenerate with `python scripts/build_notebooks.py` and
execute with `jupyter nbconvert --execute --to notebook --inplace
--ExecutePreprocessor.timeout=900 notebooks/*.ipynb`.

**Docker + cron (Phase 2).** The `Dockerfile` (python:3.12-slim) installs
from `requirements.txt` and runs `main.py`:

```bash
docker build -t stock-zoomcamp .
docker run --rm -v stock-data:/app/data stock-zoomcamp \
    --tickers 50 --years 5 --universe market-cap --force-refresh
```

The image is ~2 GB (tensorflow-cpu + xgboost dominate it). **The image
has not been build-tested** -- the dev VM has no Docker; the first build
should be run on a machine with Docker before relying on it. Without
Docker, the same weekly run works from cron: `scripts/weekly_run.sh`
refreshes data, retrains, and logs to `logs/weekly_YYYY-MM-DD.log`;
`scripts/crontab.example` documents the Monday 08:00 ET crontab line.
No live cron job is installed by this project -- install it deliberately.

## Dashboard

An observation interface for the whole system — no commands, just open
it in a browser:

- **Regime**: the 200-day MA filter as a big IN/OUT badge, SPY vs its
  200-day MA (values, as-of date, 2-year chart).
- **This week's picks**: top-5 long / bottom-5 short tables (ticker,
  close, RF score) with a universe toggle (market-cap / name order),
  plus a ticker search (rank, score, close for any ticker in the
  universe — e.g. NVDA).
- **Full ranking**: all 50 stocks, sortable, with score bars.
- **Strategies**: standard-run backtest metrics + equity curves.
- **Walk-forward**: precomputed validation results (pooled metrics,
  per-year table, equity curve) — see
  `scripts/precompute_walkforward.py` for how the numbers are
  generated.
- **About**: how the model picks stocks, what the score means
  (baseline 0.20, not 0.50), the regime filter, and the honest caveats.

Run it locally:

```bash
pip install -r requirements-app.txt   # slim: Streamlit + pipeline deps only
streamlit run app.py                  # http://localhost:8501
```

The app uses only the RandomForest for live picks (fast); LSTM/XGBoost
are skipped on purpose. Data is cached for 12 hours in the app; the
sidebar "Refresh data" button clears the cache and re-downloads from
yfinance. On a cold start (e.g. Render free tier after spin-down) the
app re-downloads everything — under a minute — because the free-tier
filesystem does not persist the `data/` cache. Macro source is FRED
when the `FRED_API_KEY` environment variable is set, otherwise the
yfinance proxies (set it in the Render dashboard, never in a file).

## Results (Phase 4)

Phase 4 adds a market-timing overlay to Strategy A: the 200-day
moving-average regime filter (Faber 2007). Each week the portfolio is
invested only if SPY closed above its 200-day MA on the prior Friday;
otherwise it holds cash at 0% (no T-bill yield modelled). The filter
gates *whether* the strategy is invested, never *which* stocks it
picks, and the decision uses only closes available at Friday's close
(no lookahead — verified by an independent manual recomputation of the
MA at seven switch dates, see `src/regime.py`). Switching costs are
charged: liquidating and re-entering each count as one-way turnover of
1.0. Full explanation with the numbers as worked examples:
[LEARNING_NOTES.md](LEARNING_NOTES.md) §6.

### Walk-forward: filtered vs unfiltered Strategy A (same 20 folds)

Run 2026-11-25: same 20-fold walk-forward as Phase 3 (50 tickers, 8
years, 260 out-of-sample weeks, Nov 2021 – Nov 2026), with Strategy A
simulated twice per fold — once unfiltered, once with the filter. The
unfiltered leg reproduces Phase 3 (pooled +244.4% vs +245.0%, Sharpe
0.91, max DD −46.3% vs −46.9% — the small drift is the documented
fresh-download wobble, not a code change).

| Pooled (260 weeks) | Unfiltered | + 200d MA filter | + filter, 1% band |
|---|---|---|---|
| Total return | +244.4% | **+250.8%** | +233.7% |
| CAGR | +28.1% | **+28.5%** | +27.3% |
| Sharpe | 0.91 | **1.09** | 1.05 |
| Max drawdown | −46.3% | **−28.5%** | −29.5% |
| Win rate (pooled) | 56.2% | 45.0% | — |
| Win rate (invested weeks only) | 56.2% | **57.9%** | — |
| Avg turnover | 0.464 | 0.395 | — |
| Weeks invested | 100% | 77.7% (202/260) | 77.7% |
| Exposure switches | — | 16 | 14 |

The filter's value is the drawdown cut (−46.3% → −28.5%, an 18 pp
improvement) and the Sharpe gain (0.91 → 1.09), not the return — the
+6.4 pp total-return gain over five years is small and within noise.
The pooled win rate *falls* for the filtered variant (45.0%) only
because cash weeks (0% return) count as non-wins; on invested weeks
alone the win rate is 57.9%, slightly above the unfiltered 56.2%.

Filtered Strategy A by calendar year (vs unfiltered):

| Year | Weeks | Invested | Filtered total | Unfiltered total | Filtered max DD | Unfiltered max DD |
|---|---|---|---|---|---|---|
| 2021 | 7 | 100% | +4.0% | +4.0% | −1.3% | −1.3% |
| 2022 | 52 | 17% | **−20.2%** | −43.5% | **−19.5%** | −45.2% |
| 2023 | 52 | 94% | +16.8% | +39.4% | −23.1% | −21.0% |
| 2024 | 53 | 100% | +49.6% | +49.6% | −22.0% | −22.0% |
| 2025 | 52 | 83% | +83.3% | +100.9% | −14.3% | −24.7% |
| 2026 | 44 | 93% | +31.9% | +39.8% | −12.5% | −12.5% |

2022 is where the filter earns its keep: out for 43 of 52 weeks, it
more than halves the year's loss (−20.2% vs −43.5%) and drawdown
(−19.5% vs −45.2%). The cost shows up in 2023 and 2025: the filter was
out during rebounds (Oct 2023: out one week, missed the +5.8% SPY
rebound week; Mar–May 2025: out nine weeks, missed the April rebound),
so filtered 2023 returns less than half the unfiltered year. That
lag/whipsaw cost is the price of the drawdown protection, and it is
real — see LEARNING_NOTES.md §6.

**The 1% hysteresis band did not help.** It cut switches from 16 to 14
but returned less (+233.7% vs +250.8%), with a slightly lower Sharpe
(1.05 vs 1.09) and slightly worse drawdown (−29.5% vs −28.5%). The
band's delayed entries and exits cost more than the two avoided
whipsaws saved. Reported as a negative result, not hidden.

**Current state (as of 2026-11-23): OUT (cash).** SPY $722.29 vs
200-day MA $730.47 (−1.1%). The filter flipped out in the week of Nov
16 (Friday Nov 20 close $689.81 vs MA $730.29) and has stayed out since.
Run `python main.py --regime-filter` for the live state.

## Results (Phase 3)

Phase 3 re-tests the Phase 2 strategy the way a practitioner would:
quarterly retraining across five years, two universes, and borrow costs
on the short leg. Full explanation with the numbers as worked examples:
[LEARNING_NOTES.md](LEARNING_NOTES.md).

### Walk-forward validation (the headline result)

Run 2026-11-25: 50 tickers (name order), 8 years of weekly data
(418 weeks of prices; 364 weeks in the panel after the 52-week
lookback). Expanding window, 104-week initial training set, 13-week
folds → **20 folds, 260 out-of-sample weeks, Nov 2021 – Nov 2026**.
RF / XGBoost / FF retrained per fold; LSTM is walk-forward-untested
(runtime — see LEARNING_NOTES.md §1).

Mean ROC AUC across folds: RandomForest **0.580** (min 0.497, max 0.653,
std 0.038); XGBoost **0.586** (0.506–0.650); FF regression **0.552**
(0.483–0.615). The single-split AUC of 0.604 was above the 20-fold
average — one window cannot tell skill from a favourable draw.

Strategy A (RF top-5 equal-weight), pooled out-of-sample:
**+245.0% total over 260 weeks, CAGR +28.1%, Sharpe 0.91, max drawdown
−46.9%, win rate 57.3%**. 12 of 20 folds were positive; best fold
+40.9% (May–Aug 2025), worst −23.4% (Feb–May 2022).

Strategy A by calendar year vs SPY (same weeks):

| Year | Weeks | Strategy A | SPY | A Sharpe | SPY Sharpe | A max DD |
|---|---|---|---|---|---|---|
| 2021 | 7 | +7.4% | −0.3% | 2.96 | −0.03 | −1.3% |
| 2022 | 52 | **−45.8%** | −15.4% | −1.38 | −0.61 | **−46.4%** |
| 2023 | 52 | +45.2% | +22.4% | 1.37 | 1.56 | −22.1% |
| 2024 | 53 | +57.6% | +25.7% | 1.52 | 1.88 | −21.7% |
| 2025 | 52 | +89.6% | +21.0% | 1.98 | 1.20 | −27.2% |
| 2026 | 44 | +36.5% | +9.5% | 1.49 | 0.92 | −14.0% |

The strategy beats SPY on total return in 5 of 6 windows but loses
three times the market in the one bad year (2022) — it behaves like a
leveraged momentum exposure. It also *loses* on Sharpe in 2023 and 2024
despite winning on return: it earns more by taking much more risk. The
pooled numbers above (Sharpe 0.91, DD −46.9%) are the honest headline,
not the single-split +91.6% / Sharpe 2.02 / DD −12.0%.

### Top-10 HRP test (63-week test, same RF scores)

| | Top-5 equal (A) | Top-5 HRP (B) | Top-10 equal (A10) | Top-10 HRP (B10) |
|---|---|---|---|---|
| Name-order total | +91.6% | +70.9% | +62.2% | +53.4% |
| Name-order Sharpe | 2.02 | 1.75 | 1.82 | **2.11** |
| Name-order max DD | −12.0% | −11.8% | −9.6% | **−7.8%** |
| Market-cap total | +46.1% | +29.6% | +37.4% | **+39.2%** |
| Market-cap Sharpe | 0.97 | 0.77 | 0.94 | **1.10** |

At 5 holdings HRP loses on every metric; at 10 it wins on Sharpe,
drawdown, and win rate in both universes (and on total return in the
market-cap universe). HRP's diversification needs enough holdings to
work with — and it should be judged on risk-adjusted return, not raw
return. Top-10 returns less than top-5 in both universes (dilution:
the 6th–10th picks are weaker signals), at lower volatility.

### Borrow-fee drag on the long-short strategy (63 weeks)

| | No fee (F) | 2%/yr fee (F2) | Drag |
|---|---|---|---|
| Name-order | +34.0%, Sharpe 1.74 | +32.4%, Sharpe 1.67 | **1.61 pp** |
| Market-cap | +16.5%, Sharpe 0.70 | +15.1%, Sharpe 0.65 | **1.40 pp** |

The fee sum over 63 weeks is 1.21 pp (0.5 × 2% × 63/52); the
total-return drag is larger because fees reduce the capital that
compounds. At 2%/yr the fee is material but not decisive for large
caps; at hard-to-borrow rates it would dominate. Verified by a
hand-checked 2-week example (see `src/simulate.py`).

### Market-cap vs name-order universe (63-week test, same dates)

Same test weeks, same models — only the 50 tickers differ. RF AUC
0.597 vs 0.604 (barely moved), but Strategy A +46.1% vs +91.6%, Sharpe
0.97 vs 2.02, max drawdown −27.2% vs −12.0%. The market-cap universe is
concentrated in mega-cap tech: its names are more correlated (less
diversification in the top-5) and more efficiently priced (less
dispersion to exploit). All strategies still beat SPY (+10.8%) in both
universes, but by much less in market-cap. Universe choice is a
research decision with big consequences and must be disclosed.

## Results (Phase 2 fuller run)

Run 2026-11-25: 50 tickers (first 50 of the Wikipedia S&P 500 list,
name order), 5 years of weekly data. Features: 10,550 rows / 211 weeks
after the 52-week lookback; target positive rate 0.200. Train
2022-11-14 → 2025-09-01 (7,350 rows); test 2025-09-08 → 2026-11-16
(63 weeks, 3,150 rows).

Model quality (test): RandomForest ROC AUC **0.604**; XGBoost 0.593;
LSTM **0.595** (1 LSTM layer, 48 units, 12-week windows; early stopping
at 7 epochs on a validation AUC of 0.502 -- the validation window was a
hard regime, yet the test AUC landed with the other models). The three
models sit in a tight 0.59–0.60 band: on weekly returns, model choice
matters less than features and validation discipline.

| Strategy (test, 63 weeks) | Total | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| A: RF top-5 equal-weight | +91.6% | +71.0% | 2.02 | −12.0% | 0.48 |
| B: RF top-5 HRP-weight | +70.9% | +55.6% | 1.75 | −11.8% | 0.66 |
| F: RF long-short top-5/bottom-5 | +34.0% | +27.3% | 1.74 | −7.0% | 0.55 |
| C: FF-regression top-5 equal-weight | +32.8% | +26.4% | 0.86 | −31.6% | 0.18 |
| D: XGBoost top-5 equal-weight | +40.0% | +32.0% | 1.14 | −23.6% | 0.44 |
| E: LSTM top-5 equal-weight | +47.4% | +37.8% | 1.19 | −17.9% | 0.26 |
| Benchmark: SPY buy-and-hold | +10.8% | +8.9% | 0.64 | −11.4% | 0.00 |

Exact simulation check (Strategy A): vector +91.58% vs exact +91.55%,
gap −0.040 pp -- the expected cost-timing cross-term (costs are charged
before the week's return accrues in the exact engine). Final capital
$191,545 from $100,000; total costs paid $2,168. Latest trades (week of
2026-11-23, RF retrained on all data): AMAT, APP, ANET, AMD, ADI.

**Reading these numbers honestly.** Strategy A's +91.6% over 63 weeks
with a 2.02 Sharpe is a striking headline and should be treated with
suspicion, not celebrated: 63 weeks is one short window, the model was
trained once (no walk-forward retraining), the test window was a strong
market, and survivorship bias flatters the backtest (see below). What
the run *does* support: the ranking signal is real but modest (AUC
~0.60), the long-short leg works (Sharpe 1.74 with the lowest drawdown,
−7.0%), and the LSTM is the most interesting secondary finding:
equal-weight returns close to XGBoost's at the *lowest* turnover (0.26)
-- its picks are stickier, which is exactly what a sequence model should
produce. Phase 3 re-tests all of this — walk-forward validation puts the
honest headline at Sharpe 0.91 with a −46.9% max drawdown over 260
out-of-sample weeks, and the top-10 HRP test (done in Phase 3) confirms
HRP's edge appears at 10 holdings, not 5. See "Results (Phase 3)" above
and LEARNING_NOTES.md.

Phase 1 test run (30 tickers, 3 years, 32 test weeks) for reference:
RF AUC 0.556, XGBoost 0.508; Strategy A +17.3% vs SPY +1.7%.

## Assumptions and limitations (honest discussion)

- **Survivorship bias.** The universe is the *current* S&P 500 list. Stocks
  that were delisted or dropped from the index are missing, so the backtest
  is biased optimistic. A point-in-time constituent list would fix this but
  is not freely available; the bias is acknowledged, not hidden.
- **No lookahead.** Every feature uses only data available at Friday close;
  the target is next week's return; the train/test split is by time
  (first ~70% of weeks train, last ~30% test) and is never shuffled.
  Scaling in the Ridge pipeline is fit on train only.
- **Costs are approximate.** 5 bps one-way covers commission + slippage for
  liquid large caps, but real slippage varies with size and volatility.
- **Weekly bars are Monday-dated** in yfinance but contain the full week's
  data (Friday close). Fama-French weekly factors are Friday-dated and are
  merged onto the same week's bar — point-in-time correct relative to the
  bar's information content.
- **Macro source.** FRED (DGS10, VIXCLS, T10Y2Y, DGS2) is the primary
  macro source when `FRED_API_KEY` is set; otherwise yfinance ^VIX and
  ^TNX proxies are used. Changing the macro source changes the feature
  set and therefore the rankings — see LEARNING_NOTES.md §5.
- **Borrow costs are modelled (Phase 3).** The long-short strategy
  charges a configurable borrow fee on the short leg (default 2%
  annualised on short notional, pro-rata weekly), reported side by side
  with the no-fee run. The 2% rate is a large-cap assumption; a
  hard-to-borrow universe would cost much more.
- **Reproducibility across re-downloads.** Results are exactly
  reproducible from the pinned `data/` cache (7-day freshness). A fresh
  download from yfinance can shift tree-model strategy returns by a few
  percentage points: auto-adjusted closes come back with ~1e-7 relative
  float noise between downloads (verified: 6,057 of 13,100 rows differed
  in the 6th decimal), which flips a few top-5 rankings at the cutoff.
  Across three fresh downloads, Strategy A read +84.3% / +80.2% /
  +91.6%; the linear factor regression (+32.8%) and the LSTM (+47.4%)
  were identical every time -- smooth models are robust to the noise,
  threshold-based tree ensembles are not. Treat the headline return as
  a ~±5 pp band, not a precise figure. (Phase 3 adds a stronger point:
  changing the *universe* or the *macro source* moves strategy returns
  far more than re-download noise — see LEARNING_NOTES.md §5.)
- **Walk-forward validation is done (Phase 3) but bounded.** The model
  is retrained quarterly over 20 folds (Nov 2021 – Nov 2026), but the
  history is still only 8 years, the LSTM is walk-forward-untested
  (runtime), and the expanding window adapts slowly to regime change.
  A rolling-window variant and a walk-forward HRP comparison are open
  follow-ups.
- **Short backtest.** An 8-year history leaves 5 years of
  out-of-sample weeks; results are indicative, not proof of an edge.
- **Regime filter (Phase 4).** Cash weeks earn 0% — no T-bill yield is
  modelled, so the filtered return is slightly understated (by roughly
  the T-bill rate × the 22% of weeks spent in cash). The filter's
  benefit is measured on one 5-year out-of-sample window that contains
  a single bear market (2022); the drawdown cut is the robust result,
  the small total-return gain (+6.4 pp over five years) is within
  noise. The pooled win rate is not comparable between filtered and
  unfiltered runs (cash weeks count as non-wins) — use the
  invested-weeks win rate (57.9% vs 56.2%) instead. The current
  filter state is a weekly signal, not a recommendation: it says
  whether the trend rule is in or out, not where the market goes next.

## TODO / Phase 3

- [x] LSTM comparison model (Phase 2 -- done, AUC 0.595).
- [x] Long-short strategy (Phase 2 -- done, Sharpe 1.74, lowest drawdown).
- [x] Exact row-by-row simulation with reinvestment (Phase 2 -- done,
      reconciles with the vector backtest to −0.04 pp).
- [x] Docker + cron automation (Phase 2 -- files done; Docker image
      build-untested, no Docker in the dev VM).
- [x] EDA + modelling notebooks in `notebooks/` (Phase 2 -- all 5
      executed).
- [x] Market-cap-ordered universe (Phase 2 -- `--universe market-cap`).
- [x] Walk-forward retraining + multi-window validation (Phase 3 --
      done: 20 folds, pooled Strategy A +245% / Sharpe 0.91 / DD
      −46.9% over 260 out-of-sample weeks).
- [x] Top-10 selection as the fairer HRP test (Phase 3 -- done: B10
      beats A10 on Sharpe in both universes).
- [x] Borrow-fee drag on the short leg (Phase 3 -- done: 2%/yr default,
      1.61 pp drag over 63 weeks, hand-verified).
- [x] FRED integration (Phase 3 -- done: DGS10, VIXCLS, T10Y2Y, DGS2
      via `FRED_API_KEY`; yfinance proxies remain the fallback).
- [x] XGBoost: if it fails to install in an environment, the pipeline
      skips it cleanly and reports the skip (same for TensorFlow/LSTM).
- [x] 200-day MA regime filter (Phase 4 -- done: Faber 2007 overlay on
      Strategy A; walk-forward pooled Sharpe 0.91 -> 1.09, max DD
      -46.3% -> -28.5%, 2022 loss halved; 1% band variant tested and
      reported as a negative result; no-lookahead verified by manual
      recomputation; current state OUT as of 2026-11-23).

Open follow-ups (not started): Docker image build test (needs a machine
with Docker); rolling-window walk-forward variant; walk-forward HRP
comparison; LSTM walk-forward run (runtime); volatility targeting as a
second risk overlay (complements the regime filter, which is binary).
