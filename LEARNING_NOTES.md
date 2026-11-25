# Learning Notes — Phase 3

*What this phase teaches, with the project's own numbers as the worked
examples. Written for a CFA-level finance background: the finance is
assumed, the ML-trading-systems reasoning is explained.*

The short version: **Phase 2's headline (+91.6% in 63 weeks) was real
but flattering.** It came from one test window, one training set, one
universe, and a bull market. Phase 3 re-tests the same strategy the way
a practitioner would — retraining quarterly, across five years, in two
universes, with borrow costs — and the honest picture is: a real but
modest ranking edge (AUC ≈ 0.58), strong bull-market returns, and a
2022 drawdown of −46% that the single split never showed. That is a
better project to present, not a worse one: the story is now
*measured*, not *lucky*.

---

## 1. Walk-forward validation: why one train/test split overstates performance

### What it is

The standard pipeline trains once on the first 70% of weeks and tests on
the last 30%. Walk-forward validation replaces that single split with a
sequence of **folds**:

1. Train on weeks 1–104 (2 years), test on weeks 105–117 (one quarter).
2. Train on weeks 1–117, test on weeks 118–130.
3. Keep rolling forward — each fold's training set is everything *before*
   the fold, each fold's test set is the next 13 weeks.

This is also called **rolling-origin evaluation**. Two properties make it
the standard in time-series work:

- **No look-ahead.** Every test week is scored by a model trained only on
  earlier data. (A random train/test shuffle would leak the future into
  training — the single most common backtesting error.)
- **It measures the *process*, not one model.** In live trading you would
  retrain periodically as new data arrives. Walk-forward simulates exactly
  that: retrain → trade one quarter → retrain. The pooled out-of-sample
  return is the return of the *retraining process itself*.

### Expanding vs rolling window

We use an **expanding** window (train on all history before the fold),
not a rolling window (train on, say, only the last 2 years). Rationale:
more training data helps the models, and it matches how a live strategy
would actually be run — you would not throw away old data unless you
believed the market had structurally changed. The trade-off is that an
expanding window is slower to adapt to regime change; a rolling window
adapts faster but trains on less data. If you suspect the market's
structure changed (e.g. post-2022 rate regime), a rolling window is the
right experiment to run next — `walk_forward_folds` documents where the
switch would go.

### What the single split got wrong

| | Single split (63 wks, Sep 2025–Nov 2026) | Walk-forward (20 folds, Nov 2021–Nov 2026) |
|---|---|---|
| RF ROC AUC | 0.604 | **0.580** mean (min 0.497, max 0.653, std 0.038) |
| XGB ROC AUC | 0.593 | **0.586** mean (min 0.506, max 0.650, std 0.039) |
| FF AUC | — | **0.552** mean (min 0.483, max 0.615, std 0.028) |
| Strategy A total return | +91.6% | **+245.0% pooled over 260 weeks** (CAGR +28.1%) |
| Strategy A Sharpe | 2.02 | **0.91** pooled |
| Strategy A max drawdown | −12.0% | **−46.9%** pooled |

Three lessons:

1. **The single-split AUC was above average.** 0.604 vs a 20-fold mean of
   0.580. The test window (late 2025–2026) happened to be a good regime
   for the model. One window cannot tell you whether you are looking at
   skill or a favourable draw; twenty windows can.
2. **The single-split Sharpe was roughly double the honest one** (2.02 vs
   0.91), and the single-split drawdown understated the real tail risk by
   a factor of four (−12% vs −47%). Both are the same error: the test
   window contained no bear market. Section 4 shows where the −47% came
   from.
3. **The edge is real but modest.** A mean AUC of 0.58 with a standard
   deviation of 0.038 across 20 folds is consistently above 0.5 — but
   individual folds fall to 0.497 (fold 11, May–Aug 2024: a coin flip)
   and even 0.483 for the FF model (fold 2: *worse* than random). A
   reviewer who asks "is your model better than chance?" gets the honest
   answer: *on average, yes; in any given quarter, not guaranteed.*

### How to read the per-fold results

- **Mean AUC** is the headline skill measure. **Std across folds** is the
  stability measure: 0.038 means a typical quarter's AUC lands within
  roughly ±0.04 of the mean.
- **Min AUC** is the stress test: the worst quarter was a coin flip.
  Any claim about the model should survive the worst fold, not just the
  average one.
- **Per-fold strategy returns** are far noisier than per-fold AUCs:
  fold 15 (May–Aug 2025) returned +40.9% (Sharpe 4.06); fold 2 (Feb–May
  2022) returned −23.4% (Sharpe −3.17). 12 of 20 folds were positive.
  Strategy returns amplify AUC differences because a top-5 portfolio is
  a concentrated bet on the model's ranking — small ranking changes
  reshuffle the picks (see Section 5).
- **Per-fold CAGR is an annualised extrapolation of 13 weeks** — fold
  15's "CAGR 293%" does not mean the strategy would return 293% in a
  year; it means *if* every quarter were like that quarter, it would.
  Treat short-window CAGRs as a comparability device, not a forecast.
- **Per-fold drawdown resets at each fold boundary**: a fold's drawdown
  is the worst fall *within* that fold. The pooled drawdown (−46.9%) is
  the one that spans folds, and it is the number that matters for
  capital at risk.

### Why LSTM is walk-forward-untested

Walk-forward retrains every model once per fold — 20 fits per model.
An LSTM fit is ~10× slower than a tree fit, so retraining it 20 times
would dominate the runtime for little extra information (its single-split
AUC, 0.595, already sits in the same band as the trees). The walk-forward
run therefore covers RF / XGBoost / FF only. This is a runtime choice,
documented in `src/walkforward.py`, not a methodological one — and it is
the kind of pragmatic trade-off worth stating explicitly in a project
write-up rather than hiding.

---

## 2. Securities borrow fees: the cost of the short leg

### How borrowing works

To short a stock you must first **borrow** it — from a broker's inventory,
a custodian, or another lender — sell it, and later buy it back to return
it. The lender charges a **borrow fee**, quoted as an annual rate on the
short notional (the market value of the borrowed shares), accruing daily
for as long as the position is open. Two things a CFA background makes
intuitive:

- The fee is a **carrying cost**, like the funding rate on a leveraged
  long — except it is charged on top of the position's P&L, not netted
  against it. A short that goes nowhere still loses the fee.
- The fee varies enormously by name. Liquid large caps ("easy to borrow",
  general collateral) cost ~0.3–2% annualised. Hard-to-borrow names —
  small caps, heavily shorted stocks, recent IPOs — can cost 10%, 50%,
  or more. Our universe is S&P 500 large caps, so 2%/yr is a reasonable
  central assumption, and the `--borrow-fee` flag lets you stress-test
  higher rates.

### How it is modelled here

The long-short strategy is dollar-neutral: 50% of capital long, 50%
short. The weekly fee is therefore

```
weekly fee = 0.5 × borrow_fee_annual / 52
```

i.e. at 2%/yr, each week's return is reduced by 0.5 × 0.02 / 52 ≈
0.019% (1.9 bps). Over the 63-week test window the fees sum to
63 × 0.019% ≈ **1.21%** of capital. Small per week; visible over a year.

### With vs without — the actual numbers

| | No borrow fee (F) | 2%/yr borrow fee (F2) | Drag |
|---|---|---|---|
| Name-order universe | +34.0% total, Sharpe 1.74 | +32.4% total, Sharpe 1.67 | **1.61 pp** |
| Market-cap universe | +16.5% total, Sharpe 0.70 | +15.1% total, Sharpe 0.65 | **1.40 pp** |

Two observations:

1. **The drag on total return (1.61 pp) is larger than the sum of the
   fees (1.21 pp).** The extra ~0.4 pp is compounding: fees reduce the
   capital that would otherwise have earned the strategy's return. In
   the market-cap universe the strategy returned less, so the compounding
   effect was smaller (1.40 pp drag). The code reports `borrow_fee_drag`
   (the 1.21 pp sum) *and* the with/without total-return difference, so
   both numbers are visible — the sum is the accounting, the difference
   is the economics.
2. **At 2%/yr the fee does not change the conclusion** — the long-short
   strategy stays profitable in both universes. But it is ~5% of the
   strategy's total return in the name-order run and ~8% in the
   market-cap run. At a 10%/yr fee (a hard-to-borrow universe) the drag
   would be ~5× larger and would erase most of the market-cap long-short
   return. That is why borrow fees matter most exactly where a naive
   backtest looks best: strategies that short small, volatile, heavily
   shorted names are precisely the ones with the highest borrow costs.

---

## 3. Why HRP needs enough holdings: top-10 vs top-5

### The intuition

Hierarchical Risk Parity (HRP) allocates by the **covariance structure**:
it clusters correlated assets and spreads weight across clusters, so the
portfolio is not accidentally concentrated in one correlated group. With
5 holdings, there is not much structure to work with — the clusters are
tiny, the weights cannot differ much from equal weight, and HRP's edge
is mostly noise. With 10 holdings, the covariance matrix has enough
breadth for genuine clusters to form, and HRP can express a genuinely
different portfolio from equal weight.

A second, subtler point: **HRP optimises risk-adjusted return, not raw
return.** It will underweight the highest-volatility names even if the
model likes them. So HRP should be judged on Sharpe ratio and drawdown,
not on total return — comparing HRP to equal weight on total return
alone is grading a risk manager on a return target.

### The numbers (63-week test, same RF scores)

| | Top-5 equal (A) | Top-5 HRP (B) | Top-10 equal (A10) | Top-10 HRP (B10) |
|---|---|---|---|---|
| **Name-order** | | | | |
| Total return | +91.6% | +70.9% | +62.2% | +53.4% |
| Sharpe | 2.02 | 1.75 | 1.82 | **2.11** |
| Max drawdown | −12.0% | −11.8% | −9.6% | **−7.8%** |
| Win rate | 58.7% | 57.1% | 58.7% | **66.7%** |
| **Market-cap** | | | | |
| Total return | +46.1% | +29.6% | +37.4% | **+39.2%** |
| Sharpe | 0.97 | 0.77 | 0.94 | **1.10** |
| Max drawdown | −27.2% | −19.1% | −28.6% | **−26.5%** |

The pattern is consistent across both universes:

- **At 5 holdings, HRP loses on every metric** (return, Sharpe, and
  roughly ties on drawdown). There is nothing for it to diversify.
- **At 10 holdings, HRP wins on Sharpe, drawdown, and win rate in both
  universes** — and in the market-cap universe it wins on total return
  too. The risk-adjusted advantage appears exactly where the theory
  says it should.
- **Top-10 returns less than top-5 in both universes** (A10 +62.2% vs
  A +91.6%; B10 +53.4% vs B +70.9%). Dilution: the 6th–10th ranked
  stocks are weaker signals than the top 5, so adding them lowers
  expected return — but it also lowers volatility and drawdown. That is
  the classic concentration-vs-diversification trade-off, and which
  side you prefer depends on your risk tolerance, not on which number
  is bigger.

### The honest caveat

These are still 63-week numbers from one bull-market window. HRP's
top-10 advantage is *consistent* (two universes, same direction) but
not *proven* — a walk-forward HRP comparison would be the next step if
you wanted to make the claim rigorous. The project reports the
comparison as evidence, not as a conclusion.

---

## 4. Multi-window stability: what it means to win in one window and lose in another

### The per-year picture (Strategy A, walk-forward, out-of-sample)

| Year | Weeks | Strategy A | SPY | A Sharpe | SPY Sharpe | A max DD | SPY max DD |
|---|---|---|---|---|---|---|---|
| 2021 | 7 | +7.4% | −0.3% | 2.96 | −0.03 | −1.3% | −2.0% |
| 2022 | 52 | **−45.8%** | −15.4% | −1.38 | −0.61 | **−46.4%** | −22.3% |
| 2023 | 52 | +45.2% | +22.4% | 1.37 | **1.56** | −22.1% | −9.8% |
| 2024 | 53 | +57.6% | +25.7% | 1.52 | **1.88** | −21.7% | −5.3% |
| 2025 | 52 | +89.6% | +21.0% | 1.98 | 1.20 | −27.2% | −16.9% |
| 2026 | 44 | +36.5% | +9.5% | 1.49 | 0.92 | −14.0% | −8.1% |

### Reading it

**The strategy beats SPY on total return in 5 of 6 windows — and loses
far more in the one window where it loses.** In 2022 it fell −45.8%
against SPY's −15.4%: it did not just participate in the bear market,
it amplified it threefold. That is the single most important fact in
this phase, and the single-split backtest could never have shown it.

Why does it amplify a bear market? The model ranks on momentum,
volatility, and recent returns. The stocks it picks are high-beta
momentum names — the same names that fall hardest when the market turns.
A concentrated top-5 portfolio of high-beta stocks is, in effect, a
leveraged long position in the market's riskiest corner. In a bull
market that leverage pays (2025: +89.6% vs +21.0%); in a bear market it
charges (2022: −45.8% vs −15.4%). The strategy's beta is not 1.0 — it
behaves like a leveraged market exposure, and its returns should be
judged with that in mind.

**Total return and Sharpe tell different stories.** In 2023 and 2024 the
strategy beat SPY on return (+45.2% vs +22.4%; +57.6% vs +25.7%) but
*lost* on Sharpe (1.37 vs 1.56; 1.52 vs 1.88). It earned more by taking
much more risk — its drawdowns in those years (−22.1%, −21.7%) were
roughly 2–4× SPY's (−9.8%, −5.3%). A Sharpe comparison is the honest
one for a strategy that takes this much risk, and on that measure the
strategy wins 3 windows (2021, 2025, 2026), loses 2 (2023, 2024), and
loses badly in 1 (2022).

**What "wins in one window, loses in another" means.** It does *not*
mean the strategy is broken, and it does not mean it is a fraud. It
means the edge is **regime-dependent**: the model's signal (momentum +
volatility + factor exposures) works when trends persist and fails when
trends reverse. That is a property of the signal, not a bug — momentum
is well documented to crash in reversals (the "momentum crash"
literature). The practical implications:

1. **Position sizing matters more than signal quality.** A strategy with
   a −47% drawdown cannot be run at full capital without a drawdown
   control (volatility targeting, a max-drawdown stop, or a market-regime
   filter). The backtest measures the *signal*; a real deployment would
   need a *risk overlay*.
2. **Report the bad window first.** In a presentation, leading with 2022
   ("here is where it loses, and why") is more credible than leading
   with 2025. Reviewers trust a backtest that shows its own failure
   modes.
3. **The pooled numbers are the headline, not the best window.**
   Pooled: +245% total, CAGR +28.1%, Sharpe 0.91, max DD −46.9%, win
   rate 57.3% over 260 out-of-sample weeks. That is a good result with
   an honest risk statement — and it is the number to quote, not the
   +91.6% from the single split.

---

## 5. A bonus lesson: same AUC, very different returns

Phase 3 produced the same pattern three times, and it is worth naming
explicitly: **small changes in the setup barely move the AUC but
dramatically move the strategy return.**

| Change | RF AUC | Strategy A total return |
|---|---|---|
| Proxy macro → FRED macro (same universe, same weeks) | 0.604 → 0.609 | +91.6% → **+48.3%** |
| Name-order → market-cap universe (same weeks) | 0.604 → 0.597 | +91.6% → **+46.1%** |
| Long-short, proxy → FRED macro | — | +34.0% → **+6.4%** |

The AUC measures *average ranking quality* across all 50 stocks; the
strategy return depends on *which 5 stocks land at the very top* each
week. A small score change reshuffles the top 5, and with only 5
holdings, one different pick can swing a week's return by several
percent. Over 63 weeks those swings compound into very different totals.
Three consequences:

1. **Top-N strategy returns are high-variance statistics.** Treat any
   single 63-week total return as a noisy draw, not a precise estimate.
   The walk-forward pooled return (260 weeks) is the lower-variance
   number.
2. **Universe choice is a research decision with big consequences.**
   The alphabetical name-order universe happens to be sector-diversified;
   the market-cap universe is concentrated in mega-cap tech, whose names
   are more correlated with each other (less diversification in the
   top-5) and more efficiently priced (less dispersion to exploit).
   Neither universe is "right" — but the choice must be disclosed, and
   results should ideally be shown in both.
3. **Feature changes are not free.** Swapping the macro source from
   yfinance proxies to FRED changed the feature set (4 proxy features →
   8 FRED features), which changed the rankings, which halved the
   strategy return — while *improving* the AUC slightly. Better data
   does not automatically mean better backtest returns; it means a more
   honest model. Do not judge a data upgrade by the backtest total.

---

## Talking points (for presenting the project)

1. "The model has a real but modest edge: mean walk-forward AUC 0.58
   across 20 quarterly folds, versus 0.50 for chance."
2. "The pooled out-of-sample return is +245% over five years, CAGR
   28%, Sharpe 0.91 — but the max drawdown is −47%, driven by 2022,
   when the strategy lost three times the market."
3. "The strategy is a leveraged momentum exposure: it beats the market
   in bull years and amplifies it in bear years. Any live version would
   need a risk overlay."
4. "HRP beats equal weight on Sharpe at 10 holdings but not at 5 —
   diversification needs enough names to work with."
5. "Borrow fees cost ~1.6 pp over 63 weeks at 2%/yr — material but not
   decisive for large caps; they would dominate for hard-to-borrow
   names."
6. "Strategy-level returns are sensitive to universe and feature
   choices even when AUC is stable — so I report results across
   universes and windows rather than quoting one backtest."
