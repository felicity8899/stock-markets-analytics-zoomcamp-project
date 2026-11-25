# Learning Notes — Phases 3–5

*What these phases teach, with the project's own numbers as the worked
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

**Phase 4** answers the question Phase 3 raised — *what do you do about
the −46% drawdown?* — with the industry's simplest risk overlay: a
200-day moving-average regime filter (Faber 2007). It cuts the pooled
drawdown to −28.5% and lifts the Sharpe from 0.91 to 1.09, at the price
of lag and whipsaws that cost real return in 2023 and 2025. Section 6
walks through the trade-off with the actual numbers.

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

## 6. Trend following as a risk overlay: the 200-day MA regime filter

### The idea (Faber 2007)

Phase 3 ended with a problem: Strategy A is a leveraged momentum
exposure with a −46% pooled drawdown, driven by 2022, when it lost
three times the market. Section 4's first practical implication was
that a strategy like this cannot be run at full capital without a
drawdown control. Phase 4 implements the simplest one in the
literature — the **200-day moving-average rule** from Faber (2007),
"A Quantitative Approach to Tactical Asset Allocation":

> Be invested when the market is above its 200-day moving average;
> hold cash when it is below.

The rule is trend following applied to the *market* rather than to
individual stocks. It does not predict anything: it reacts. When the
market has been falling long enough for the price to drop below its
own 200-day average, the rule assumes the trend has turned and steps
aside; when the price recovers above the average, it steps back in.
The 200-day window is the industry-standard long-term trend length —
long enough to ignore weekly noise, short enough to catch a regime
change within a few months.

Two things make it the right first overlay for this project:

1. **It attacks the specific failure mode.** The strategy's losses are
   concentrated in bear markets (2022: −43.5%). A filter that is out
   during bear markets removes exactly those losses, while leaving the
   bull-market returns (where the strategy earns its edge) mostly
   untouched.
2. **It is a *separate* decision from stock selection.** The filter
   gates *whether* the portfolio is invested; the model still decides
   *which* 5 stocks to hold. The two layers can be evaluated
   independently — and the walk-forward comparison below does exactly
   that, running the same folds with and without the filter.

### How it is implemented (and why the timing is not lookahead)

The filter is computed in `src/regime.py` from **daily** SPY closes
(the 200-day MA needs daily data; weekly bars are too coarse). The
timing convention is the part that matters most, and it is easy to get
wrong:

- The strategy rebalances at each week's close. The decision for week
  `t` is made at **Friday's close of week `t`** — using the SPY close
  of the last trading day of that week and the 200-day MA computed
  from the 200 closes ending on that same day.
- The exposure for week `t` then gates the return earned over week
  `t+1` — the week the strategy would have been invested in. Both
  inputs are known at the rebalance moment, so nothing from the future
  enters the decision.

Two details worth noting:

- **Holiday Fridays** are handled by aligning each week's Friday to
  the last *trading* day on or before it (`merge_asof`, backward). If
  Friday is a holiday, Thursday's close decides — which is what a live
  trader would have used.
- **Verification.** The MA was independently recomputed by hand (the
  mean of the 200 raw daily closes ending on the decision day) for
  seven weeks, including both 2022 switch dates and the Nov 2026
  switch — the manual values match the module exactly (difference
  0.00). This is the kind of check that is cheap to do and expensive
  to skip: a one-day alignment error in a filter is a lookahead leak
  that flatters every result downstream.

Cash weeks earn **0%** (no T-bill yield is modelled — a deliberate
simplification, disclosed in the README; it slightly *understates*
the filtered return). Switching costs are charged: liquidating into
cash and re-entering each count as one-way turnover of 1.0, so the
16 switches over 260 weeks cost 16 × 5 bps ≈ 0.8 pp in total — small,
but not zero, and visible in the numbers rather than assumed away.

### The results: what the filter buys, and what it costs

Same 20-fold walk-forward as Phase 3, Strategy A simulated twice per
fold — once unfiltered, once filtered. The unfiltered leg reproduces
Phase 3 (+244.4% vs +245.0% pooled), so the comparison is
apples-to-apples.

| Pooled (260 weeks) | Unfiltered | + 200d MA filter |
|---|---|---|
| Total return | +244.4% | **+250.8%** |
| CAGR | +28.1% | **+28.5%** |
| Sharpe | 0.91 | **1.09** |
| Max drawdown | −46.3% | **−28.5%** |
| Avg turnover | 0.464 | 0.395 |
| Weeks invested | 100% | 77.7% (202/260) |
| Exposure switches | — | 16 |

**The drawdown cut is the result.** Max drawdown falls from −46.3% to
−28.5% — an 18 pp improvement, more than a third of the tail risk
removed — and the Sharpe rises from 0.91 to 1.09. The total-return gain
(+6.4 pp over five years) is small and within noise; do not present the
filter as a return enhancer. It is a risk reducer that happens not to
cost return in this sample.

**2022 is where the filter earns its keep.** It was out for 43 of 52
weeks (invested 17%), and the year's loss more than halves: −20.2%
vs −43.5%, drawdown −19.5% vs −45.2%. The fold detail shows how: fold
2 (Feb–May 2022) goes from −23.5% to −6.2%, and fold 4 (Aug–Nov 2022)
from −13.2% to +0.0% — the filter was simply out for the whole quarter.
The long out-period ran from Apr 11 to Nov 28, 2022: seven months in
cash while the bear market did its worst.

**But the filter charges for it — in lag and whipsaws.** The filter
only knows the trend has turned *after* the price has crossed the
average, so it exits late (after part of the fall) and re-enters late
(after part of the recovery). Three concrete costs in this sample:

| Year | Unfiltered | Filtered | What happened |
|---|---|---|---|
| 2023 | +39.4% | +16.8% | Out for one week in late Oct — the week SPY rebounded +5.8%. The filter exited after the drop (Oct 23: SPY $396 vs MA $406) and re-entered after the rebound had already happened (Oct 30: SPY $419 vs MA $406). One missed week cost most of the year's gap. |
| 2025 | +100.9% | +83.3% | Out Mar 10 – May 12 (nine weeks), missing the April rebound. Fold 14 (Feb–May 2025) flips from +0.9% to −7.9%. |
| 2026 | +39.8% | +31.9% | Out Mar 16 – Apr 6 (three weeks), missing part of the spring recovery. Fold 18 drops from +21.9% to +15.0%. |

And the whipsaws: the filter switched 16 times in 260 weeks. Early
2022 alone had four switches in three months (out Jan 17, in Jan 24,
out Feb 7, in Mar 14) as the market chopped around its average while
topping — each switch costs 5 bps and, worse, each re-entry risks
buying back in just before the next leg down. Fold 5 (Nov 2022 – Feb
2023) is the ugly case: filtered −11.3% vs unfiltered −0.9%, because
the filter re-entered in late November and early January during weeks
when the strategy's picks fell. The filter does not only miss rebounds;
it sometimes re-enters into declines.

**The 1% hysteresis band did not help.** Adding a band (switch out only
below MA × 0.99, in only above MA × 1.01) cut switches from 16 to 14 —
but returned less (+233.7% vs +250.8%), with a slightly lower Sharpe
(1.05 vs 1.09) and slightly worse drawdown (−29.5% vs −28.5%). The
band's delayed entries and exits cost more than the two avoided
whipsaws saved. This is reported as a negative result: a reasonable
idea, tested, that did not pay in this sample. (With only 16 switches
to begin with, there was not much whipsaw to remove — the band solves
a problem this filter barely has.)

### A measurement trap: the win rate

The pooled win rate *falls* with the filter (45.0% vs 56.2%) even
though returns improve. That is an artifact, not a finding: cash weeks
return exactly 0%, and 0% is not a win, so 58 cash weeks mechanically
drag the win rate down. On **invested weeks only**, the win rate is
57.9% — slightly *above* the unfiltered 56.2%. The lesson generalises:
any metric that counts weeks must be re-based when a strategy is
sometimes out of the market, or the comparison is meaningless. (The
same trap appears in 2022's filtered win rate: 3.9% — two winning
weeks out of 52 — because 43 weeks were cash. The year's return,
−20.2% vs −43.5%, is the informative number, not the win rate.)

### The honest caveat

The filter's benefit is measured on one five-year window containing a
single bear market. The drawdown cut is the robust result — it follows
directly from being out during the worst months, which the filter did
by construction. The return gain is within noise, and in a different
sample (a bear market with faster reversals, or a choppier top) the
whipsaw costs could exceed the drawdown savings. A second bear market
in the sample would be needed to make the claim stronger, and the
history does not contain one. Present the filter as drawdown insurance
with a known premium (lag + whipsaws), not as a free lunch.

### Current state

As of 2026-11-23 the filter says **OUT (cash)**: SPY $722.29 vs
200-day MA $730.47 (−1.1%). It flipped out in the week of Nov 16
(Friday Nov 20 close $689.81 vs MA $730.29) and has stayed out since.
That is a statement about the trend rule's current position, not a
forecast — the rule will flip back in when the price recovers above
the average, whenever that is.

---

## 7. Volatility-adjusted ranking: a negative result (Phase 5)

### The question

The MU analysis (2026-11-25) showed the RF persistently ranking the
highest-volatility stock #1: MU's `vol_4w` was 1.34, the highest in the
pool, and the model had learned "high vol → likely to land in the top
return quintile". In the single-split test window (Sep 2025–Nov 2026)
that went wrong — MU picks averaged −0.26% with a 47% win rate, and
the highest-conviction picks preceded −22% and −23.6% weeks. The
question was whether to fix it in the *strategy layer* instead of the
model: keep the same RF scores, but select the top-5 by **score / vol**
instead of raw score, so the portfolio prefers "high score *for its
volatility*".

### What was tested

Same RF scores, same 20 walk-forward folds, market-cap universe
(50 tickers, 8 years, 260 out-of-sample weeks). Three selection rules:
raw score, `score / vol_4w`, `score / vol_12w`. MU pick statistics
(how often MU was selected, and its realised next-week return when
selected) were tracked per rule, so the question "did the vol
adjustment fix the MU false picks?" is answered directly.

### The numbers (pooled out-of-sample, 260 weeks)

| Ranking rule | Total | CAGR | Sharpe | Max DD | Turnover | Win rate |
|---|---|---|---|---|---|---|
| Raw score | +561.9% | +45.9% | 1.21 | −43.4% | 0.47 | 56.2% |
| score / vol_4w | +22.6% | +4.2% | 0.31 | −23.1% | 0.61 | 51.5% |
| score / vol_12w | +70.2% | +11.2% | 0.74 | −19.0% | 0.49 | 53.9% |

Dividing by vol crushes the return. The drawdown improvement is real
(−43% → −23%/−19%), but the Sharpe says the trade is bad: return per
unit of risk falls from 1.21 to 0.31/0.74.

### Why it fails

Dividing by vol does not just penalise high-vol stocks — it actively
*rewards* low-vol stocks. And low-vol stocks almost never land in the
top return quintile, which is exactly what the model predicts. The
ranking fights the model's objective: the model is a magnitude
predictor, and volatility is the price of magnitude. The vol_4w
variant flatlines in the big up years (2023: +2.0% vs raw +115.7%;
2025: +3.5% vs +70.3%; 2026: +8.8% vs +64.9%) because it holds calm
stocks that do not move. Turnover also rises (0.61 vs 0.47) because
volatility reshuffles the ranking every week, so the variant pays more
in trading costs for the privilege of holding quieter stocks.

### The MU nuance — the problem is the regime, not the stock

Over the full five-year walk-forward, raw-score MU picks were actually
*good*: picked in 90 of 260 weeks (34.6%), mean next-week return
+2.84%, win rate 61.1% — better than MU's overall mean next-week
return of +1.19% across all 260 weeks. The MU problem is
recent-regime-specific (the 2025–2026 AI unwind), not a permanent
model flaw. The vol filter "fixes" MU by almost never picking it (9
weeks for vol_4w, 2 weeks for vol_12w) — including during the years it
worked. And the few vol_4w picks were mediocre (+1.54% mean) and still
included the −16.0% worst week. Avoiding a stock entirely is not the
same as timing it well.

### What to try instead

- **A vol cap, not a vol division.** Exclude the top-volatility decile
  from selection instead of dividing by vol: it removes the extreme
  names without rewarding the calm ones. Untested — the natural next
  experiment.
- **The Phase 4 regime filter for drawdown control.** It takes the
  portfolio out of the market during selloffs instead of changing
  which stocks are picked — the right tool for a regime problem,
  which is what the MU failure turned out to be.
- **Vol-weighted position sizing within the top-5.** HRP already does
  this at 10 holdings (§3); the ranking need not carry the risk
  adjustment alone.

### Also noticed: the market-cap universe is stronger

Same walk-forward setup, market-cap universe: pooled +561.9%, CAGR
+45.9%, Sharpe 1.21 — versus the name-order universe's +244%, CAGR
+28.1%, Sharpe 0.91 (Phase 3), at the same mean AUC (0.581 vs 0.580).
Mega-cap momentum rode the 2023–2026 AI boom, and the model captured
more of it in the concentrated universe. Universe choice moves results
more than the model does — the same lesson as Phase 3's market-cap vs
name-order split, now confirmed out-of-sample.

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
7. "The 200-day MA regime filter cuts the pooled drawdown from −46%
   to −28% and lifts the Sharpe from 0.91 to 1.09 — it is drawdown
   insurance, not a return enhancer: it cost most of 2023's return by
   missing the October rebound."
8. "The filter is currently OUT (SPY $722 vs 200-day MA $730 as of
   Nov 23) — the trend rule says cash until the price recovers above
   its average."
9. "Ranking by score divided by volatility was a clear negative
   result: it cut the pooled return from +562% to +23% (vol_4w) and
   the Sharpe from 1.21 to 0.31. Dividing by vol rewards calm stocks
   that never jump — it fights the model's objective, which is
   predicting big weekly moves."
10. "The MU problem turned out to be regime-specific, not a stock
    problem: over five walk-forward years the model's MU picks averaged
    +2.84% with a 61% win rate. The failures cluster in the 2025–2026
    AI unwind, which is what the regime filter — not the ranking — is
    for."
