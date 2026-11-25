# EDA outline (Phase 2 — notebooks to be built)

Planned notebooks under `notebooks/` (run in order; each exports its logic
to `src/` so `main.py` stays the single source of truth):

1. **`01_data_overview.ipynb`** — universe coverage (tickers × weeks),
   missing-data heatmap, price/volume sanity checks, weekly return
   distribution (fat tails? skew?), SPY vs universe comparison.
2. **`02_features_eda.ipynb`** — distribution of every NUMERIC feature;
   correlation of each feature with `target` (point-biserial) and with
   `next_ret`; correlation heatmap between features (multicollinearity
   check for the Ridge model); momentum/volatility decile analysis
   (do high-momentum stocks actually outperform next week?).
3. **`03_target_analysis.ipynb`** — base rate of the top-quintile label by
   year and by month (seasonality?); persistence of winners (does last
   week's top quintile stay top?); sector concentration of picks.
4. **`04_model_comparison.ipynb`** — ROC AUC / precision / recall per model
   on the test period; feature importances (RF + XGBoost); calibration of
   predicted probabilities; error analysis by ticker and by month.
5. **`05_strategy_deep_dive.ipynb`** — equity curves vs SPY; rolling Sharpe
   and rolling returns; drawdown periods (what happened in the worst
   weeks?); HRP vs equal-weight weight evolution; turnover and cost
   decomposition; long-short extension.
