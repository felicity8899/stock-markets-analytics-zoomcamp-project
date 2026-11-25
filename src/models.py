"""Models.

Three models, one common interface: each ``train_*`` returns a fitted
estimator and each ``score_*`` returns a per-(week, ticker) *score* where a
higher score means "more likely to be a top-quintile stock next week".
The simulator selects the top-N stocks by score, so classifiers contribute
P(target=1) and the factor regression contributes predicted return.

Models
------
1. RandomForest -- course baseline. 2026 course best config:
   max_depth=13, n_estimators=100, decision threshold 0.55.
2. Fama-French-style factor regression -- Ridge regression of next-week
   return on market excess return, SMB, HML, momentum and volatility.
   Interpretable industry-style baseline (see README).
3. XGBoost -- gradient boosting classifier (optional; skipped cleanly if
   the package is not installed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _XGB_AVAILABLE = False

# Course 2026 best RandomForest configuration.
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=13,
    random_state=42,
    n_jobs=-1,
)
# Decision threshold from the 2026 course material: only trade when
# P(top-quintile) exceeds this.
RF_THRESHOLD = 0.55

# Fama-French-style factor regression features.
FF_FEATURES = ["mkt_rf", "smb", "hml", "ret_12w", "vol_12w"]

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


def xgboost_available() -> bool:
    return _XGB_AVAILABLE


# ---------------------------------------------------------------------------
# RandomForest (course baseline)
# ---------------------------------------------------------------------------
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def score_random_forest(model, X: pd.DataFrame) -> np.ndarray:
    """P(target=1) for each row."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Fama-French-style factor regression (Ridge on next-week return)
# ---------------------------------------------------------------------------
def train_factor_regression(X_train: pd.DataFrame, y_ret_train: pd.Series):
    """Ridge regression predicting next-week return from factor exposures.

    Features are the Fama-French weekly factors for the current week plus
    the stock's own momentum and volatility. StandardScaler is inside the
    pipeline so scaling is fit on train only (no leakage).
    """
    cols = [c for c in FF_FEATURES if c in X_train.columns]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train[cols], y_ret_train)
    # Pipeline.feature_names_in_ is read-only, so keep the column list
    # under a custom attribute for scoring.
    model.ff_cols_ = cols  # type: ignore[attr-defined]
    return model


def score_factor_regression(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted next-week return (used as the ranking score)."""
    cols = list(model.ff_cols_)
    return model.predict(X[cols])


# ---------------------------------------------------------------------------
# XGBoost (optional)
# ---------------------------------------------------------------------------
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    if not _XGB_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Install it (pip install xgboost) "
            "or skip this model -- see README 'TODO / Phase 2'."
        )
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def score_xgboost(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def add_scores(
    df: pd.DataFrame,
    model,
    score_fn,
    feature_cols: list[str],
    score_col: str = "score",
) -> pd.DataFrame:
    """Return ``df`` with a ``score_col`` column of model scores."""
    out = df.copy()
    out[score_col] = score_fn(model, out[feature_cols])
    return out


def classification_report_simple(y_true, y_score, threshold: float = 0.5):
    """Small precision/recall summary for a probability score."""
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision@thr": precision_score(y_true, y_pred, zero_division=0),
        "recall@thr": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
