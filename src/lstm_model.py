"""LSTM comparison model (Phase 2).

A small Long Short-Term Memory network trained on panel sequences: for
each (ticker, week) row, the input is the trailing ``window`` weeks of the
existing numeric features; the target is the same top-quintile
next-week-return label as the tree models, so ROC AUC is directly
comparable.

Design choices (deliberately conservative -- see README discussion):

* **Small network only.** 1 LSTM layer (48 units), dropout, one sigmoid
  output. Weekly stock returns have a very low signal-to-noise ratio, so
  a large network mostly memorises noise. The LSTM is a *comparison*
  model: if it loses to RandomForest, that is a legitimate finding, not
  a failure.
* **Same split as Phase 1.** Sequences are built from the full panel and
  split by the same time cutoff as ``target.time_split``, so the test
  AUC is directly comparable to RF (0.556) and XGBoost (0.508).
* **Scaling fit on train only.** A StandardScaler is fit on the training
  sequences (per feature, flattened over time steps) and applied to the
  test sequences -- no leakage.
* **Fixed seeds.** Python, NumPy and TensorFlow seeds are all set, and
  TF threading is capped so runs are reproducible on a CPU box.
* **Early stopping** on validation loss, with the validation set taken
  as the last 15% of *training* weeks (time-based, no shuffle).

TensorFlow is an optional dependency: if it is not installed,
``tensorflow_available()`` is False and the pipeline skips the LSTM
cleanly (same pattern as XGBoost in ``models.py``).
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

# Keep TF quiet and CPU-friendly before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    _TF_AVAILABLE = False

SEED = 42
WINDOW = 12          # trailing weeks per sequence
LSTM_UNITS = 48
DROPOUT = 0.2
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
VAL_FRAC = 0.15      # last 15% of train weeks -> validation


def tensorflow_available() -> bool:
    return _TF_AVAILABLE


def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TF_AVAILABLE:
        tf.random.set_seed(seed)
        # Cap threading: the box is CPU-only and shared; more threads
        # than cores just adds scheduling overhead.
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(
    panel: pd.DataFrame,
    feature_cols: list[str],
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build trailing-window sequences from the modelling panel.

    Parameters
    ----------
    panel : panel with ``date, ticker, target`` and all ``feature_cols``
        (output of ``features.build_features`` + ``target.make_target``).
    feature_cols : numeric feature columns to include in each time step.
    window : number of trailing weeks per sequence (the current week is
        the last step).

    Returns
    -------
    (X, y, meta): X has shape (n, window, n_features); y is the binary
    target; meta is the panel subset the sequences correspond to (one row
    per sequence, with ``date, ticker, target, next_ret``). Sequences
    whose window is not exactly ``window`` consecutive weeks (IPO gaps,
    halts) are skipped.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    X_list, y_list, meta_rows = [], [], []
    for ticker, g in panel.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        vals = g[feature_cols].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"])
        for i in range(window - 1, len(g)):
            # Require exactly consecutive weeks: a 12-step window must
            # span 11 weeks (77 days) -- otherwise the sequence mixes
            # non-adjacent bars and the "trailing 12 weeks" claim breaks.
            span = (dates.iloc[i] - dates.iloc[i - window + 1]).days
            if span != 7 * (window - 1):
                continue
            X_list.append(vals[i - window + 1: i + 1])
            y_list.append(g["target"].iloc[i])
            meta_rows.append(g.iloc[i])
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    print(f"LSTM sequences: {len(X):,} sequences, shape {X.shape}, "
          f"positive rate {y.mean():.3f}")
    return X, y, meta


# ---------------------------------------------------------------------------
# Scaling (fit on train only)
# ---------------------------------------------------------------------------
class SequenceScaler:
    """StandardScaler over (samples * timesteps, features)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        self.mean_ = flat.mean(axis=0)
        self.scale_ = flat.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_lstm_model(n_features: int, seed: int = SEED):
    """Small LSTM classifier: LSTM(48) -> Dropout -> Dense(sigmoid)."""
    if not _TF_AVAILABLE:
        raise ImportError(
            "tensorflow is not installed. Install it "
            "(pip install tensorflow-cpu) or skip the LSTM model."
        )
    _set_seeds(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(WINDOW, n_features)),
        keras.layers.LSTM(
            LSTM_UNITS,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed),
        ),
        keras.layers.Dropout(DROPOUT, seed=seed),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_dates: pd.Series,
    seed: int = SEED,
    verbose: int = 0,
):
    """Train the LSTM with time-based internal validation.

    The last ``VAL_FRAC`` of training weeks form the validation set
    (early stopping monitors validation loss). Returns
    (model, scaler, history, val_auc).
    """
    if not _TF_AVAILABLE:
        raise ImportError("tensorflow is not installed.")
    _set_seeds(seed)

    weeks = sorted(pd.to_datetime(train_dates).unique())
    cut = weeks[int(len(weeks) * (1.0 - VAL_FRAC))]
    d = pd.to_datetime(train_dates)
    val_mask = (d >= cut).to_numpy()
    tr_mask = ~val_mask

    scaler = SequenceScaler()
    X_tr = scaler.fit_transform(X_train[tr_mask])
    X_va = scaler.transform(X_train[val_mask])

    model = build_lstm_model(X_train.shape[-1], seed=seed)
    # Class weight compensates the 20% positive rate (keeps the loss
    # from collapsing to "always predict 0").
    n_pos = float(y_train[tr_mask].sum())
    n_neg = float((1 - y_train[tr_mask]).sum())
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1.0)}
    history = model.fit(
        X_tr, y_train[tr_mask],
        validation_data=(X_va, y_train[val_mask]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True),
        ],
        verbose=verbose,
    )
    val_auc = float(history.history["val_auc"][-1])
    print(f"LSTM trained: {len(history.history['loss'])} epochs, "
          f"val AUC {val_auc:.3f}")
    return model, scaler, history, val_auc


def score_lstm(model, scaler: SequenceScaler, X: np.ndarray) -> np.ndarray:
    """P(target=1) for each sequence."""
    return model.predict(scaler.transform(X), verbose=0).flatten()
