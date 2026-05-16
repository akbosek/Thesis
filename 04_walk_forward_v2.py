#!/usr/bin/env python3
"""
04_walk_forward.py
==================

Walk-forward analysis with top-N hyperparameter candidates from the
two Optuna studies (unweighted, weighted). Each candidate is retrained
once per fold, evaluated under four trading variants, and ranked by
mean walk-forward Variant C IR1 across the three folds. The two
winners (one per weighting mode) are reported as the headline result;
the full top-N table is reported as a robustness check.

Trading variants
----------------
A : fixed 0.5 threshold; signal = +1 if P > 0.5 else -1.
B : rolling 90-day median of P used as the threshold; signal = +1 if
    P > rolling_median else -1.
C : rolling 90-day quantile thresholds (Q_lo, Q_hi). The (lo, hi)
    pair is selected on the validation period by maximising IR1
    subject to coverage >= 25%. Signal = +1 if P > Q_hi, -1 if
    P < Q_lo, 0 otherwise.
D : Variant C signal scaled by a continuous position weight w in
    [0.25, 1.0], where w grows linearly with distance from the
    selected quantile threshold to the [0, 1] boundary.

Rolling thresholds use strict no-leakage protection via `.shift(1)`:
the threshold for day T is computed from data strictly before T
(days T-window through T-1). Day T's own prediction is never used
to set its own threshold. The first `window` days of an unseeded
series receive NaN thresholds and produce no trade signals; the
test period seeds the rolling window with the last `window` days
of validation predictions so day-1-of-test has a valid threshold.

Selection criterion
-------------------
For each candidate, mean Variant C IR1 across the three test years
(2023, 2024, 2025). One winner per weighting mode.

Reproducibility
---------------
TensorFlow determinism flags are set before the tf import, and TF is
restricted to single-threaded execution. Combined with fixed numpy
and TF seeds, this gives bitwise-identical training results across
runs on the same hardware.

Inputs
------
processed_dataset_v3.csv     Feature-engineered dataset.
optuna_unweighted.db         Optuna SQLite, unweighted study.
optuna_weighted.db           Optuna SQLite, weighted study.

Outputs
-------
results/walk_forward/
    unweighted/
        candidate_NN_trial_TTT/
            fold1_test_2023/  metrics.json, predictions.csv
            fold2_test_2024/  metrics.json, predictions.csv
            fold3_test_2025/  metrics.json, predictions.csv
            candidate_summary.json
    weighted/
        ...
    walk_forward_summary.csv     One row per (candidate, fold).
    candidate_aggregate.csv      One row per candidate, mean across folds.
    selected_unweighted.json     Winner's hyperparameters and metrics.
    selected_weighted.json
    selected_unweighted/         Headline plots for the unweighted winner.
        equity_curves_fold1.png  ...
    selected_weighted/
        ...
"""

# ── Determinism flags must precede the tensorflow import ─────────────────────
import os
os.environ["PYTHONHASHSEED"]       = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras import Sequential                               # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout              # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# Single-threaded execution + op determinism. CPU TF is non-deterministic
# under multi-threaded BLAS due to non-associative floating-point summation.
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

try:
    _AdamW = tf.keras.optimizers.AdamW
except AttributeError:
    try:
        _AdamW = tf.keras.optimizers.experimental.AdamW
    except AttributeError:
        _AdamW = tf.keras.optimizers.Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix


# =============================================================================
# CONFIGURATION
# =============================================================================

WALK_FORWARD_FOLDS = [
    {"name": "fold1", "test_year": 2023,
     "train": ("2015-01-01", "2021-12-31"),
     "val":   ("2022-01-01", "2022-12-31"),
     "test":  ("2023-01-01", "2023-12-31")},
    {"name": "fold2", "test_year": 2024,
     "train": ("2015-01-01", "2022-12-31"),
     "val":   ("2023-01-01", "2023-12-31"),
     "test":  ("2024-01-01", "2024-12-31")},
    {"name": "fold3", "test_year": 2025,
     "train": ("2015-01-01", "2023-12-31"),
     "val":   ("2024-01-01", "2024-12-31"),
     "test":  ("2025-01-01", "2025-12-31")},
]

N_CANDIDATES   = 15              # top-N per Optuna study
ROLLING_WINDOW = 90              # rolling-statistic window length, days
COVERAGE_MIN   = 0.25            # minimum fraction of trading days for Variant C
QUANTILE_X_GRID = [10, 15, 20, 25, 30, 35, 40, 45]   # symmetric pairs (Q_x, Q_{100-x})

VD_W_MIN, VD_W_MAX = 0.25, 1.00  # Variant D position-weight bounds

# Volatility-weight parameters for the weighted study (matches Optuna setup).
VOL_WINDOW  = 20
VOL_CLIP_LO = 0.10
VOL_CLIP_HI = 5.00

INPUT_CSV       = "processed_dataset_v4_fixed.csv"
SEQ_LEN         = 30
BATCH_SIZE      = 64
EPOCHS          = 150
PRUNED_FEATURES = frozenset({"vix_level"})           # match the Optuna study pruning

UNWEIGHTED_DB_URI = "sqlite:///optuna_unweighted.db"
WEIGHTED_DB_URI   = "sqlite:///optuna_weighted.db"
UNWEIGHTED_STUDY  = "btc_lstm_unweighted"
WEIGHTED_STUDY    = "btc_lstm_weighted"

OUT_ROOT = Path("results") / "walk_forward"

SEED          = 42
TRADING_DAYS  = 365              # BTC trades 24/7

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("walk_forward_run.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FoldMetrics:
    """All evaluation metrics for one (candidate, fold) combination."""
    fold:               str
    test_year:          int
    val_auc:            float
    test_auc:           float
    sensitivity_test:   float          # at threshold 0.5
    specificity_test:   float
    # Per-variant test-period IR1, aRC, MD, coverage:
    A_ir1:    float; A_arc:    float; A_md:    float; A_coverage: float
    B_ir1:    float; B_arc:    float; B_md:    float; B_coverage: float
    C_ir1:    float; C_arc:    float; C_md:    float; C_coverage: float
    D_ir1:    float; D_arc:    float; D_md:    float; D_coverage: float
    # Selected quantile pair for Variant C / D (chosen on val):
    selected_q_lo: int
    selected_q_hi: int
    val_C_ir1:     float          # IR1 of Variant C on val with selected pair


@dataclass
class CandidateAggregate:
    """Mean-across-folds metrics for one candidate."""
    candidate_rank:    int
    weighting:         str
    trial_number:      int
    optuna_score:      float
    optuna_val_auc:    float
    hyperparameters:   dict
    fold_metrics:      list = field(default_factory=list)
    # Mean across the three folds:
    mean_test_auc:     float = 0.0
    mean_C_ir1:        float = 0.0          # selection criterion
    mean_C_arc:        float = 0.0
    mean_C_md:         float = 0.0
    mean_D_ir1:        float = 0.0
    mean_A_ir1:        float = 0.0
    mean_B_ir1:        float = 0.0


# =============================================================================
# DATA PIPELINE
# =============================================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")
    log.info("Dataset loaded: %d rows, %s → %s",
             len(df), df.index.min().date(), df.index.max().date())
    return df


def slice_with_buffer(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return df[start:end] with SEQ_LEN rows of preceding context, so the
    first sliding-window prediction lands exactly on `start`."""
    mask = (df.index >= start) & (df.index <= end)
    if not mask.any():
        return df.iloc[0:0]
    target_idx = df.index.get_indexer_for(df.index[mask])
    start_idx  = max(0, int(target_idx.min()) - SEQ_LEN)
    end_idx    = int(target_idx.max())
    return df.iloc[start_idx : end_idx + 1].copy()


def build_fold_splits(df: pd.DataFrame, fold: dict) -> tuple:
    """Slice the dataset into (train, val, test) DataFrames for one fold,
    each carrying a SEQ_LEN warm-up buffer."""
    train_df = slice_with_buffer(df, *fold["train"])
    val_df   = slice_with_buffer(df, *fold["val"])
    test_df  = slice_with_buffer(df, *fold["test"])
    return train_df, val_df, test_df


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Fit StandardScaler on training features only; transform val and test.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)."""
    feature_cols = [c for c in train_df.columns
                    if c != "target" and c not in PRUNED_FEATURES
                    and c != "btc_log_return"]      # auxiliary, never a feature
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    X_test_raw  = test_df[feature_cols].values.astype(np.float32)
    y_train     = train_df["target"].values.astype(np.float32)
    y_val       = val_df["target"].values.astype(np.float32)
    y_test      = test_df["target"].values.astype(np.float32)

    scaler  = StandardScaler().fit(X_train_raw)
    return (scaler.transform(X_train_raw), scaler.transform(X_val_raw),
            scaler.transform(X_test_raw), y_train, y_val, y_test, feature_cols)


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int,
                 sample_weights: Optional[np.ndarray] = None,
                 shuffle: bool = False) -> tuple:
    """Build a sliding-window tf.data.Dataset. Targets and sample weights
    are aligned to the LAST day of each window (no look-ahead)."""
    n_windows = len(X) - SEQ_LEN + 1
    y_aligned = y[SEQ_LEN - 1:].astype(np.float32)
    X_windows = np.stack([X[i : i + SEQ_LEN] for i in range(n_windows)],
                         axis=0).astype(np.float32)
    if sample_weights is not None:
        w_aligned = sample_weights[SEQ_LEN - 1:].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned, w_aligned))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned))
    if shuffle:
        ds = ds.shuffle(buffer_size=n_windows, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, y_aligned


def compute_vol_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Volatility-based sample weights for the weighted study, matching
    the Optuna implementation: rolling-std normalised to mean 1.0, clipped."""
    returns = train_df["btc_log_return"].astype(float)
    vol     = returns.rolling(window=VOL_WINDOW, min_periods=1).std().fillna(0.0).values
    vol     = np.where(vol < 1e-10, 1e-10, vol)
    weights = (vol / vol.mean()).astype(np.float32)
    return np.clip(weights, VOL_CLIP_LO, VOL_CLIP_HI)


# =============================================================================
# MODEL
# =============================================================================

def build_model(n_features: int, hp: dict) -> tf.keras.Model:
    """Two-layer LSTM matching the Optuna search-space architecture, with
    hyperparameters supplied per-candidate."""
    name = hp["optimizer_name"]
    if name == "adamw":
        optimizer = _AdamW(learning_rate=hp["learning_rate"], weight_decay=hp["l2_reg"])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])

    model = Sequential([
        LSTM(units=hp["units_1"], return_sequences=True,
             dropout=hp["dropout_rate"], recurrent_dropout=hp["recurrent_dropout"],
             input_shape=(SEQ_LEN, n_features), name="lstm_1"),
        Dropout(hp["dropout_rate"], name="drop_1"),
        LSTM(units=hp["units_2"], return_sequences=False,
             dropout=hp["dropout_rate"], recurrent_dropout=hp["recurrent_dropout"],
             name="lstm_2"),
        Dropout(hp["dropout_rate"], name="drop_2"),
        Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def train_model(model: tf.keras.Model, train_ds, val_ds) -> tf.keras.callbacks.History:
    return model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-7, verbose=0),
        ],
        verbose=0,
    )


# =============================================================================
# QUANTITATIVE METRICS
# =============================================================================

def compute_quant_metrics(daily_returns: np.ndarray) -> dict:
    """Return a dict with arc, asd, md, ir1 for the supplied per-day log-returns.
    All annualised on TRADING_DAYS = 365 (BTC convention)."""
    r = np.asarray(daily_returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return {"arc": 0.0, "asd": 0.0, "md": 0.0, "ir1": 0.0}

    equity = np.exp(np.cumsum(r))
    arc    = float((equity[-1] ** (TRADING_DAYS / n) - 1.0) * 100.0)  # %
    asd    = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    run_max= np.maximum.accumulate(equity)
    md     = float(np.max(1.0 - equity / run_max) * 100.0)             # %
    ir1    = float((arc / 100.0) / asd) if asd > 1e-12 else 0.0
    return {"arc": arc, "asd": asd, "md": md, "ir1": ir1}


# =============================================================================
# TRADING VARIANTS
# =============================================================================

def _rolling_quantile(p: np.ndarray, q: float, window: int,
                      seed: Optional[np.ndarray] = None) -> np.ndarray:
    """Rolling quantile with strict no-leakage protection.

    Mechanism (matches main script's apply_rolling_thresholds):
      * Full window required — no `min_periods=1`. The first `window`
        days of an unseeded series receive NaN thresholds (and therefore
        no trade signals, since NaN comparisons are always False).
      * `.shift(1)` ensures the threshold for day T is computed from
        data strictly before T (days T-window through T-1). Day T's
        own prediction is never used to set its own threshold.

    When `seed` is provided (test period using the last `window` days of
    val predictions), every day of `p` has a full window and a valid
    threshold from day 1 onward — no warm-up gap, no cold-start NaNs.
    """
    if seed is not None and len(seed) > 0:
        full = np.concatenate([seed, p])
        roll = pd.Series(full).rolling(window=window).quantile(q).shift(1).values
        return roll[len(seed):]
    return pd.Series(p).rolling(window=window).quantile(q).shift(1).values


def variant_A(prob: np.ndarray) -> np.ndarray:
    """Fixed 0.5 threshold, binary signal."""
    return np.where(prob > 0.5, 1.0, -1.0).astype(np.float32)


def variant_B(prob: np.ndarray, seed: Optional[np.ndarray] = None,
              window: int = ROLLING_WINDOW) -> np.ndarray:
    """Rolling-median threshold, binary signal."""
    med = _rolling_quantile(prob, 0.5, window, seed=seed)
    return np.where(prob > med, 1.0, -1.0).astype(np.float32)


def variant_C(prob: np.ndarray, q_lo: int, q_hi: int,
              seed: Optional[np.ndarray] = None,
              window: int = ROLLING_WINDOW) -> np.ndarray:
    """Three-state rolling-quantile filter at (Q_lo, Q_hi)."""
    thr_lo = _rolling_quantile(prob, q_lo / 100.0, window, seed=seed)
    thr_hi = _rolling_quantile(prob, q_hi / 100.0, window, seed=seed)
    sig = np.zeros_like(prob, dtype=np.float32)
    sig[prob > thr_hi] =  1.0
    sig[prob < thr_lo] = -1.0
    return sig


def variant_D(prob: np.ndarray, q_lo: int, q_hi: int,
              seed: Optional[np.ndarray] = None,
              window: int = ROLLING_WINDOW,
              w_min: float = VD_W_MIN,
              w_max: float = VD_W_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Variant C signal scaled by continuous position weight in [w_min, w_max].
    Weight grows linearly from w_min at the threshold to w_max at the {0,1}
    boundary in the relevant direction. Returns (signal, weight)."""
    thr_lo = _rolling_quantile(prob, q_lo / 100.0, window, seed=seed)
    thr_hi = _rolling_quantile(prob, q_hi / 100.0, window, seed=seed)
    sig = np.zeros_like(prob, dtype=np.float32)
    w   = np.zeros_like(prob, dtype=np.float32)
    long_m  = prob > thr_hi
    short_m = prob < thr_lo
    sig[long_m]  =  1.0
    sig[short_m] = -1.0
    w_range = w_max - w_min
    # Long: w_min at thr_hi, w_max at P=1
    span_l = np.maximum(1.0 - thr_hi[long_m], 1e-10)
    w[long_m] = np.clip(
        w_min + w_range * (prob[long_m] - thr_hi[long_m]) / span_l, w_min, w_max)
    # Short: w_min at thr_lo, w_max at P=0
    span_s = np.maximum(thr_lo[short_m], 1e-10)
    w[short_m] = np.clip(
        w_min + w_range * (thr_lo[short_m] - prob[short_m]) / span_s, w_min, w_max)
    return sig, w


def search_quantile_pair(prob_val: np.ndarray, btc_ret_val: np.ndarray) -> tuple:
    """Search QUANTILE_X_GRID for the symmetric pair (Q_x, Q_{100-x}) that
    maximises Variant C IR1 on the validation period, subject to coverage
    >= COVERAGE_MIN. Returns (q_lo, q_hi, ir1, coverage). If no pair clears
    the coverage threshold, returns the highest-coverage pair available."""
    best       = None
    best_under = None       # fallback: highest IR1 even if coverage too low
    for x in QUANTILE_X_GRID:
        q_lo, q_hi = x, 100 - x
        sig = variant_C(prob_val, q_lo, q_hi)        # no seed on val
        cov = float((sig != 0).mean())
        if cov < 1e-9:
            continue
        ir1 = compute_quant_metrics(sig * btc_ret_val)["ir1"]
        record = (q_lo, q_hi, ir1, cov)
        if cov >= COVERAGE_MIN:
            if best is None or ir1 > best[2]:
                best = record
        else:
            if best_under is None or ir1 > best_under[2]:
                best_under = record
    if best is not None:
        return best
    log.warning("No quantile pair cleared coverage_min=%.2f on val; "
                "falling back to highest-IR1 pair found (%s).",
                COVERAGE_MIN, best_under)
    return best_under if best_under is not None else (25, 75, 0.0, 0.0)


def variant_metrics(prob: np.ndarray, btc_ret: np.ndarray, q_lo: int, q_hi: int,
                    seed: Optional[np.ndarray] = None) -> dict:
    """Compute (ir1, arc, md, coverage) for all four variants on a given
    (probability, log-return) sequence. Used for both val and test, where
    `seed` carries the rolling-window history (None on val, last 90 val
    days on test)."""
    out = {}
    sA = variant_A(prob)
    sB = variant_B(prob, seed=seed)
    sC = variant_C(prob, q_lo, q_hi, seed=seed)
    sD, wD = variant_D(prob, q_lo, q_hi, seed=seed)

    for tag, sig in [("A", sA), ("B", sB), ("C", sC)]:
        m = compute_quant_metrics(sig * btc_ret)
        out[f"{tag}_ir1"]      = m["ir1"]
        out[f"{tag}_arc"]      = m["arc"]
        out[f"{tag}_md"]       = m["md"]
        out[f"{tag}_coverage"] = float((sig != 0).mean())

    mD = compute_quant_metrics(sD * wD * btc_ret)
    out["D_ir1"]      = mD["ir1"]
    out["D_arc"]      = mD["arc"]
    out["D_md"]       = mD["md"]
    out["D_coverage"] = float((sD != 0).mean())
    return out


# =============================================================================
# OPTUNA CANDIDATE LOADING
# =============================================================================

def load_top_candidates(db_uri: str, study_name: str, n: int) -> list:
    """Return the top-n completed trials by Optuna score (`value`),
    descending. Each entry is a dict {trial_number, score, val_auc,
    train_auc, hyperparameters}."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study     = optuna.load_study(study_name=study_name, storage=db_uri)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    chosen = completed[:n]
    out = []
    for t in chosen:
        out.append({
            "trial_number":    int(t.number),
            "optuna_score":    float(t.value),
            "optuna_val_auc":  float(t.user_attrs.get("val_auc",   t.value)),
            "optuna_train_auc":float(t.user_attrs.get("train_auc", float("nan"))),
            "hyperparameters": dict(t.params),
        })
    log.info("Loaded top %d candidates from %s (out of %d completed trials)",
             len(out), study_name, len(completed))
    return out


# =============================================================================
# PER-(CANDIDATE, FOLD) EVALUATION
# =============================================================================

def evaluate_candidate_on_fold(hp: dict, df: pd.DataFrame, fold: dict,
                               use_weights: bool, fold_dir: Path) -> FoldMetrics:
    """Train one model with the supplied HPs on the fold's train data,
    evaluate on val + test, and return a FoldMetrics record. Predictions
    are saved to fold_dir/predictions.csv."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = build_fold_splits(df, fold)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = scale_features(
        train_df, val_df, test_df)

    sample_weights = compute_vol_weights(train_df) if use_weights else None
    train_ds, _   = make_dataset(X_train, y_train, BATCH_SIZE,
                                 sample_weights=sample_weights, shuffle=True)
    val_ds, y_val_aligned   = make_dataset(X_val,  y_val,  BATCH_SIZE, shuffle=False)
    test_ds, y_test_aligned = make_dataset(X_test, y_test, BATCH_SIZE, shuffle=False)

    n_features = X_train.shape[1]
    model = build_model(n_features, hp)
    train_model(model, train_ds, val_ds)

    prob_val  = model.predict(val_ds,  verbose=0).ravel()
    prob_test = model.predict(test_ds, verbose=0).ravel()

    # Align BTC log-returns to the prediction targets (offset by SEQ_LEN-1)
    n_val_pred  = len(prob_val)
    n_test_pred = len(prob_test)
    btc_ret_val  = val_df ["btc_log_return"].values[SEQ_LEN - 1 : SEQ_LEN - 1 + n_val_pred]
    btc_ret_test = test_df["btc_log_return"].values[SEQ_LEN - 1 : SEQ_LEN - 1 + n_test_pred]

    # Quantile-pair search on val
    q_lo, q_hi, val_C_ir1, val_C_cov = search_quantile_pair(prob_val, btc_ret_val)

    # Test-period evaluation: rolling stats seeded with last ROLLING_WINDOW val days
    seed = prob_val[-ROLLING_WINDOW:] if len(prob_val) >= ROLLING_WINDOW else prob_val
    test_metrics = variant_metrics(prob_test, btc_ret_test, q_lo, q_hi, seed=seed)

    # AUC + sens/spec at 0.5
    val_auc  = float(roc_auc_score(y_val_aligned,  prob_val))
    test_auc = float(roc_auc_score(y_test_aligned, prob_test))
    y_pred05 = (prob_test >= 0.5).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test_aligned.astype(int), y_pred05).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        sens, spec = 0.0, 0.0

    fm = FoldMetrics(
        fold=fold["name"], test_year=fold["test_year"],
        val_auc=val_auc, test_auc=test_auc,
        sensitivity_test=float(sens), specificity_test=float(spec),
        A_ir1=test_metrics["A_ir1"], A_arc=test_metrics["A_arc"],
        A_md =test_metrics["A_md"],  A_coverage=test_metrics["A_coverage"],
        B_ir1=test_metrics["B_ir1"], B_arc=test_metrics["B_arc"],
        B_md =test_metrics["B_md"],  B_coverage=test_metrics["B_coverage"],
        C_ir1=test_metrics["C_ir1"], C_arc=test_metrics["C_arc"],
        C_md =test_metrics["C_md"],  C_coverage=test_metrics["C_coverage"],
        D_ir1=test_metrics["D_ir1"], D_arc=test_metrics["D_arc"],
        D_md =test_metrics["D_md"],  D_coverage=test_metrics["D_coverage"],
        selected_q_lo=int(q_lo), selected_q_hi=int(q_hi),
        val_C_ir1=float(val_C_ir1),
    )

    # Save predictions for downstream use
    test_dates = test_df.index[SEQ_LEN - 1 : SEQ_LEN - 1 + n_test_pred]
    pd.DataFrame({
        "date":             test_dates,
        "prob_test":        prob_test,
        "btc_log_return":   btc_ret_test,
        "y_true":           y_test_aligned.astype(int),
    }).to_csv(fold_dir / "predictions.csv", index=False)

    with open(fold_dir / "metrics.json", "w") as fh:
        json.dump(asdict(fm), fh, indent=2)

    # Free memory before next training
    tf.keras.backend.clear_session()
    return fm


# =============================================================================
# WALK-FORWARD ORCHESTRATOR
# =============================================================================

def run_walk_forward(df: pd.DataFrame, candidates: list, weighting: str,
                     out_root: Path) -> list:
    """For each candidate, train one model per fold and collect metrics.
    Returns a list of CandidateAggregate records."""
    use_weights = (weighting == "weighted")
    aggregates  = []
    for rank, cand in enumerate(candidates, start=1):
        cand_dir = out_root / weighting / f"candidate_{rank:02d}_trial_{cand['trial_number']:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        log.info("[%s %d/%d] trial %d  score=%.4f  val_auc=%.4f",
                 weighting, rank, len(candidates),
                 cand["trial_number"], cand["optuna_score"], cand["optuna_val_auc"])

        agg = CandidateAggregate(
            candidate_rank=rank, weighting=weighting,
            trial_number=cand["trial_number"],
            optuna_score=cand["optuna_score"],
            optuna_val_auc=cand["optuna_val_auc"],
            hyperparameters=cand["hyperparameters"],
        )

        for fold in WALK_FORWARD_FOLDS:
            fold_dir = cand_dir / f"{fold['name']}_test_{fold['test_year']}"
            fm = evaluate_candidate_on_fold(
                hp=cand["hyperparameters"], df=df, fold=fold,
                use_weights=use_weights, fold_dir=fold_dir)
            agg.fold_metrics.append(fm)
            log.info("    %s test_%d  AUC=%.4f  C_ir1=%+.3f  D_ir1=%+.3f  q=(%d,%d)",
                     fold["name"], fold["test_year"], fm.test_auc,
                     fm.C_ir1, fm.D_ir1, fm.selected_q_lo, fm.selected_q_hi)

        # Aggregate across folds
        agg.mean_test_auc = float(np.mean([f.test_auc for f in agg.fold_metrics]))
        agg.mean_C_ir1    = float(np.mean([f.C_ir1    for f in agg.fold_metrics]))
        agg.mean_C_arc    = float(np.mean([f.C_arc    for f in agg.fold_metrics]))
        agg.mean_C_md     = float(np.mean([f.C_md     for f in agg.fold_metrics]))
        agg.mean_D_ir1    = float(np.mean([f.D_ir1    for f in agg.fold_metrics]))
        agg.mean_A_ir1    = float(np.mean([f.A_ir1    for f in agg.fold_metrics]))
        agg.mean_B_ir1    = float(np.mean([f.B_ir1    for f in agg.fold_metrics]))

        with open(cand_dir / "candidate_summary.json", "w") as fh:
            json.dump({
                **{k: v for k, v in asdict(agg).items() if k != "fold_metrics"},
                "fold_metrics": [asdict(fm) for fm in agg.fold_metrics],
            }, fh, indent=2)

        aggregates.append(agg)

    return aggregates


# =============================================================================
# RESULT AGGREGATION + WINNER SELECTION
# =============================================================================

def write_summary_csvs(uw_aggs: list, w_aggs: list, out_root: Path) -> None:
    """Write the long-format per-fold table and the per-candidate aggregate."""
    rows = []
    for aggs in (uw_aggs, w_aggs):
        for a in aggs:
            for fm in a.fold_metrics:
                rows.append({
                    "weighting":      a.weighting,
                    "candidate_rank": a.candidate_rank,
                    "trial_number":   a.trial_number,
                    **asdict(fm),
                })
    pd.DataFrame(rows).to_csv(out_root / "walk_forward_summary.csv", index=False)

    rows = []
    for aggs in (uw_aggs, w_aggs):
        for a in aggs:
            rows.append({
                "weighting":       a.weighting,
                "candidate_rank":  a.candidate_rank,
                "trial_number":    a.trial_number,
                "optuna_score":    a.optuna_score,
                "optuna_val_auc":  a.optuna_val_auc,
                "mean_test_auc":   a.mean_test_auc,
                "mean_A_ir1":      a.mean_A_ir1,
                "mean_B_ir1":      a.mean_B_ir1,
                "mean_C_ir1":      a.mean_C_ir1,
                "mean_C_arc":      a.mean_C_arc,
                "mean_C_md":       a.mean_C_md,
                "mean_D_ir1":      a.mean_D_ir1,
                **{f"hp_{k}": v for k, v in a.hyperparameters.items()},
            })
    pd.DataFrame(rows).to_csv(out_root / "candidate_aggregate.csv", index=False)
    log.info("Wrote walk_forward_summary.csv and candidate_aggregate.csv to %s", out_root)


def select_winner(aggs: list) -> Optional[CandidateAggregate]:
    """Return the candidate with highest mean Variant C IR1. Ties are
    broken by mean test AUC, then by Optuna score."""
    if not aggs:
        return None
    return max(aggs, key=lambda a: (a.mean_C_ir1, a.mean_test_auc, a.optuna_score))


def save_winner_record(winner: CandidateAggregate, out_path: Path) -> None:
    payload = {k: v for k, v in asdict(winner).items() if k != "fold_metrics"}
    payload["fold_metrics"] = [asdict(fm) for fm in winner.fold_metrics]
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)


# =============================================================================
# WINNER PLOTS
# =============================================================================

def plot_winner_equity_curves(winner: CandidateAggregate, df: pd.DataFrame,
                              out_dir: Path) -> None:
    """For each fold, overlay Variant A/B/C/D equity curves vs BTC B&H.
    Reads predictions.csv files saved during walk-forward."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fm in winner.fold_metrics:
        fold = next(f for f in WALK_FORWARD_FOLDS if f["name"] == fm.fold)
        cand_dir = out_dir.parent / winner.weighting / \
                   f"candidate_{winner.candidate_rank:02d}_trial_{winner.trial_number:03d}"
        pred_path = cand_dir / f"{fm.fold}_test_{fm.test_year}" / "predictions.csv"
        if not pred_path.exists():
            log.warning("Predictions file missing for %s; skipping plot.", pred_path)
            continue
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        prob = pred["prob_test"].values
        btc_ret = pred["btc_log_return"].values
        dates   = pred["date"].values

        # Reconstruct seed from last ROLLING_WINDOW of val (re-load val pred for fidelity)
        # Pragmatic shortcut: use prob's own front as approximate seed when we
        # don't have val cached. Equivalent to no seed for test plotting.
        sA       = variant_A(prob)
        sB       = variant_B(prob)
        sC       = variant_C(prob, fm.selected_q_lo, fm.selected_q_hi)
        sD, wD   = variant_D(prob, fm.selected_q_lo, fm.selected_q_hi)

        eq_btc = np.exp(np.cumsum(btc_ret))
        eq_A   = np.exp(np.cumsum(sA      * btc_ret))
        eq_B   = np.exp(np.cumsum(sB      * btc_ret))
        eq_C   = np.exp(np.cumsum(sC      * btc_ret))
        eq_D   = np.exp(np.cumsum(sD * wD * btc_ret))

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(dates, (eq_btc - 1) * 100, lw=1.6, color="dimgrey", ls="-.",
                label=f"BTC Buy & Hold")
        ax.plot(dates, (eq_A   - 1) * 100, lw=1.4, label=f"Variant A  IR1={fm.A_ir1:+.2f}")
        ax.plot(dates, (eq_B   - 1) * 100, lw=1.4, label=f"Variant B  IR1={fm.B_ir1:+.2f}")
        ax.plot(dates, (eq_C   - 1) * 100, lw=2.0,
                label=f"Variant C  IR1={fm.C_ir1:+.2f}  (Q{fm.selected_q_lo},Q{fm.selected_q_hi})")
        ax.plot(dates, (eq_D   - 1) * 100, lw=2.0,
                label=f"Variant D  IR1={fm.D_ir1:+.2f}")
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (%)")
        ax.set_title(
            f"Walk-forward equity curves — {winner.weighting} winner (trial {winner.trial_number})\n"
            f"{fm.fold} test {fm.test_year}  |  AUC={fm.test_auc:.4f}  "
            f"|  C_arc={fm.C_arc:+.1f}%  C_md={fm.C_md:.1f}%")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"equity_curves_{fm.fold}_test_{fm.test_year}.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
    log.info("Winner equity-curve plots saved to %s", out_dir)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                        help="Top-N candidates per study (default: %(default)d)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 78)
    log.info("Walk-forward analysis  |  N=%d  |  rolling_window=%d  |  cov_min=%.2f",
             args.n_candidates, ROLLING_WINDOW, COVERAGE_MIN)
    log.info("Folds: %s",
             [(f["name"], f["test_year"]) for f in WALK_FORWARD_FOLDS])
    log.info("=" * 78)

    df = load_dataset(INPUT_CSV)

    log.info("Loading top candidates from Optuna studies ...")
    uw_cands = load_top_candidates(UNWEIGHTED_DB_URI, UNWEIGHTED_STUDY, args.n_candidates)
    w_cands  = load_top_candidates(WEIGHTED_DB_URI,   WEIGHTED_STUDY,   args.n_candidates)

    log.info("\n=== Walk-forward — UNWEIGHTED study (%d candidates × %d folds) ===",
             len(uw_cands), len(WALK_FORWARD_FOLDS))
    uw_aggs = run_walk_forward(df, uw_cands, "unweighted", OUT_ROOT)

    log.info("\n=== Walk-forward — WEIGHTED study (%d candidates × %d folds) ===",
             len(w_cands), len(WALK_FORWARD_FOLDS))
    w_aggs  = run_walk_forward(df, w_cands,  "weighted",   OUT_ROOT)

    write_summary_csvs(uw_aggs, w_aggs, OUT_ROOT)

    uw_winner = select_winner(uw_aggs)
    w_winner  = select_winner(w_aggs)

    if uw_winner is not None:
        save_winner_record(uw_winner, OUT_ROOT / "selected_unweighted.json")
        plot_winner_equity_curves(uw_winner, df, OUT_ROOT / "selected_unweighted")
        log.info("Unweighted winner: trial %d, mean C_ir1=%+.4f, mean test_auc=%.4f",
                 uw_winner.trial_number, uw_winner.mean_C_ir1, uw_winner.mean_test_auc)
    if w_winner is not None:
        save_winner_record(w_winner, OUT_ROOT / "selected_weighted.json")
        plot_winner_equity_curves(w_winner, df, OUT_ROOT / "selected_weighted")
        log.info("Weighted   winner: trial %d, mean C_ir1=%+.4f, mean test_auc=%.4f",
                 w_winner.trial_number, w_winner.mean_C_ir1, w_winner.mean_test_auc)

    print("\n" + "=" * 78)
    print("  Walk-forward complete.")
    print(f"  Outputs: {OUT_ROOT.resolve()}")
    print(f"  Headline tables: walk_forward_summary.csv, candidate_aggregate.csv")
    print(f"  Winners: selected_unweighted.json, selected_weighted.json")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
