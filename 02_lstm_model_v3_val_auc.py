#!/usr/bin/env python3
# =============================================================================
# 02_lstm_model_v2.py
# =============================================================================
# PURPOSE:
#   Version 2 of the LSTM model script. Trains and evaluates the BTC daily
#   direction classifier with four structural upgrades over LSTM_model_1.02.py:
#
#   [V2-1] EXTENDED TRAINING WINDOW
#          Train: 2015-01-01 → 2023-12-31  (9 years, ~3,000 rows)
#          Val:   2024-01-01 → 2024-12-31  (1 full year)
#          Test:  2025-01-01 → 2025-12-31  (1 full year, locked until final run)
#          Requires processed_dataset_v2.csv (START_DATE="2014-11-01") so
#          training data genuinely starts at 2015-01-01 without burn-in loss.
#
#   [V2-2] SEQ_LEN = 20  (was 30 in v1)
#          20 trading days ≈ 4 calendar weeks (one macro decision cycle).
#          Shorter look-back reduces stale-signal risk for BTC's fast-moving
#          regime and marginally speeds up both training and tuning.
#
#   [V2-3] VARIANT C — DYNAMIC POSITION SIZING
#          Moves beyond binary signal × return to a continuous weighting
#          scheme that scales position size with prediction confidence.
#
#          HOLD zone  P ∈ [0.45, 0.55]  →  weight = 0.0  (flat / cash)
#          Long  P > 0.55  →  w = 0.25 + 0.75 × (P − 0.55) / (1 − 0.55)
#          Short P < 0.45  →  w = 0.25 + 0.75 × (0.45 − P) / 0.45
#
#          At the band edge (P=0.55/0.45), w = 0.25 (minimum exposure).
#          At maximum conviction (P=1.0/0.0),  w = 1.0 (full exposure).
#          Strategy return[t] = signal[t] × weight[t] × btc_log_return[t]
#
#   [V2-4] ADVANCED QUANTITATIVE METRICS + SPY BENCHMARK
#          plot_equity_curve() produces four equity curve PNGs per split:
#            • Variant A  (static 0.50, always ±1 — naive baseline)
#            • Variant B  (rolling 90d median, always ±1 — dynamic baseline)
#            • Variant C  (rolling 30th/70th pct — 3-state ±1/0)
#            • Variant D  (rolling 30th/70th pct — dynamic position sizing)
#          Each strategy is annotated with quant metrics (aRC, aSD, IR2).
#
# PIPELINE STEPS:
#   1. Load & Split  – Load CSV; filter by date boundaries.
#   2. Scale         – StandardScaler fitted ONLY on Train (no leakage).
#   3. Window        – tf.keras.utils.timeseries_dataset_from_array.
#   4. Build & Train – 2-layer LSTM: L2 + Dropout + Recurrent Dropout.
#   5. Evaluate A    – Static 0.50: Acc, Sens, Spec, AUC, Gini.
#   6. Evaluate B/C/D– Rolling thresholds seeded by prior split (no leakage).
#   8. Visualize     – Training history, ROC, confusion matrices.
#   9. Advanced Plots:
#        • Probability distribution (Train, Val, [Test])
#        • Equity curves with SPY + Variant C + Sharpe/Vol/MDD annotation
#        • Threshold sensitivity analysis (Val only)
#        • Permutation Feature Importance (Val only)
#
# OUTPUTS:
#   • training_history.png
#   • roc_curve_[val|test].png
#   • confusion_matrix_[val|test].png
#   • prob_distribution_[train|val|test].png
#   • equity_curve_[train|val|test].png   ← SPY B&H + Variant C + metrics
#   • threshold_sensitivity.png
#   • feature_importance_val.png
#
# REQUIREMENTS:
#   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import calendar
import logging
import os
import sys

# ── CPU determinism flags (must precede tensorflow import) ────────────────────


# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential          # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.regularizers import l2 as L2Reg     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# ── Force single-threaded TF execution for full CPU determinism ───────────────


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# ███  GLOBAL HYPERPARAMETERS  ███
# =============================================================================

# ── [V2-1] Date-based chronological splits ────────────────────────────────────
# Train: 9 full years.  Val: 1 full year (hyperparameter feedback).
# Test: 1 full year locked until the single final reported run.
TRAIN_START: str = "2015-01-01"
TRAIN_END:   str = "2021-12-31"   # reduced by 1 year → 8 years total
VAL_START:   str = "2022-01-01"   # extended to 2 years (2023–2024)
VAL_END:     str = "2022-12-31"
TEST_START:  str = "2023-01-01"
TEST_END:    str = "2023-12-31"

# Test set toggle — same semantics as v1.
# False during all tuning phases. True only for the final reported run.
EVALUATE_ON_TEST: bool = True

# ── [V2-2] Sequence length ────────────────────────────────────────────────────
SEQ_LEN: int = 30   # 30 trading days

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 64
EPOCHS:     int = 150

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_UNITS_1:  int   = 256
LSTM_UNITS_2:  int   = 64
DROPOUT_RATE:  float = 0.5034769871945717
RECURRENT_DROPOUT_RATE: float = 0.48976619202009397
L2_FACTOR:     float = 0.00015179813847411442
LEARNING_RATE:  float = 0.006638848742089275

# ── 3-state / dynamic-sizing confidence thresholds ───────────────────────────
UPPER_THRESHOLD: float = 0.52
LOWER_THRESHOLD: float = 0.42

# ── [V2-3] Variant C weight bounds ───────────────────────────────────────────
# Position weight at the band edge (minimum non-zero exposure).
# Position weight at maximum conviction (full exposure).
VC_MIN_WEIGHT: float = 0.25
VC_MAX_WEIGHT: float = 1

# ── [NEW] Threshold Calibration Settings ─────────────────────────────────────
CALIBRATION_OBJECTIVE: str = "sharpe"  # Zmień na "accuracy" jeśli wolisz optymalizować pod Win Rate
MIN_COVERAGE: float = 0.5           # Minimum 50% dni w których model musi handlować
Q_CANDIDATES: list[float] = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

# ── Feature pruning ─────────────────────────────────────────────────────────── 
# gold_log_return and nvda_log_return showed negative permutation importance
# (permuting them improves val AUC) → toxic features, removed before scaling.
# MUST match 03_hyperparameter_tuning_v2.py exactly for reproducibility.
PRUNED_FEATURES: frozenset = frozenset({"vix_level"})

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV:    str = "processed_dataset_v4_fixed.csv"
RUN_NAME:     str = "fold1_test_2023 ES15"
BASE_OUT_DIR: str = os.path.join("walk-forward/unweighted champion", RUN_NAME)

# ── Reproducibility seed ─────────────────────────────────────────────────────
SEED: int = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — DATA LOADING & DATE-BASED SPLITTING
# =============================================================================

def load_and_split(csv_path: str) -> tuple:
    """
    Load the processed CSV and partition it into strictly chronological
    subsets using exact calendar date boundaries.

    Warm-up buffer fix
    ──────────────────
    Each split includes SEQ_LEN rows BEFORE the target START_DATE so that the
    first sliding-window prediction lands exactly on START_DATE, not SEQ_LEN
    days after it.  The buffer rows are consumed by scale_features() and
    _make_dataset() and never appear in evaluation outputs.

    Chronological wall:
        TRAIN [2015–2023] → VAL [2024] → TEST [2025]

    Returns (train_df, val_df, test_df).  test_df is None if EVALUATE_ON_TEST=False.
    """
    log.info("Loading dataset from '%s' ...", csv_path)
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )

    def _slice_with_buffer(start_date, end_date):
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        if not mask.any():
            return None
        start_idx  = df[mask].index.min()
        end_idx    = df[mask].index.max()
        safe_start = max(0, start_idx - SEQ_LEN + 1)
        sliced = df.iloc[safe_start : end_idx + 1].copy().reset_index(drop=True)
        return sliced.set_index("Date")   # restore DatetimeIndex for downstream compat

    train_df = _slice_with_buffer(TRAIN_START, TRAIN_END)
    val_df   = _slice_with_buffer(VAL_START,   VAL_END)

    for name, split in [("Train", train_df), ("Val", val_df)]:
        log.info(
            "  %-5s → %5d rows  [%s → %s]  (incl. %d-day warm-up buffer)",
            name, len(split),
            split.index.min().date(), split.index.max().date(), SEQ_LEN,
        )

    test_df = None
    if EVALUATE_ON_TEST:
        test_df = _slice_with_buffer(TEST_START, TEST_END)
        log.info(
            "  %-5s → %5d rows  [%s → %s]  (incl. %d-day warm-up buffer)",
            "Test", len(test_df),
            test_df.index.min().date(), test_df.index.max().date(), SEQ_LEN,
        )
    else:
        log.info(
            "  Test  → LOCKED  (EVALUATE_ON_TEST=False). "
            "Set to True only for the final reported run.",
        )

    return train_df, val_df, test_df


# =============================================================================
# STEP 2 — FEATURE SCALING
# =============================================================================

def scale_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame | None,
) -> tuple:
    """
    Apply StandardScaler fitted strictly on training data (no leakage).

    PRUNED_FEATURES are excluded before scaler fitting so their statistics
    never contaminate the μ/σ estimates used to transform the model inputs.

    If USE_RAW_PRICES=True was used to generate the CSV, the column
    'btc_log_return' will be present as an auxiliary reference column.
    It is explicitly excluded from model features here via PRUNED_FEATURES
    (ensure "btc_log_return" is in PRUNED_FEATURES when using raw-price mode).

    Returns
    -------
    X_train, X_val, X_test   : np.ndarray (or None for X_test)
    y_train, y_val, y_test   : np.ndarray (or None for y_test)
    scaler                   : fitted StandardScaler
    feature_cols             : list[str]
    """
    feature_cols = [
        c for c in train_df.columns
        if c != "target" and c not in PRUNED_FEATURES
    ]
    if PRUNED_FEATURES:
        log.info(
            "Pruned features (excluded from model): %s",
            sorted(PRUNED_FEATURES),
        )
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)

    y_train = train_df["target"].values.astype(np.float32)
    y_val   = val_df["target"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    log.info(
        "Scaler fitted on training set. Feature means (rounded): %s",
        dict(zip(feature_cols, np.round(scaler.mean_, 4))),
    )

    if test_df is not None:
        X_test_raw = test_df[feature_cols].values.astype(np.float32)
        y_test     = test_df["target"].values.astype(np.float32)
        X_test     = scaler.transform(X_test_raw)
    else:
        X_test = None
        y_test = None

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


# =============================================================================
# STEP 3 — WINDOWED DATASET CONSTRUCTION
# =============================================================================

def make_tf_dataset(
    X:              np.ndarray,
    y:              np.ndarray,
    seq_len:        int,
    batch_size:     int,
    shuffle:        bool = False,
    sample_weights: np.ndarray | None = None,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows.

    Window ↔ Target alignment (no look-ahead bias):
        Window i = features[i : i+L]      (L = seq_len)
        Target i = y[i + L - 1]           (direction of day i + L)

    Returns (dataset, y_aligned) where y_aligned = y[seq_len-1:].

    sample_weights
    ──────────────
    When provided, the dataset yields (x, y, w) triples instead of (x, y)
    pairs. Keras model.fit() uses the third element as per-sample loss weights
    automatically. Pass None for val/test datasets used only for prediction
    or metric computation (backward-compatible path uses timeseries_dataset
    _from_array which is more memory-efficient for large datasets).

    sample_weights must have length ≥ N − seq_len + 1 (n_windows).
    Alignment: sample_weights[i] is the weight for window i, which predicts
    the direction of row (i + seq_len − 1) in the raw split DataFrame.
    See _compute_sample_weights() for the construction of this array.
    """
    N         = len(X)
    n_windows = N - seq_len + 1
    y_aligned = y[seq_len - 1:].astype(np.float32)

    if sample_weights is not None:
        # Build (x_window, y, w) dataset via pre-stacked numpy windows.
        # Memory: ~3000 × 30 × 12 × 4 bytes ≈ 4 MB — negligible.
        X_windows = np.stack(
            [X[i : i + seq_len] for i in range(n_windows)]
        ).astype(np.float32)
        w = sample_weights[:n_windows].astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned, w))
        if shuffle:
            ds = ds.shuffle(buffer_size=n_windows, seed=SEED)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, y_aligned

    # ── Default path (no sample weights) — memory-efficient strided view ──────
    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[seq_len - 1:]
    targets_aligned[n_windows:] = 0.0

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X.astype(np.float32),
        targets=targets_aligned,
        sequence_length=seq_len,
        sequence_stride=1,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )
    return dataset, y_aligned


def _compute_sample_weights(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Compute per-window sample weights from the absolute BTC log return of
    the TARGET day (the day each window is predicting).

    Alignment:
        Window i targets day index (i + seq_len − 1) in the split DataFrame.
        weight[i] = abs(btc_log_return[i + seq_len − 1])
        ⟹ weight array  = abs(btc_log_return)[seq_len − 1:]
        ⟹ same length as y_aligned = y[seq_len − 1:]

    Normalisation:
        Dividing by the mean keeps mean(weight) = 1.0, so the total gradient
        magnitude is unchanged relative to unweighted training while
        concentrating learning signal on high-volatility days.

    Returns np.ndarray of shape (N − seq_len + 1,), dtype float32.
    """
    raw      = np.abs(df["btc_log_return"].values[seq_len - 1:]).astype(np.float32)
    mean_val = float(raw.mean())
    if mean_val < 1e-8:
        return np.ones(len(raw), dtype=np.float32)
    return raw / mean_val


# =============================================================================
# STEP 4 — MODEL ARCHITECTURE
# =============================================================================

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Two-layer stacked LSTM with three anti-overfitting defences:
    inter-layer Dropout, Recurrent Dropout, EarlyStopping.
    (Weight decay will be re-added once the Optuna study selects an optimizer.)
    """
    model = Sequential(
        [
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                input_shape=(seq_len, n_features),
                name="lstm_layer_1",
            ),
            Dropout(DROPOUT_RATE, name="dropout_1"),
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                name="lstm_layer_2",
            ),
            Dropout(DROPOUT_RATE, name="dropout_2"),
            Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="btc_direction_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()
    return model


# =============================================================================
# STEP 5 — TRAINING
# =============================================================================

def train_model(
    model:    tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    log_dir:  str | None = None,
) -> tf.keras.callbacks.History:
    """
    Train with EarlyStopping(patience=30) and ReduceLROnPlateau(patience=5).

    patience=30 allows the model to survive LR reductions without stopping
    prematurely.  ReduceLROnPlateau fires first (patience=5), halving the LR
    several times before EarlyStopping terminates training.

    If log_dir is provided, a CSVLogger saves epoch-by-epoch loss/AUC to
    {log_dir}/training_log.csv for full auditability.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if log_dir is not None:
        csv_path = os.path.join(log_dir, "training_log.csv")
        callbacks.append(
            tf.keras.callbacks.CSVLogger(csv_path, append=False)
        )
        log.info("CSVLogger → %s", csv_path)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch = int(np.argmax(history.history["val_auc"])) + 1
    log.info(
        "Training complete. Best val_auc = %.5f at epoch %d / %d.",
        max(history.history["val_auc"]),
        best_epoch,
        len(history.history["val_auc"]),
    )
    return history


# =============================================================================
# STEP 5B — ROLLING DYNAMIC THRESHOLD HELPER
# =============================================================================

def apply_rolling_thresholds(
    prob_prior:  np.ndarray,
    prob_target: np.ndarray,
    lookback:    int = 90,
    lower_q:     int = 30,
    upper_q:     int = 70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-day rolling percentile thresholds for prob_target with zero leakage.

    Seeds the window with prob_prior[-lookback:] so day-0 of prob_target
    already has a full 90-day window (no cold-start NaNs). .shift(1) ensures
    the threshold for day T is computed from data strictly before T.

    Returns
    -------
    (lower_bound, rolling_median, upper_bound) each of shape (len(prob_target),)
    """
    seed     = prob_prior[-lookback:]
    combined = pd.Series(np.concatenate([seed, prob_target]))
    roll     = combined.rolling(window=lookback)
    lower_bound    = roll.quantile(lower_q / 100.0).shift(1).values[lookback:]
    rolling_median = roll.median().shift(1).values[lookback:]
    upper_bound    = roll.quantile(upper_q / 100.0).shift(1).values[lookback:]
    return lower_bound, rolling_median, upper_bound


# =============================================================================
# STEP 6A — VARIANT A: STATIC NAIVE BASELINE (±1, fixed 0.50)
# =============================================================================

def evaluate_standard(
    y_true:        np.ndarray,
    y_prob:        np.ndarray,
    split_name:    str,
    df:            pd.DataFrame | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Standard 0.5-threshold evaluation: Acc, Sensitivity, Specificity, AUC, Gini.

    If df is provided, also computes Variant A financial performance metrics
    (Sharpe, Volatility, MaxDD) using the same N−SEQ_LEN alignment as
    evaluate_variant_d.  Strategy: signal = +1 / −1 on every day (always in market).
    """
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred, sample_weight=sample_weight).ravel()

    accuracy    = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc         = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    gini        = 2.0 * auc - 1.0

    print(f"\n{'═' * 58}")
    print(f"  Variant A — Static Naive Baseline  [{split_name} Set]")
    print(f"  Threshold: fixed 0.50  |  Signal: ±1 (always in market)")
    print(f"{'─' * 58}")
    print(f"  Accuracy    :  {accuracy:.4f}")
    print(f"  Sensitivity :  {sensitivity:.4f}   (True Positive Rate)")
    print(f"  Specificity :  {specificity:.4f}   (True Negative Rate)")
    print(f"  ROC AUC     :  {auc:.4f}")
    print(f"  Gini Coeff. :  {gini:.4f}   (= 2 × AUC − 1)")
    print(f"  Confusion   :  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    if df is not None:
        n_tradeable  = len(df) - SEQ_LEN
        prob_aligned = y_prob[:n_tradeable]
        btc_ret      = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
        signal_a     = np.where(prob_aligned >= 0.5, 1.0, -1.0)
        daily_ret_a  = signal_a * btc_ret
        arc, asd, md, mld, ir1, ir2, sortino, calmar, profit_factor = _compute_quant_metrics(daily_ret_a)
        print(f"{'─' * 58}")
        print(f"  [Financial Performance — Variant A (static 0.50, always ±1)]")
        print(f"  aRC  (Ann. Ret Compound):  {arc:>+8.2f}%")
        print(f"  aSD  (Ann. Std Dev)     :  {asd:>8.4f}")
        print(f"  MD   (Max Drawdown)     :  {md:>+8.2f}%")
        print(f"  MLD  (Longest DD, yrs)  :  {mld:>8.2f}")
        print(f"  IR1  (aRC / aSD)        :  {ir1:>+8.3f}")
        print(f"  IR2  (IR1 × |aRC|)      :  {ir2:>+8.3f}")
        print(f"  Sortino Ratio           :  {sortino:>+8.3f}")
        print(f"  Calmar Ratio            :  {calmar:>+8.3f}")
        print(f"  Profit Factor           :  {profit_factor:>8.3f}")
    print(f"{'═' * 58}\n")

    return dict(accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, auc=auc, gini=gini)


# =============================================================================
# STEP 6B — VARIANT B: DYNAMIC MEDIAN BASELINE (±1, rolling median)
# =============================================================================

def evaluate_rolling_median(
    y_true:         np.ndarray,
    y_prob:         np.ndarray,
    rolling_median: np.ndarray,
    split_name:     str,
    df:             pd.DataFrame | None = None,
    sample_weight:  np.ndarray | None = None,
) -> dict:
    """Variant B — Dynamic Median Baseline.

    Signal = +1 if prob >= 90-day rolling median, else -1. Always in market.
    Uses the rolling median as an adaptive decision boundary instead of the
    fixed 0.50 — this is a stronger baseline than Variant A because it adjusts
    to the current distribution of model confidence.
    """
    y_pred = (y_prob >= rolling_median).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true.astype(int), y_pred, sample_weight=sample_weight,
    ).ravel()
    accuracy    = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc         = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    gini        = 2.0 * auc - 1.0

    print(f"\n{'═' * 58}")
    print(f"  Variant B — Dynamic Median Baseline  [{split_name} Set]")
    print(f"  Threshold: 90-day rolling median  |  Signal: ±1 (always in market)")
    print(f"{'─' * 58}")
    print(f"  Accuracy    :  {accuracy:.4f}")
    print(f"  Sensitivity :  {sensitivity:.4f}   (True Positive Rate)")
    print(f"  Specificity :  {specificity:.4f}   (True Negative Rate)")
    print(f"  ROC AUC     :  {auc:.4f}")
    print(f"  Gini Coeff. :  {gini:.4f}   (= 2 × AUC − 1)")
    print(f"  Confusion   :  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    if df is not None:
        n_tradeable  = len(df) - SEQ_LEN
        prob_aligned = y_prob[:n_tradeable]
        med          = rolling_median[:n_tradeable]
        btc_ret      = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
        signal_b     = np.where(prob_aligned >= med, 1.0, -1.0)
        daily_ret_b  = signal_b * btc_ret
        arc, asd, md, mld, ir1, ir2, sortino, calmar, profit_factor = _compute_quant_metrics(daily_ret_b)
        print(f"{'─' * 58}")
        print(f"  [Financial Performance — Variant B (rolling median, always ±1)]")
        print(f"  aRC  (Ann. Ret Compound):  {arc:>+8.2f}%")
        print(f"  aSD  (Ann. Std Dev)     :  {asd:>8.4f}")
        print(f"  MD   (Max Drawdown)     :  {md:>+8.2f}%")
        print(f"  MLD  (Longest DD, yrs)  :  {mld:>8.2f}")
        print(f"  IR1  (aRC / aSD)        :  {ir1:>+8.3f}")
        print(f"  IR2  (IR1 × |aRC|)      :  {ir2:>+8.3f}")
        print(f"  Sortino Ratio           :  {sortino:>+8.3f}")
        print(f"  Calmar Ratio            :  {calmar:>+8.3f}")
        print(f"  Profit Factor           :  {profit_factor:>8.3f}")
    print(f"{'═' * 58}\n")

    return dict(accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, auc=auc, gini=gini)


# =============================================================================
# STEP 6C — VARIANT C: 3-STATE ROLLING QUANTILE FILTER
# =============================================================================

def evaluate_3state(
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    split_name:  str,
    df:          pd.DataFrame | None = None,
    lower_bound: np.ndarray | None = None,
    upper_bound: np.ndarray | None = None,
    q_star:      float = 30.0,
) -> dict:
    """Variant C — 3-State Rolling Quantile Filter.

    Long if P > upper_bound (90th-day 70th pct), Short if P < lower_bound
    (90th-day 30th pct), else Cash. Metrics on traded days only.
    When bounds are None, falls back to static UPPER/LOWER_THRESHOLD globals.
    """
    _ub = upper_bound if upper_bound is not None else np.full(len(y_prob), UPPER_THRESHOLD)
    _lb = lower_bound if lower_bound is not None else np.full(len(y_prob), LOWER_THRESHOLD)
    trade_mask = (y_prob > _ub) | (y_prob < _lb)
    n_traded   = int(trade_mask.sum())
    n_total    = len(y_prob)
    coverage   = n_traded / n_total if n_total > 0 else 0.0

    if n_traded == 0:
        log.warning(
            "[%s] Zero days exceed threshold bands. Consider widening the gap.",
            split_name,
        )
        return dict(coverage=0.0, conditional_win_rate=float("nan"), n_traded=0)

    cond_win_rate = accuracy_score(
        y_true[trade_mask].astype(int),
        (y_prob[trade_mask] >= 0.5).astype(int),
    )

    thresh_label = (
        f"Dynamic 90d {q_star:.1f}th/{100.0 - q_star:.1f}th pct" if upper_bound is not None
        else f"UPPER={UPPER_THRESHOLD:.2f}  |  LOWER={LOWER_THRESHOLD:.2f}"
    )
    print(f"{'═' * 58}")
    print(f"  Variant C — 3-State Rolling Quantile  [{split_name} Set]")
    print(f"  Thresholds:  {thresh_label}")
    print(f"{'─' * 58}")
    print(f"  Total days    :  {n_total}")
    print(f"  Traded days   :  {n_traded}  ({coverage * 100:.1f}% coverage)")
    print(f"  Hold days     :  {n_total - n_traded}  ({(1 - coverage) * 100:.1f}%)")
    print(f"  Cond. Win Rate:  {cond_win_rate:.4f}  (accuracy on traded days)")

    if df is not None:
        n_tradeable  = len(df) - SEQ_LEN
        prob_aligned = y_prob[:n_tradeable]
        btc_ret      = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
        ub = upper_bound[:n_tradeable] if upper_bound is not None else np.full(n_tradeable, UPPER_THRESHOLD)
        lb = lower_bound[:n_tradeable] if lower_bound is not None else np.full(n_tradeable, LOWER_THRESHOLD)
        signal_c    = np.where(prob_aligned > ub, 1.0, np.where(prob_aligned < lb, -1.0, 0.0))
        daily_ret_c = signal_c * btc_ret
        arc, asd, md, mld, ir1, ir2, sortino, calmar, profit_factor = _compute_quant_metrics(daily_ret_c)
        print(f"{'─' * 58}")
        print(f"  [Financial Performance — Variant C (cash days = zero return)]")
        print(f"  aRC  (Ann. Ret Compound):  {arc:>+8.2f}%")
        print(f"  aSD  (Ann. Std Dev)     :  {asd:>8.4f}")
        print(f"  MD   (Max Drawdown)     :  {md:>+8.2f}%")
        print(f"  MLD  (Longest DD, yrs)  :  {mld:>8.2f}")
        print(f"  IR1  (aRC / aSD)        :  {ir1:>+8.3f}")
        print(f"  IR2  (IR1 × |aRC|)      :  {ir2:>+8.3f}")
        print(f"  Sortino Ratio           :  {sortino:>+8.3f}")
        print(f"  Calmar Ratio            :  {calmar:>+8.3f}")
        print(f"  Profit Factor           :  {profit_factor:>8.3f}")
    print(f"{'═' * 58}\n")

    return dict(coverage=coverage, conditional_win_rate=cond_win_rate, n_traded=n_traded)


# =============================================================================
# STEP 6D — VARIANT D: DYNAMIC POSITION SIZING
# =============================================================================

def _compute_variant_d_weights(
    probs:        np.ndarray,
    lower_bound:  np.ndarray | None = None,
    upper_bound:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Variant D signals and continuous position weights.

    Uses the same 30th/70th rolling percentile bands as Variant C, but instead
    of a flat ±1 signal, position size scales linearly from VC_MIN_WEIGHT at
    the band edge to VC_MAX_WEIGHT at maximum conviction (P=1.0 or P=0.0).

    When lower_bound/upper_bound are None, falls back to static globals.
    Returns (signals, weights): both np.ndarray of shape (N,).
    """
    signals = np.zeros(len(probs), dtype=np.float32)
    weights = np.zeros(len(probs), dtype=np.float32)

    ub = upper_bound if upper_bound is not None else np.full(len(probs), UPPER_THRESHOLD, dtype=np.float32)
    lb = lower_bound if lower_bound is not None else np.full(len(probs), LOWER_THRESHOLD, dtype=np.float32)

    long_mask  = probs > ub
    short_mask = probs < lb

    signals[long_mask]  = 1.0
    signals[short_mask] = -1.0

    weight_range = VC_MAX_WEIGHT - VC_MIN_WEIGHT

    weights[long_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (probs[long_mask] - ub[long_mask])
        / (1.0 - ub[long_mask])
    )

    weights[short_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (lb[short_mask] - probs[short_mask])
        / lb[short_mask]
    )

    return signals, weights


def evaluate_variant_d(
    df:           pd.DataFrame,
    prob:         np.ndarray,
    split_name:   str,
    lower_bound:  np.ndarray | None = None,
    upper_bound:  np.ndarray | None = None,
    q_star:       float = 30.0,
) -> dict:
    """Variant D — Dynamic Position Sizing with rolling 30th/70th pct bands.

    Same HOLD zone as Variant C; instead of a flat ±1 signal, position weight
    scales linearly from VC_MIN_WEIGHT at the band edge to VC_MAX_WEIGHT at
    maximum model conviction.
    """
    n_tradeable  = len(df) - SEQ_LEN
    prob_aligned = prob[:n_tradeable]
    btc_returns  = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]

    ub = upper_bound[:n_tradeable] if upper_bound is not None else None
    lb = lower_bound[:n_tradeable] if lower_bound is not None else None
    signals, weights = _compute_variant_d_weights(prob_aligned, lower_bound=lb, upper_bound=ub)
    daily_ret = signals * weights * btc_returns

    n_long  = int((signals > 0).sum())
    n_short = int((signals < 0).sum())
    n_hold  = int((signals == 0).sum())

    arc, asd, md, mld, ir1, ir2, sortino, calmar, profit_factor = _compute_quant_metrics(daily_ret)

    thresh_label = (
        f"Dynamic 90d {q_star:.1f}th/{100.0 - q_star:.1f}th pct" if upper_bound is not None
        else f"[{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]"
    )
    print(f"{'═' * 58}")
    print(f"  Variant D — Dynamic Position Sizing  [{split_name} Set]")
    print(f"  HOLD zone: {thresh_label},  weight ∈ [{VC_MIN_WEIGHT:.2f}, {VC_MAX_WEIGHT:.2f}]")
    print(f"{'─' * 58}")
    print(f"  Long days   :  {n_long}  |  Short days: {n_short}  |  Hold: {n_hold}")
    print(f"  aRC  (Ann. Ret Compound):  {arc:>+8.2f}%")
    print(f"  aSD  (Ann. Std Dev)     :  {asd:>8.4f}")
    print(f"  MD   (Max Drawdown)     :  {md:>+8.2f}%")
    print(f"  MLD  (Longest DD, yrs)  :  {mld:>8.2f}")
    print(f"  IR1  (aRC / aSD)        :  {ir1:>+8.3f}")
    print(f"  IR2  (IR1 × |aRC|)      :  {ir2:>+8.3f}")
    print(f"  Sortino Ratio           :  {sortino:>+8.3f}")
    print(f"  Calmar Ratio            :  {calmar:>+8.3f}")
    print(f"  Profit Factor           :  {profit_factor:>8.3f}")
    print(f"{'═' * 58}\n")

    return {
        "arc": arc, "asd": asd, "md": md, "mld": mld, "ir1": ir1, "ir2": ir2,
        "sortino": sortino, "calmar": calmar, "profit_factor": profit_factor,
        "total_log_ret": float(np.sum(daily_ret)),
        "n_long": n_long, "n_short": n_short, "n_hold": n_hold,
    }


# =============================================================================
# ADVANCED QUANT METRICS HELPER
# =============================================================================

def export_predictions_to_csv(
    df:             pd.DataFrame,
    prob:           np.ndarray,
    seq_len:        int,
    exp_dir:        str,
    rolling_median: np.ndarray | None = None,
    lower_bound:    np.ndarray | None = None,
    upper_bound:    np.ndarray | None = None,
) -> None:
    """Export test-set predictions and all 4 variant signals to CSV.

    Saved to: exp_dir/test_predictions.csv
    """
    n_tradeable = len(df) - seq_len
    prob_al     = prob[:n_tradeable]
    dates       = df.index[seq_len : seq_len + n_tradeable]
    actual_ret  = df["btc_log_return"].values[seq_len : seq_len + n_tradeable]

    # Variant A — static 0.50, always ±1
    sig_a = np.where(prob_al >= 0.5, 1.0, -1.0)
    ret_a = sig_a * actual_ret

    # Variant B — rolling median, always ±1
    med   = rolling_median[:n_tradeable] if rolling_median is not None else np.full(n_tradeable, 0.5)
    sig_b = np.where(prob_al >= med, 1.0, -1.0)
    ret_b = sig_b * actual_ret

    # Variant C — rolling 30th/70th, trinary ±1/0
    ub    = upper_bound[:n_tradeable] if upper_bound is not None else np.full(n_tradeable, UPPER_THRESHOLD)
    lb    = lower_bound[:n_tradeable] if lower_bound is not None else np.full(n_tradeable, LOWER_THRESHOLD)
    sig_c = np.where(prob_al > ub, 1.0, np.where(prob_al < lb, -1.0, 0.0))
    ret_c = sig_c * actual_ret

    # Variant D — dynamic position sizing with same bands
    sig_d, wt_d = _compute_variant_d_weights(
        prob_al,
        lower_bound=lb if lower_bound is not None else None,
        upper_bound=ub if upper_bound is not None else None,
    )
    ret_d = sig_d * wt_d * actual_ret

    out_df = pd.DataFrame({
        "Date":                           dates,
        "Actual_Log_Return":              actual_ret,
        "Predicted_Prob":                 prob_al,
        "Rolling_Median":                 med,
        "Lower_Bound_30th":               lb,
        "Upper_Bound_70th":               ub,
        "Signal_Variant_A_Static050":     sig_a,
        "Return_Variant_A":               ret_a,
        "Equity_Variant_A":               np.exp(np.cumsum(ret_a)),
        "Signal_Variant_B_RollingMedian": sig_b,
        "Return_Variant_B":               ret_b,
        "Equity_Variant_B":               np.exp(np.cumsum(ret_b)),
        "Signal_Variant_C_3StateQuantile":sig_c,
        "Return_Variant_C":               ret_c,
        "Equity_Variant_C":               np.exp(np.cumsum(ret_c)),
        "Signal_Variant_D_DynSizing":     sig_d,
        "Weight_Variant_D":               wt_d,
        "Return_Variant_D":               ret_d,
        "Equity_Variant_D":               np.exp(np.cumsum(ret_d)),
        "Equity_BTC":                     np.exp(np.cumsum(actual_ret)),
    }).set_index("Date")

    csv_path = os.path.join(exp_dir, "test_predictions.csv")
    out_df.to_csv(csv_path)
    log.info("Test predictions exported → %s  (%d rows)", csv_path, len(out_df))


def export_sanity_check_csv(
    df:      pd.DataFrame,
    prob:    np.ndarray,
    seq_len: int,
    exp_dir: str,
    n_days:  int = 30,
) -> None:
    """
    Emit an auditable n_days-row CSV proving that Model_Probability[row_i]
    is aligned with the FUTURE-dated bar on Date[row_i] — i.e. the model is
    predicting forward, not recalling the past.

    Alignment (identical to evaluate_variant_d / export_predictions_to_csv):
        prob[k]                   ⇌  row (k + seq_len) of df
        Date                      = df.index[k + seq_len]
        Actual_Close_Price        = df['btc_close'][k + seq_len]
        Actual_Direction          = 1 if df['btc_log_return'][k + seq_len] > 0 else 0
        Model_Predicted_Direction = 1 if prob[k] >= 0.5 else 0

    Output: exp_dir/sanity_check_test_predictions.csv  (last n_days rows).
    """
    n_tradeable = len(df) - seq_len
    if n_tradeable <= 0:
        log.warning(
            "[sanity-check] insufficient rows (%d ≤ SEQ_LEN); skipping.",
            len(df),
        )
        return

    prob_al = prob[:n_tradeable]
    dates   = df.index[seq_len : seq_len + n_tradeable]
    closes  = df["btc_close"].values[seq_len : seq_len + n_tradeable]
    log_ret = df["btc_log_return"].values[seq_len : seq_len + n_tradeable]

    actual_dir = (log_ret > 0.0).astype(int)
    pred_dir   = (prob_al >= 0.5).astype(int)

    out = pd.DataFrame({
        "Date":                      pd.to_datetime(dates).date,
        "Actual_Close_Price":        closes,
        "Actual_Direction":          actual_dir,
        "Model_Probability":         prob_al,
        "Model_Predicted_Direction": pred_dir,
    }).tail(n_days)

    csv_path = os.path.join(exp_dir, "sanity_check_test_predictions.csv")
    out.to_csv(csv_path, index=False)
    log.info("Sanity-check CSV → %s  (%d rows)", csv_path, len(out))


def _compute_quant_metrics(
    strat_returns: np.ndarray,
    trading_days:  int = 365,
) -> tuple:
    """
    R-metric translation.

    Returns 9-tuple: (arc, asd, md, mld, ir1, ir2, sortino, calmar, profit_factor)
        arc          : Annualised Return Compound (%)  — R: getARC
        asd          : Annualised Std Deviation        — R: getASD
        md           : Maximum Drawdown (%)            — R: getMD
        mld          : Longest Drawdown Duration (yrs) — R: getLD2
        ir1          : IR1 = aRC / aSD                 — R: getIR
        ir2          : IR2 = IR1 × |aRC|               — R: getIR2
        sortino      : (mean_daily / downside_dev) × √trading_days
        calmar       : (mean_daily × trading_days) / |MD_decimal|
        profit_factor: gross_wins / gross_losses
    """
    strat_returns = np.asarray(strat_returns, dtype=np.float64)
    equity = np.exp(np.cumsum(strat_returns))
    n_int  = len(equity)

    # aRC — Annualised Return Compound (geometric CAGR)
    arc = float(((equity[-1] / 1.0) ** (trading_days / n_int) - 1.0) * 100.0) if n_int > 0 else 0.0

    # aSD — Annualised Standard Deviation
    asd = float(np.std(strat_returns, ddof=1) * np.sqrt(trading_days)) if n_int > 1 else 0.0

    # MD — Maximum Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns   = 1.0 - equity / running_max
    md = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    # MLD — Longest Drawdown Duration (years)
    ld       = np.zeros(n_int, dtype=np.float64)
    curr_max = equity[0] if n_int > 0 else 1.0
    for i in range(1, n_int):
        if equity[i] >= curr_max:
            curr_max = equity[i]
            ld[i]    = 0.0
        else:
            ld[i] = ld[i - 1] + 1.0
    mld = float(np.max(ld) / trading_days) if n_int > 0 else 0.0

    # IR1 and IR2
    ir1 = float((arc / 100.0) / asd) if asd > 1e-12 else 0.0
    ir2 = float(ir1 * (arc / 100.0) * np.sign(arc / 100.0))

    # Sortino Ratio: (mean_daily / downside_dev) × √trading_days
    neg_ret = strat_returns[strat_returns < 0]
    if len(neg_ret) > 1:
        sortino = float((np.mean(strat_returns) / np.std(neg_ret, ddof=1)) * np.sqrt(trading_days))
    else:
        sortino = float("inf")

    # Calmar Ratio: (mean_daily × trading_days) / |MD_decimal|
    if md > 1e-10:
        calmar = float((np.mean(strat_returns) * trading_days) / (md / 100.0))
    else:
        calmar = float("inf")

    # Profit Factor: gross_wins / gross_losses
    gross_win  = float(np.sum(strat_returns[strat_returns > 0]))
    gross_loss = float(abs(np.sum(strat_returns[strat_returns < 0])))
    profit_factor = gross_win / gross_loss if gross_loss > 1e-10 else float("inf")

    return float(arc), float(asd), float(md), float(mld), float(ir1), float(ir2), sortino, calmar, profit_factor


# =============================================================================
# STEP 7 — DIAGNOSTIC VISUALIZATIONS
# =============================================================================

def export_simple_excel(
    df:             pd.DataFrame,
    prob:           np.ndarray,
    seq_len:        int,
    exp_dir:        str,
    split_name:     str,
    rolling_median: np.ndarray | None = None,
) -> None:
    """Export a clean per-split Excel for thesis supervisor review.

    Core columns: Date | BTC_Close | Probability_UP_Tomorrow | Signal_A_Static050
    When rolling_median is provided (Val/Test): adds Signal_B_RollingMedian and Rolling_Median.
    """
    n_tradeable = len(df) - seq_len
    prob_al  = prob[:n_tradeable]
    dates    = df.index[seq_len : seq_len + n_tradeable]
    closes   = df["btc_close"].values[seq_len : seq_len + n_tradeable]
    sig_a    = np.where(prob_al >= 0.5, 1, -1)

    out_dict = {
        "Date":                    pd.to_datetime(dates).date,
        "BTC_Close":               closes,
        "Probability_UP_Tomorrow": prob_al,
        "Signal_A_Static050":      sig_a,
    }

    if rolling_median is not None:
        med = rolling_median[:n_tradeable]
        out_dict["Rolling_Median"]         = med
        out_dict["Signal_B_RollingMedian"] = np.where(prob_al >= med, 1, -1)

    out_df   = pd.DataFrame(out_dict)
    out_path = os.path.join(exp_dir, f"Next_day_prediction_export_{split_name.lower()}.xlsx")
    out_df.to_excel(out_path, index=False)
    log.info("Next day prediction Excel export saved → %s", out_path)


def plot_training_history(history: tf.keras.callbacks.History, out_dir: str = ".") -> None:
    """Save 3-panel BCE loss + accuracy + AUC figure for Train/Val splits.

    The AUC panel includes:
    - A horizontal dashed line at 0.5 (random-chance baseline)
    - A vertical red dashed line at the best epoch (argmax val_auc) to make
      the epoch selection criterion explicit and auditable.
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"LSTM Training History — BTC Daily Direction Classifier\n"
        f"Architecture: {LSTM_UNITS_1}→{LSTM_UNITS_2} units | "
        f"L2={L2_FACTOR} | Dropout={DROPOUT_RATE} | "
        f"RecDrop={RECURRENT_DROPOUT_RATE} | SeqLen={SEQ_LEN}",
        fontsize=11,
    )

    # ── Panel 1: Loss ──────────────────────────────────────────────────────────
    axes[0].plot(epochs_range, history.history["loss"],
                 lw=2, color="steelblue", label="Train Loss")
    axes[0].plot(epochs_range, history.history["val_loss"],
                 lw=2, color="tomato", ls="--", label="Validation Loss")
    best_loss_epoch = int(np.argmin(history.history["val_loss"])) + 1
    axes[0].axvline(best_loss_epoch, color="red", ls=":", lw=1.2,
                    label=f"Best epoch: {best_loss_epoch}")
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Panel 2: Accuracy ─────────────────────────────────────────────────────
    axes[1].plot(epochs_range, history.history["accuracy"],
                 lw=2, color="steelblue", label="Train Accuracy")
    axes[1].plot(epochs_range, history.history["val_accuracy"],
                 lw=2, color="tomato", ls="--", label="Validation Accuracy")
    axes[1].set_title("Classification Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ── Panel 3: AUC ──────────────────────────────────────────────────────────
    if "auc" in history.history and "val_auc" in history.history:
        axes[2].plot(epochs_range, history.history["auc"],
                     lw=2, color="steelblue", label="Train AUC")
        axes[2].plot(epochs_range, history.history["val_auc"],
                     lw=2, color="tomato", ls="--", label="Validation AUC")
        axes[2].axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7,
                        label="Random baseline (0.5)")
        best_auc_epoch = int(np.argmax(history.history["val_auc"])) + 1
        axes[2].axvline(best_auc_epoch, color="red", ls="--", lw=1.5,
                        label=f"Best val AUC: epoch {best_auc_epoch}")
        axes[2].set_title("ROC AUC", fontsize=12)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, "training_history.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("Training history saved → %s", fname)


def plot_roc_curve(
    y_true:        np.ndarray,
    y_prob:        np.ndarray,
    auc_score:     float,
    split_name:    str = "Val",
    out_dir:       str = ".",
    sample_weight: np.ndarray | None = None,
) -> None:
    """ROC curve with AUC and Gini annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weight)
    gini = 2.0 * auc_score - 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="steelblue",
            label=f"LSTM  (AUC = {auc_score:.4f}  |  Gini = {gini:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier  (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
    ax.set_title(f"ROC Curve — {split_name} Set", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"roc_curve_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("ROC curve saved → %s", fname)




# =============================================================================
# STEP 8 — ADVANCED QUANTITATIVE FINANCE VISUALIZATIONS
# =============================================================================

def plot_prob_distribution(y_prob: np.ndarray, split_name: str, out_dir: str = ".") -> None:
    """Clean Probability distribution KDE without cluttered zone annotations."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 1. Czysty histogram i linia KDE dla rozkładu prawdopodobieństwa
    sns.histplot(
        y_prob, bins=40, kde=True, stat="density",
        color="steelblue", alpha=0.55, edgecolor="white", linewidth=0.4,
        label=f"LSTM Output Distribution — {split_name} Set",
        ax=ax, zorder=1,
    )
    
    # 2. Zostawiamy tylko linię 0.50 jako punkt odniesienia (środek skali)
    ax.axvline(0.50, color="black", lw=1.8, ls=":", zorder=2,
               label="Decision Boundary = 0.50  (Variant A)")

    # 3. Formatowanie osi i tytułu (usunięto tekst o Noise Zone)
    ax.set_xlabel("Predicted Probability  P(BTC closes higher tomorrow)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Probability Distribution of LSTM Output v2 — {split_name} Set",
        fontsize=13, pad=14,
    )
    
    # Oś X od 0 do 1, by pokazać jak bardzo skompresowany jest rozkład
    ax.set_xlim(0.0, 1.0)
    
    # Przeniosłem legendę w prawy górny róg, żeby nie zasłaniała rozkładu (który jest zazwyczaj po lewej)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.30)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, f"prob_distribution_v2_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Clean probability distribution saved → %s", fname)


def _plot_single_equity_curve(
    variant_label:     str,
    table_label:       str,
    variant_daily_ret: np.ndarray,
    btc_ret:           np.ndarray,
    spy_ret:           np.ndarray | None,
    has_spy:           bool,
    dates,
    split_name:        str,
    variant_color:     str,
    fname:             str,
) -> None:
    """
    Internal helper: save one equity-curve PNG for a single strategy variant.

    Legend format: "<Label>  (Sharpe: X.XX)  →  +Y.Z%"

    Annotation table (monospace box, bottom-left):
        Strategy          Tot Ret  Sharpe     Vol     MDD
        ─────────────────────────────────────────────────
        BTC B&H           +NNN.N%  +N.NNN   N.NNN   NN.N%
        SPY B&H (opt.)    +NNN.N%  +N.NNN   N.NNN   NN.N%
        <Variant>         +NNN.N%  +N.NNN   N.NNN   NN.N%
    """
    mult_variant = np.exp(np.cumsum(variant_daily_ret))
    mult_btc     = np.exp(np.cumsum(btc_ret))
    arc_v, asd_v, md_v, mld_v, ir1_v, ir2_v, *_ = _compute_quant_metrics(variant_daily_ret)
    arc_b, asd_b, md_b, mld_b, ir1_b, ir2_b, *_ = _compute_quant_metrics(btc_ret)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        dates, mult_btc, lw=2.0, color="dimgrey", ls="-.", zorder=2,
        label=(f"BTC Buy & Hold  "
               f"(aRC: {arc_b:+.2f}%)  →  {mult_btc[-1]:.3f}×"),
    )

    if has_spy and spy_ret is not None:
        mult_spy = np.exp(np.cumsum(spy_ret))
        arc_s, asd_s, md_s, mld_s, _, ir2_s, *_ = _compute_quant_metrics(spy_ret)
        ax.plot(
            dates, mult_spy, lw=1.8, color="olivedrab", ls="-.", zorder=2,
            label=(f"SPY Buy & Hold  "
                   f"(aRC: {arc_s:+.2f}%)  →  {mult_spy[-1]:.3f}×"),
        )

    ax.plot(
        dates, mult_variant, lw=2.5, color=variant_color, ls="-", zorder=4,
        label=(f"{variant_label}  "
               f"(aRC: {arc_v:+.2f}%)  →  {mult_variant[-1]:.3f}×"),
    )

    ax.fill_between(dates, 1, mult_btc, alpha=0.04, color="dimgrey", zorder=1)
    ax.axhline(1, color="black", lw=0.9, ls="--", alpha=0.35)

    # ── Metrics annotation table ───────────────────────────────────────────────
    # col widths: label=20, aRC=9 (8-digit value + "%"), aSD=6, MD=6, MLD=6, IR2=7
    col_hdr = f"{'Metric':<20} {'aRC(%)':>9}  {'aSD':>6}  {'MD(%)':>6}  {'MLD(y)':>6}  {'IR2':>7}"
    sep     = "-" * 63
    def _row(lbl: str, arc: float, asd: float, md: float, mld: float, ir2: float) -> str:
        return (f"{lbl:<20} {arc:>+8.1f}%  "
                f"{asd:>6.3f}  "
                f"{md:>5.1f}%  {mld:>6.2f}  {ir2:>+7.3f}")

    rows = [col_hdr, sep, _row("BTC B&H", arc_b, asd_b, md_b, mld_b, ir2_b)]
    if has_spy and spy_ret is not None:
        rows.append(_row("SPY B&H", arc_s, asd_s, md_s, mld_s, ir2_s))
    rows.append(_row(table_label, arc_v, asd_v, md_v, mld_v, ir2_v))

    ax.text(
        0.01, 0.02, "\n".join(rows),
        transform=ax.transAxes, fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.90, lw=0.8),
        fontfamily="monospace",
    )

    date_span = (dates[-1] - dates[0]).days
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3 if date_span > 730 else 1)
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8.5)

    ax.set_xlabel("Date  (day the return is realised)", fontsize=11)
    ax.set_ylabel("Portfolio Value  (Multiplier, start = 1.0)", fontsize=11)
    ax.set_title(
        f"Simulated Equity Curve  ·  {split_name} Set  ·  {variant_label}\n"
        f"vs. {'SPY B&H  ·  ' if has_spy else ''}BTC B&H  |  "
        f"Period: {str(dates[0].date())}  →  {str(dates[-1].date())}",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92,
              edgecolor="#cccccc", borderaxespad=0.6)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Equity curve saved → %s", fname)


def plot_equity_curve(
    prob:           np.ndarray,
    df:             pd.DataFrame,
    split_name:     str,
    out_dir:        str = ".",
    rolling_median: np.ndarray | None = None,
    lower_bound:    np.ndarray | None = None,
    upper_bound:    np.ndarray | None = None,
    q_star:         float = 30.0,
) -> None:
    """Generate FOUR separate equity curve PNGs for a given split.

        VariantA.png — static 0.50, always ±1
        VariantB.png — rolling 90d median, always ±1
        VariantC.png — rolling 30th/70th pct, trinary ±1/0
        VariantD.png — rolling 30th/70th pct, dynamic position sizing

    When rolling_median / lower_bound / upper_bound are None (training split),
    Variant B falls back to static 0.50 and C/D fall back to static globals.
    """
    n_rows      = len(df)
    n_tradeable = n_rows - SEQ_LEN

    if n_tradeable <= 0:
        log.warning(
            "[%s] Rows (%d) ≤ SEQ_LEN (%d). Skipping equity curves.",
            split_name, n_rows, SEQ_LEN,
        )
        return

    btc_ret      = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
    prob_aligned = prob[:n_tradeable]
    dates        = df.index[SEQ_LEN : SEQ_LEN + n_tradeable]

    has_spy = "spy_log_return" in df.columns
    spy_ret = (
        df["spy_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
        if has_spy else None
    )
    if not has_spy:
        log.warning(
            "[%s] 'spy_log_return' not in df — SPY B&H line will be omitted.",
            split_name,
        )

    # ── Strategy daily returns ─────────────────────────────────────────────────
    # Variant A: static 0.50, always ±1
    signal_a = np.where(prob_aligned >= 0.5, 1.0, -1.0)
    ret_a    = signal_a * btc_ret

    # Variant B: rolling median, always ±1
    med      = rolling_median[:n_tradeable] if rolling_median is not None else np.full(n_tradeable, 0.5)
    signal_b = np.where(prob_aligned >= med, 1.0, -1.0)
    ret_b    = signal_b * btc_ret

    # Variant C: rolling 30th/70th pct, trinary ±1/0
    ub = upper_bound[:n_tradeable] if upper_bound is not None else np.full(n_tradeable, UPPER_THRESHOLD)
    lb = lower_bound[:n_tradeable] if lower_bound is not None else np.full(n_tradeable, LOWER_THRESHOLD)
    signal_c = np.where(prob_aligned > ub, 1.0, np.where(prob_aligned < lb, -1.0, 0.0))
    ret_c    = signal_c * btc_ret

    # Variant D: dynamic position sizing with same rolling bands
    signal_d, weight_d = _compute_variant_d_weights(
        prob_aligned,
        lower_bound=lb if lower_bound is not None else None,
        upper_bound=ub if upper_bound is not None else None,
    )
    ret_d = signal_d * weight_d * btc_ret

    # ── Per-variant directional accuracy ──────────────────────────────────────
    acc_a = np.mean(np.sign(btc_ret) == np.where(prob_aligned >= 0.5, 1, -1)) * 100
    acc_b = np.mean(np.sign(btc_ret) == np.where(prob_aligned >= med, 1, -1)) * 100
    traded_c = signal_c != 0
    acc_c = (np.mean(np.sign(btc_ret[traded_c]) == signal_c[traded_c]) * 100
             if traded_c.any() else float("nan"))
    traded_d = signal_d != 0
    acc_d = (np.mean(np.sign(btc_ret[traded_d]) == signal_d[traded_d]) * 100
             if traded_d.any() else float("nan"))

    b_label = "rolling median" if rolling_median is not None else "static 0.50 fallback"
    cd_label = (
        f"rolling {q_star:.0f}th/{100.0 - q_star:.0f}th"
        if upper_bound is not None
        else f"static ±{UPPER_THRESHOLD:.2f}"
    )

    sname = split_name.lower()
    _plot_single_equity_curve(
        f"Variant A  (static 0.50 — always ±1)  ·  Acc: {acc_a:.1f}%",
        "Variant A",
        ret_a, btc_ret, spy_ret, has_spy, dates, split_name,
        "steelblue",
        os.path.join(out_dir, f"equity_curve_{sname}_VariantA.png"),
    )
    _plot_single_equity_curve(
        f"Variant B  ({b_label} — always ±1)  ·  Acc: {acc_b:.1f}%",
        "Variant B",
        ret_b, btc_ret, spy_ret, has_spy, dates, split_name,
        "mediumseagreen",
        os.path.join(out_dir, f"equity_curve_{sname}_VariantB.png"),
    )
    _plot_single_equity_curve(
        f"Variant C  ({cd_label} — 3-state)  ·  Acc: {acc_c:.1f}%",
        "Variant C (3-state)",
        ret_c, btc_ret, spy_ret, has_spy, dates, split_name,
        "tomato",
        os.path.join(out_dir, f"equity_curve_{sname}_VariantC.png"),
    )
    _plot_single_equity_curve(
        f"Variant D  ({cd_label} — dyn. sizing)  ·  Acc: {acc_d:.1f}%",
        "Variant D (dyn.sz.)",
        ret_d, btc_ret, spy_ret, has_spy, dates, split_name,
        "darkorange",
        os.path.join(out_dir, f"equity_curve_{sname}_VariantD.png"),
    )


def plot_threshold_sensitivity(
    y_true_val: np.ndarray, prob_val: np.ndarray, out_dir: str = ".",
) -> None:
    """Sweep thresholds 0.50–0.70; plot Conditional Win Rate + Coverage on Val."""
    upper_thresholds = np.arange(0.50, 0.71, 0.02)
    win_rates: list = []
    coverages: list = []

    for upper in upper_thresholds:
        lower      = 1.0 - upper
        trade_mask = (prob_val > upper) | (prob_val < lower)
        n_traded   = int(trade_mask.sum())
        n_total    = len(prob_val)
        coverage   = (n_traded / n_total) * 100.0

        if n_traded > 0:
            win_rate = accuracy_score(
                y_true_val[trade_mask].astype(int),
                (prob_val[trade_mask] >= 0.5).astype(int),
            ) * 100.0
        else:
            win_rate = float("nan")

        win_rates.append(win_rate)
        coverages.append(coverage)

        log.info(
            "  u=%.2f / l=%.2f  →  Coverage=%5.1f%%  WinRate=%5.1f%%",
            upper, lower, coverage, win_rate,
        )

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    line_wr,  = ax1.plot(upper_thresholds, win_rates,
                         color="steelblue", lw=2.5, ls="-", marker="o", ms=7, zorder=3,
                         label="Conditional Win Rate  (left axis, %)")
    line_cov, = ax2.plot(upper_thresholds, coverages,
                         color="tomato", lw=2.5, ls="-", marker="s", ms=7, zorder=3,
                         label="Coverage  (right axis, % days traded)")

    ax1.axvline(UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
                label=f"Selected threshold  =  {UPPER_THRESHOLD:.2f}")
    ax1.axhline(50.0, color="black", lw=0.9, ls=":", alpha=0.50, zorder=1,
                label="50% Win Rate  (random baseline)")

    ax1.set_xlabel("Upper Threshold  u   (Lower  l = 1 − u)", fontsize=11)
    ax1.set_ylabel("Conditional Win Rate  (%)", fontsize=11, color="steelblue")
    ax2.set_ylabel("Coverage  (% Validation Days Traded)", fontsize=11, color="tomato")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_xticks(upper_thresholds)
    ax1.set_xticklabels([f"{x:.2f}" for x in upper_thresholds], fontsize=9)
    ax1.set_title(
        "Threshold Sensitivity Analysis — Validation Set\n"
        "Blue: Conditional Win Rate  |  Red: Coverage  |  Green: selected threshold",
        fontsize=11, pad=14,
    )
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(ax1_handles + [line_cov], ax1_labels + [line_cov.get_label()],
               fontsize=9, loc="best", framealpha=0.92)
    ax1.grid(axis="both", alpha=0.28)
    plt.tight_layout()
    fname = os.path.join(out_dir, "threshold_sensitivity.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Threshold sensitivity saved → %s", fname)


# =============================================================================
# STEP 9 — TRADING SIGNAL OVERLAY
# =============================================================================

def plot_trading_signals(
    df:             pd.DataFrame,
    prob:           np.ndarray,
    split_name:     str,
    out_dir:        str = ".",
    variant:        str = "C",
    rolling_median: np.ndarray | None = None,
    lower_bound:    np.ndarray | None = None,
    upper_bound:    np.ndarray | None = None,
    q_star:         float = 30.0,
) -> None:
    """2-panel trading signal chart with a shared X-axis.

    variant="A" — Variant A: static 0.50, ±1. Bottom panel: single 0.50 line.
    variant="B" — Variant B: rolling median, ±1. Bottom panel: rolling median line.
    variant="C" — Variant C: rolling 30th/70th, ±1/0. Bottom panel: rolling ribbon.
    variant="D" — Variant D: same masks as C. Bottom panel: same rolling ribbon.

    Saves: out_dir/trading_signals_{split_name.lower()}_variant{variant}.png
    """
    n_tradeable = len(df) - SEQ_LEN
    if n_tradeable <= 0:
        log.warning(
            "[%s] Insufficient rows (%d) for trading signals chart — skipping.",
            split_name, len(df),
        )
        return

    prob_al  = prob[:n_tradeable]
    btc_ret  = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
    dates    = df.index[SEQ_LEN : SEQ_LEN + n_tradeable]
    cum_btc  = np.exp(np.cumsum(btc_ret))

    # ── Signal masks per variant ───────────────────────────────────────────────
    if variant == "A":
        long_mask  = prob_al >= 0.5
        short_mask = prob_al <  0.5
        hold_mask  = np.zeros(len(prob_al), dtype=bool)
        n_hold     = 0
    elif variant == "B":
        med        = rolling_median[:n_tradeable] if rolling_median is not None else np.full(n_tradeable, 0.5)
        long_mask  = prob_al >= med
        short_mask = prob_al <  med
        hold_mask  = np.zeros(len(prob_al), dtype=bool)
        n_hold     = 0
    else:  # "C" or "D"
        ub         = upper_bound[:n_tradeable] if upper_bound is not None else np.full(n_tradeable, UPPER_THRESHOLD)
        lb         = lower_bound[:n_tradeable] if lower_bound is not None else np.full(n_tradeable, LOWER_THRESHOLD)
        long_mask  = prob_al >  ub
        short_mask = prob_al <  lb
        hold_mask  = ~long_mask & ~short_mask
        n_hold     = int(hold_mask.sum())

    n_long   = int(long_mask.sum())
    n_short  = int(short_mask.sum())
    n_total  = len(prob_al)
    coverage = 100.0 * (n_long + n_short) / n_total if n_total > 0 else 0.0
    date_span = (dates[-1] - dates[0]).days

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(17, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.06},
    )

    # ═══════════════════════ TOP PANEL ═══════════════════════════════════════

    ax_top.plot(dates, cum_btc, color="#3a5f8a", lw=1.8, zorder=3, alpha=0.90,
                label="BTC Buy & Hold (%)")
    ax_top.fill_between(dates, 1, cum_btc, alpha=0.06, color="#3a5f8a", zorder=2)
    ax_top.axhline(1, color="black", lw=0.75, ls="--", alpha=0.25, zorder=1)

    # Hold dots (Variant C / D only)
    if variant in ("C", "D") and hold_mask.any():
        ax_top.scatter(dates[hold_mask], cum_btc[hold_mask],
                       c="#b0b0b0", s=10, alpha=0.20, zorder=4, linewidths=0,
                       label=f" Out of market [{n_hold:,} days]")

    # Short/Long signal scatter
    if variant == "A":
        short_lbl = f"Short  P < 0.50  [{n_short:,} days]"
        long_lbl  = f"Long   P ≥ 0.50  [{n_long:,} days]"
    elif variant == "B":
        short_lbl = f"Short  P < rolling median  [{n_short:,} days]"
        long_lbl  = f"Long   P ≥ rolling median  [{n_long:,} days]"
    else:
        _uq = 100.0 - q_star
        bound_type = (
            f"rolling {q_star:.0f}th/{_uq:.0f}th"
            if upper_bound is not None
            else f"static ±{UPPER_THRESHOLD:.2f}"
        )
        short_lbl  = f"Short  [{bound_type}]  [{n_short:,} days]"
        long_lbl   = f"Long   [{bound_type}]  [{n_long:,} days]"

    ax_top.scatter(dates[short_mask], cum_btc[short_mask],
                   c="#c0392b", s=44, alpha=0.88, zorder=6, linewidths=0,
                   marker="v", label=short_lbl)
    ax_top.scatter(dates[long_mask], cum_btc[long_mask],
                   c="#1e8449", s=44, alpha=0.88, zorder=6, linewidths=0,
                   marker="^", label=long_lbl)

    _uq2 = 100.0 - q_star
    variant_titles = {
        "A": "Variant A  (static 0.50 — always ±1  |  no hold zone)",
        "B": "Variant B  (rolling 90d median — always ±1  |  no hold zone)",
        "C": f"Variant C  (rolling {q_star:.0f}th/{_uq2:.0f}th pct — 3-state)",
        "D": f"Variant D  (rolling {q_star:.0f}th/{_uq2:.0f}th pct — dynamic sizing)",
    }
    hold_part = f"  ·  {n_hold:,} Out of market" if variant in ("C", "D") else ""
    ax_top.set_title(
        f"Trading Signal Overlay  ·  {split_name} Set  ·  "
        f"Coverage: {coverage:.1f}%  ({n_long:,} long  ·  {n_short:,} short{hold_part})\n"
        f"{variant_titles[variant]}  |  "
        f"Period: {str(dates[0].date())}  →  {str(dates[-1].date())}",
        fontsize=11, pad=12, fontweight="semibold",
    )
    ax_top.set_ylabel("BTC Portfolio Value  (Multiplier, start = 1.0)", fontsize=11)
    ax_top.legend(fontsize=9.5, loc="lower left", framealpha=0.93, edgecolor="#cccccc",
                  borderaxespad=0.5)
    ax_top.grid(axis="both", color="#e0e0e0", lw=0.55, zorder=0)
    ax_top.spines[["top", "right"]].set_visible(False)

    # ═══════════════════════ BOTTOM PANEL ════════════════════════════════════

    ax_bot.plot(dates, prob_al, color="#5b7fa6", lw=1.15, alpha=0.82,
                zorder=3, label="Predicted P(BTC ↑)")

    if variant == "B":
        ax_bot.plot(dates, med, color="#2e8b57", lw=1.5, ls="--", zorder=4,
                    label="Rolling 90d Median")
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    elif variant in ("C", "D"):
        _uq3 = 100.0 - q_star
        ax_bot.fill_between(dates, lb, ub,
                            color="#888888", alpha=0.10, zorder=0,
                            label=f"Hold zone ({q_star:.0f}th–{_uq3:.0f}th)")
        ax_bot.plot(dates, ub, color="#1e8449", lw=1.2, ls="--", zorder=4,
                    label=f"Upper (rolling {_uq3:.0f}th pct)")
        ax_bot.plot(dates, lb, color="#c0392b", lw=1.2, ls="--", zorder=4,
                    label=f"Lower (rolling {q_star:.0f}th pct)")
        ax_bot.fill_between(dates, prob_al, ub, where=(prob_al > ub),
                            color="#1e8449", alpha=0.38, zorder=2, interpolate=True,
                            label=f"Long zone  (P > {_uq3:.0f}th)")
        ax_bot.fill_between(dates, prob_al, lb, where=(prob_al < lb),
                            color="#c0392b", alpha=0.38, zorder=2, interpolate=True,
                            label=f"Short zone  (P < {q_star:.0f}th)")
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    else:  # variant A
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    ax_bot.axhline(0.50, color="#555555", lw=0.9, ls=":", alpha=0.50, zorder=4,
                   label="Decision boundary = 0.50")
    ax_bot.set_ylabel("Predicted Probability", fontsize=11)
    ax_bot.set_xlabel("Date  (day the return is realised)", fontsize=11)
    ax_bot.legend(fontsize=8.5, loc="best", framealpha=0.93,
                  edgecolor="#cccccc", ncol=3)
    ax_bot.grid(axis="both", color="#e0e0e0", lw=0.55, zorder=0)
    ax_bot.spines[["top", "right"]].set_visible(False)

    if date_span > 730:
        ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8.5)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"trading_signals_{split_name.lower()}_variant{variant}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Trading signals chart saved → %s", fname)


# =============================================================================
# STEP 9B — ADVANCED TEAR SHEET VISUALIZATIONS
# =============================================================================

# ── Shared palette and labels ─────────────────────────────────────────────────
_VC = {"A": "steelblue", "B": "mediumseagreen", "C": "tomato", "D": "darkorange"}
_VL = {
    "A": "Variant A  (static 0.50)",
    "B": "Variant B  (rolling median)",
    "C": "Variant C  (rolling {q_star:.0f}th/{uq:.0f}th)",
    "D": "Variant D  (dynamic sizing)",
}


def _variant_label(v: str, q_star: float = 30.0) -> str:
    """Return a dynamic legend label for variant v reflecting the active q_star."""
    if v in ("A", "B"):
        return _VL[v]
    uq = 100.0 - q_star
    if v == "C":
        return f"Variant C  (rolling {q_star:.0f}th/{uq:.0f}th)"
    if v == "D":
        return f"Variant D  (dyn. sizing {q_star:.0f}th/{uq:.0f}th)"
    return f"Variant {v}"


def plot_variant_confusion_matrix(
    signals:    np.ndarray,
    btc_ret:    np.ndarray,
    split_name: str,
    variant:    str,
    out_dir:    str = ".",
) -> None:
    """Signal-vs-actual confusion matrix for all 4 variants.

    A / B (binary ±1)  → 2×2 matrix: rows = Short / Long, cols = Down / Up.
    C / D (trinary)    → 3×2 matrix: rows = Short / Out of market / Long, cols = Down / Up.
    Orientation (Y=model signal, X=actual direction) is consistent across all variants.
    """
    actual_up = (btc_ret > 0).astype(int)

    if variant in ("A", "B"):
        pred_cls  = (signals > 0).astype(int)
        row_labels = ["Short", "Long"]
        col_labels  = ["Down", "Up"]
        cm = np.zeros((2, 2), dtype=int)
        for r, p in enumerate([0, 1]):
            for c, a in enumerate([0, 1]):
                cm[r, c] = int(np.sum((pred_cls == p) & (actual_up == a)))
        figsize = (5.5, 4.5)
        title_extra = "Binary  |  Predicted Direction vs Actual Direction"
    else:
        row_labels = ["Short", "Out of market", "Long"]
        col_labels  = ["Down", "Up"]
        cm = np.zeros((3, 2), dtype=int)
        for r, s in enumerate([-1, 0, 1]):
            for c, a in enumerate([0, 1]):
                cm[r, c] = int(np.sum((signals == s) & (actual_up == a)))
        figsize = (5.0, 5.5)
        title_extra = "3-State  |  Model Signal vs Actual Direction"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=col_labels, yticklabels=row_labels,
        linewidths=0.5, linecolor="#cccccc",
        ax=ax, cbar=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=10)
    ax.set_xlabel("Actual Market Direction", fontsize=11, labelpad=8)
    ax.set_ylabel("Model Signal", fontsize=11, labelpad=8)
    ax.set_title(
        f"Confusion Matrix — Variant {variant}  [{split_name} Set]\n{title_extra}",
        fontsize=11, pad=12, fontweight="semibold",
    )
    plt.tight_layout()
    fname = os.path.join(out_dir, f"confusion_matrix_variant{variant}_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Variant %s confusion matrix → %s", variant, fname)


def plot_drawdown_curves(
    btc_ret:              np.ndarray,
    variant_returns_dict: dict,
    split_name:           str,
    dates:                pd.DatetimeIndex,
    out_dir:              str = ".",
    q_star:               float = 30.0,
) -> None:
    """Underwater equity curves: % drawdown from running peak for all variants vs BTC B&H."""
    def _dd(ret: np.ndarray) -> np.ndarray:
        eq = np.exp(np.cumsum(ret))
        return (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100.0

    fig, ax = plt.subplots(figsize=(17, 7))
    btc_dd  = _dd(btc_ret)
    btc_mdd = float(np.min(btc_dd))
    ax.fill_between(dates, btc_dd, 0, alpha=0.18, color="grey",
                    label=f"BTC B&H  (MDD: {btc_mdd:.1f}%)")
    ax.plot(dates, btc_dd, color="grey", lw=0.8, alpha=0.55)

    for v, ret in variant_returns_dict.items():
        dd  = _dd(ret)
        mdd = float(np.min(dd))
        ax.plot(dates, dd, lw=1.8, color=_VC[v],
                label=f"{_variant_label(v, q_star)}  (MDD: {mdd:.1f}%)")

    ax.axhline(0, color="black", lw=1.2, ls="-", alpha=0.60, zorder=3)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown  (%)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title(
        f"Underwater Equity Curves  ·  {split_name} Set\n"
        "Grey area = BTC Buy & Hold  |  Lines = Strategy Drawdowns",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.93, edgecolor="#cccccc")
    ax.grid(axis="y", color="#e0e0e0", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    date_span = (dates[-1] - dates[0]).days
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3 if date_span > 730 else 1)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8.5)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"drawdown_curves_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Drawdown curves → %s", fname)


def plot_return_distributions(
    btc_ret:              np.ndarray,
    variant_returns_dict: dict,
    split_name:           str,
    out_dir:              str = ".",
    q_star:               float = 30.0,
) -> None:
    """KDE + histogram of daily log returns: BTC B&H vs all variants in one figure."""
    items = [("BTC B&H", "#3a5f8a", btc_ret)] + [
        (_variant_label(v, q_star), _VC[v], ret) for v, ret in variant_returns_dict.items()
    ]
    ncols = len(items)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (label, color, ret) in zip(axes, items):
        pct = ret * 100.0
        mu, sigma = float(np.mean(pct)), float(np.std(pct))
        skew = float(pd.Series(pct).skew())
        kurt = float(pd.Series(pct).kurtosis())
        sns.histplot(pct, bins=50, kde=True, ax=ax, color=color,
                     alpha=0.50, edgecolor="none", stat="density")
        ax.axvline(mu, color="black", lw=1.4, ls="--", label=f"Mean = {mu:+.3f}%")
        ax.axvline(0, color="#888888", lw=0.8, ls=":", alpha=0.55)
        ax.set_xlabel("Daily Log Return  (%)", fontsize=9.5)
        ax.set_ylabel("Density", fontsize=9.5)
        ax.set_title(
            f"{label}\nμ={mu:+.3f}%  σ={sigma:.3f}%\nSkew={skew:.2f}  Kurt={kurt:.2f}",
            fontsize=9.5,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Daily Return Distributions  ·  {split_name} Set",
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(out_dir, f"return_distributions_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Return distributions → %s", fname)


def plot_rolling_sharpe(
    btc_ret:              np.ndarray,
    variant_returns_dict: dict,
    split_name:           str,
    dates:                pd.DatetimeIndex,
    window:               int = 30,
    out_dir:              str = ".",
    q_star:               float = 30.0,
) -> None:
    """Rolling annualised Sharpe (30-day window, rf=0) for all variants vs BTC B&H."""
    def _rs(ret: np.ndarray) -> np.ndarray:
        s = pd.Series(ret)
        return (s.rolling(window).mean() / s.rolling(window).std() * np.sqrt(365)).values

    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(dates, _rs(btc_ret), color="grey", lw=1.3, alpha=0.65,
            ls="--", label="BTC B&H")
    for v, ret in variant_returns_dict.items():
        rs = _rs(ret)
        ax.plot(dates, rs, lw=1.8, color=_VC[v], label=_variant_label(v, q_star))
        ax.fill_between(dates, rs, 0, where=(rs < 0),
                        color=_VC[v], alpha=0.08, interpolate=True)

    ax.axhline(0, color="black", lw=1.0, ls="-", alpha=0.55, zorder=3)
    ax.axhline(1, color="#2e7d32", lw=1.2, ls="--", alpha=0.65,
               label="Sharpe = 1  (reference)", zorder=3)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Rolling {window}-Day Sharpe  (Annualised)", fontsize=11)
    ax.set_title(
        f"Rolling {window}-Day Sharpe Ratio  ·  {split_name} Set\n"
        "Annualised  (√365 basis,  rf = 0)  —  clipped at ±6",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.93, edgecolor="#cccccc")
    ax.grid(axis="both", color="#e0e0e0", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    date_span = (dates[-1] - dates[0]).days
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3 if date_span > 730 else 1)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8.5)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"rolling_sharpe_{split_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Rolling Sharpe → %s", fname)


def plot_monthly_returns_heatmap(
    strategy_ret: np.ndarray,
    dates:        pd.DatetimeIndex,
    split_name:   str,
    variant:      str,
    out_dir:      str = ".",
    q_star:       float = 30.0,
) -> None:
    """Calendar heatmap of monthly compounded returns for a single variant.

    Rows = calendar months (Jan–Dec), columns = years.
    Red = loss month, Green = gain month. Only months present in the data are shown.
    """
    ret_series = pd.Series(strategy_ret, index=pd.DatetimeIndex(dates))
    monthly    = ret_series.resample("ME").sum()            # sum of log returns
    monthly_pct = (np.exp(monthly) - 1.0) * 100.0          # → percentage

    pivot = pd.DataFrame({
        "year":  monthly_pct.index.year,
        "month": monthly_pct.index.month,
        "ret":   monthly_pct.values,
    }).pivot(index="month", columns="year", values="ret")
    pivot.index = [calendar.month_abbr[m] for m in pivot.index]

    pivot.columns.name = ""          # suppress "year" header above x-axis
    n_rows  = pivot.shape[0]         # calendar months present (1–12)
    n_cols  = pivot.shape[1]         # years present
    cell_w, cell_h = 1.3, 0.62      # inches per cell
    fig_w = max(5.5, cell_w * n_cols + 2.8)
    fig_h = max(3.5, cell_h * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    annot_sz = max(7, min(11, int(90 / max(n_rows, n_cols))))   # scale with density
    vmax = float(np.nanpercentile(np.abs(pivot.values[~np.isnan(pivot.values)]), 95))
    vmax = max(vmax, 1.0)   # guard against near-zero range
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.4, linecolor="#dddddd",
        ax=ax, cbar_kws={"label": "Monthly Return  (%)", "shrink": 0.8},
        annot_kws={"size": annot_sz, "weight": "bold"},
    )
    ax.set_xlabel("", fontsize=11)   # year labels on tick are sufficient
    ax.set_ylabel("", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=10)
    ax.set_title(
        f"Monthly Returns Heatmap — {_variant_label(variant, q_star)}  [{split_name} Set]\n"
        f"Green = gain  |  Red = loss  |  {n_cols} year(s) × {n_rows} month(s)",
        fontsize=11, pad=12, fontweight="semibold",
    )
    plt.tight_layout()
    fname = os.path.join(
        out_dir,
        f"monthly_returns_heatmap_variant{variant}_{split_name.lower()}.png",
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Monthly returns heatmap → %s", fname)


# =============================================================================
# STEP 10 — PERMUTATION FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    model:             tf.keras.Model,
    X_val:             np.ndarray,
    y_val:             np.ndarray,
    y_val_aligned:     np.ndarray,
    prob_val_baseline: np.ndarray,
    feature_cols:      list,
    n_repeats:         int = 3,
    out_dir:           str = ".",
) -> None:
    """
    Compute and visualise Permutation Feature Importance on the Validation set.

    For each feature i:
        1. Randomly permute column i of X_val (destroys signal, preserves
           marginal distribution of all other features).
        2. Re-create windowed dataset and generate predictions.
        3. Compute AUC drop = baseline_auc − permuted_auc.
        4. Average over n_repeats to reduce Monte Carlo variance.

    Features with large positive AUC drop are most important.
    Features with negative AUC drop are likely toxic (permuting them helps).
    """
    baseline_auc = roc_auc_score(y_val_aligned, prob_val_baseline)
    log.info("Baseline val AUC = %.4f", baseline_auc)

    label_map = {
        "btc_log_return":   "BTC Return",
        "spy_log_return":   "SPY Return",
        "gold_log_return":  "Gold Return",
        "nvda_log_return":  "NVDA Return",
        "dxy_log_return":   "DXY Return",
        "vix_log_return":   "VIX Return",
        "vix_level":        "VIX Level",
        "us10y_level":      "US10Y Yield",
        "btc_volume_ratio": "BTC Vol. Ratio",
        "btc_rsi_14":       "BTC RSI(14)",
        "btc_roll_vol_21":  "BTC RollVol(21d)",
        "btc_stoch_rsi_14": "BTC StochRSI(14)",
        "btc_macd_hist":    "BTC MACD Hist.",
        "btc_atr_14_ratio": "BTC ATR Ratio(14)",
        "is_weekend":       "Is Weekend",
        # Raw-price mode aliases (if present in feature_cols)
        "btc_close":        "BTC Price",
        "spy_close":        "SPY Price",
        "gold_close":       "Gold Price",
        "nvda_close":       "NVDA Price",
        "dxy_close":        "DXY Price",
    }

    importances: list = []

    for i, col in enumerate(feature_cols):
        auc_drops: list = []
        for _ in range(n_repeats):
            X_perm  = X_val.copy()
            rng_idx = np.random.permutation(len(X_perm))
            X_perm[:, i] = X_perm[rng_idx, i]

            ds_perm, _ = make_tf_dataset(
                X_perm, y_val, SEQ_LEN, BATCH_SIZE, shuffle=False
            )
            probs_perm = model.predict(ds_perm, verbose=0).ravel()
            perm_auc   = roc_auc_score(y_val_aligned, probs_perm)
            auc_drops.append(baseline_auc - perm_auc)

        mean_drop = float(np.mean(auc_drops))
        std_drop  = float(np.std(auc_drops, ddof=1)) if len(auc_drops) > 1 else 0.0
        importances.append((col, mean_drop, std_drop))
        log.info(
            "  Feature %-22s  AUC drop = %+.4f ± %.4f (n=%d repeats)",
            col, mean_drop, std_drop, n_repeats,
        )

    importances.sort(key=lambda x: x[1], reverse=True)
    labels = [label_map.get(c, c) for c, _, _ in importances]
    values = [v for _, v, _ in importances]
    stds   = [s for _, _, s in importances]
    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, values, xerr=stds, color=colors,
                    edgecolor="white", height=0.7,
                    error_kw=dict(ecolor="black", capsize=3, lw=1.2))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.60)

    for bar, val in zip(bars, values):
        ax.text(
            val + (0.0003 if val >= 0 else -0.0003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8.5, color="black",
        )

    ax.set_xlabel(
        f"AUC Drop  (baseline − permuted)  |  n_repeats={n_repeats}  |  error bars = ±1 SD\n"
        "Blue = positive contribution (important)  |  Red = toxic (AUC drops when kept)",
        fontsize=10,
    )
    ax.set_title(
        f"Permutation Feature Importance — Validation Set\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  SEQ_LEN={SEQ_LEN}",
        fontsize=12, pad=14,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.30)
    plt.tight_layout()
    fname = os.path.join(out_dir, "feature_importance_val.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance saved → %s", fname)


# =============================================================================
# RUN_EXPERIMENT — One complete train+evaluate cycle for a single experiment
# =============================================================================

def calibrate_optimal_quantiles(
    val_preds: np.ndarray,
    val_returns: np.ndarray,
    val_true: np.ndarray,
    train_tail_preds: np.ndarray,
    lookback: int = 90,
    objective: str = CALIBRATION_OBJECTIVE,
) -> tuple[float, pd.DataFrame]:
    """
    Grid-searches the optimal symmetric quantile q* on the validation set.
    Returns (q_star, results_df).
    """
    records = []
    n_tradeable = len(val_returns)
    
    for q in Q_CANDIDATES:
        # Pobieramy dynamiczne pasma dla danego kandydata 'q'
        lb, _, ub = apply_rolling_thresholds(
            train_tail_preds, val_preds, lookback, lower_q=q, upper_q=100.0-q
        )
        
        # Skracamy do dni 'handlowalnych'
        lb_al = lb[:n_tradeable]
        ub_al = ub[:n_tradeable]
        p_al  = val_preds[:n_tradeable]
        
        # Symulujemy logikę Wariantu D
        signals, weights = _compute_variant_d_weights(p_al, lower_bound=lb_al, upper_bound=ub_al)
        daily_ret = signals * weights * val_returns
        
        traded_mask = (signals != 0)
        n_traded = int(traded_mask.sum())
        coverage = n_traded / n_tradeable if n_tradeable > 0 else 0.0
        
        # Liczymy warunkowe Accuracy (Win Rate)
        if n_traded > 0:
            cond_acc = accuracy_score(
                val_true[traded_mask].astype(int),
                (p_al[traded_mask] >= 0.5).astype(int)
            )
        else:
            cond_acc = 0.0
            
        # Liczymy wskaźniki kwantowe (m.in. Sharpe / IR1)
        arc, asd, md, mld, ir1, ir2, sortino, calmar, pf = _compute_quant_metrics(daily_ret)
        
        records.append({
            "q": q,
            "sharpe": ir1,
            "accuracy": cond_acc * 100.0,
            "coverage": coverage * 100.0
        })
        
    # Odsiewamy kombinacje, które nie spełniają minimalnego coverage
    feasible = [r for r in records if r["coverage"] >= MIN_COVERAGE * 100.0]
    
    if not feasible:
        log.warning("No q meets coverage >= %.1f%%. Falling back to max coverage.", MIN_COVERAGE * 100.0)
        best_r = max(records, key=lambda x: x["coverage"])
    else:
        if objective == "sharpe":
            best_r = max(feasible, key=lambda x: x["sharpe"])
        elif objective == "accuracy":
            best_r = max(feasible, key=lambda x: x["accuracy"])
        else:
            raise ValueError(f"Unknown objective: {objective}")
            
    return best_r["q"], pd.DataFrame(records)

def run_experiment(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    X_test:   np.ndarray | None,
    y_test:   np.ndarray | None,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame | None,
    use_weights: bool,
    exp_name:    str,
    base_dir:    str,
) -> None:
    """
    Execute one full training + evaluation + visualisation cycle.

    Outputs are isolated under base_dir/exp_name/.
    A per-experiment FileHandler is attached to 'log' for the duration and
    removed in a finally block so the global log file continues to receive all
    output from both experiments.
    """
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ── Per-experiment log file ────────────────────────────────────────────────
    exp_fh = logging.FileHandler(
        os.path.join(exp_dir, f"run_log_{exp_name}.txt"), mode="w", encoding="utf-8"
    )
    exp_fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(exp_fh)
    try:
        log.info("━" * 65)
        log.info("  EXPERIMENT: %s  |  use_weights=%s", exp_name.upper(), use_weights)
        log.info("  Output dir: %s", exp_dir)
        log.info("━" * 65)

        # ── STEP F1b: Class balance diagnostic ────────────────────────────────
        for _sname, _y in [("Train", y_train), ("Val", y_val)]:
            _pos = float(np.mean(_y))
            log.info(
                "  %s class balance: %.1f%% UP  /  %.1f%% DOWN",
                _sname, _pos * 100.0, (1.0 - _pos) * 100.0,
            )

        # ── STEP F2: Sample weights ────────────────────────────────────────────
        if use_weights:
            train_weights = _compute_sample_weights(train_df, SEQ_LEN)
            val_weights   = _compute_sample_weights(val_df,   SEQ_LEN)
            log.info("use_weights=True — sample weights applied.")
            log.info(
                "Train weights: n=%d  mean=%.4f  min=%.4f  max=%.4f",
                len(train_weights), train_weights.mean(),
                train_weights.min(), train_weights.max(),
            )
        else:
            train_weights = val_weights = None
            log.info("use_weights=False — uniform loss weighting.")

        # ── STEP F3: Build windowed datasets ──────────────────────────────────
        log.info("[STEP 3] Building sliding-window tf.data.Datasets ...")
        train_ds, y_train_aligned = make_tf_dataset(
            X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=True,
            sample_weights=train_weights,
        )
        train_eval_ds, _ = make_tf_dataset(
            X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=False,
            sample_weights=train_weights,
        )
        val_ds, y_val_aligned = make_tf_dataset(
            X_val, y_val, SEQ_LEN, BATCH_SIZE, shuffle=False,
            sample_weights=val_weights,
        )
        if X_test is not None:
            test_ds, y_test_aligned = make_tf_dataset(
                X_test, y_test, SEQ_LEN, BATCH_SIZE, shuffle=False,
            )
        else:
            test_ds = y_test_aligned = None

        # ── STEP F4: Build, train, save model ─────────────────────────────────
        n_features = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        log.info("[STEP 4] Building and training LSTM model ...")
        model   = build_model(SEQ_LEN, n_features)
        history = train_model(model, train_ds, val_ds, log_dir=exp_dir)
        model_path = os.path.join(exp_dir, "model.keras")
        model.save(model_path)
        log.info("Model saved → %s", model_path)

        # ── STEP F5: Generate predictions ─────────────────────────────────────
        log.info("[STEP 5] Generating predictions ...")
        prob_train = model.predict(train_eval_ds, verbose=0).ravel()
        prob_val   = model.predict(val_ds,   verbose=0).ravel()
        prob_test  = model.predict(test_ds,  verbose=0).ravel() if test_ds else None
        
        # ── zabezpieczenie ─────────────────────────────────────

        assert not np.isnan(prob_train).any(), "ABORT: prob_train contains NaN (Training Diverged)"
        assert not np.isnan(prob_val).any(),   "ABORT: prob_val contains NaN (Training Diverged)"

    # ── STEP F5b: Threshold Calibration & Rolling dynamic thresholds ───────
        log.info("[STEP 5b] Calibrating optimal thresholds on Validation set ...")
        
        # Przygotowujemy dane walidacyjne pod kalibrację
        _n_vl = len(val_df) - SEQ_LEN
        vl_btc_ret = val_df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + _n_vl]
        vl_y_true  = y_val_aligned[:_n_vl]
        _p_vl      = prob_val[:_n_vl]
        
        train_tail = prob_train[-90:]
        
        # Wywołujemy naszą funkcję optymalizacyjną
        q_star, cal_df = calibrate_optimal_quantiles(
            val_preds=prob_val,
            val_returns=vl_btc_ret,
            val_true=vl_y_true,
            train_tail_preds=train_tail,
            lookback=90,
            objective=CALIBRATION_OBJECTIVE
        )
        
        log.info("\nCalibration grid:\n%s", cal_df.to_string(index=False))
        log.info("Selected q* = %.1f based on objective: '%s'", q_star, CALIBRATION_OBJECTIVE)
        
        # APLIKUJEMY OPTYMALNE PRAMETRY q* DO ZBIORÓW VAL I TEST
        lb_val, med_val, ub_val = apply_rolling_thresholds(
            prob_train, prob_val, lower_q=q_star, upper_q=100.0-q_star
        )
        
        if prob_test is not None:
            lb_test, med_test, ub_test = apply_rolling_thresholds(
                prob_val, prob_test, lower_q=q_star, upper_q=100.0-q_star
            )
        else:
            lb_test = med_test = ub_test = None

        # ── STEP F5c: Pre-compute signals + returns (reused by tear sheet) ───────
        # Train — Variant A only (no rolling bounds available)
        _n_tr = len(train_df) - SEQ_LEN
        _p_tr = prob_train[:_n_tr]
        tr_btc_ret = train_df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + _n_tr]
        tr_dates   = train_df.index[SEQ_LEN : SEQ_LEN + _n_tr]
        _sa_tr     = np.where(_p_tr >= 0.5, 1.0, -1.0)
        tr_signals = {"A": _sa_tr}
        tr_returns = {"A": _sa_tr * tr_btc_ret}

        # Val — all 4 variants
        _n_vl = len(val_df) - SEQ_LEN
        _p_vl = prob_val[:_n_vl]
        vl_btc_ret = val_df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + _n_vl]
        vl_dates   = val_df.index[SEQ_LEN : SEQ_LEN + _n_vl]
        _m_vl = med_val[:_n_vl]; _ub_vl = ub_val[:_n_vl]; _lb_vl = lb_val[:_n_vl]
        _sa_vl = np.where(_p_vl >= 0.5,   1.0, -1.0)
        _sb_vl = np.where(_p_vl >= _m_vl, 1.0, -1.0)
        _sc_vl = np.where(_p_vl > _ub_vl, 1.0, np.where(_p_vl < _lb_vl, -1.0, 0.0))
        _sd_vl, _wd_vl = _compute_variant_d_weights(_p_vl, lower_bound=_lb_vl, upper_bound=_ub_vl)
        vl_signals = {"A": _sa_vl, "B": _sb_vl, "C": _sc_vl, "D": _sd_vl}
        vl_returns = {
            "A": _sa_vl * vl_btc_ret,
            "B": _sb_vl * vl_btc_ret,
            "C": _sc_vl * vl_btc_ret,
            "D": _sd_vl * _wd_vl * vl_btc_ret,
        }

        # Test — all 4 variants (if enabled)
        if prob_test is not None:
            _n_ts = len(test_df) - SEQ_LEN
            _p_ts = prob_test[:_n_ts]
            ts_btc_ret = test_df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + _n_ts]
            ts_dates   = test_df.index[SEQ_LEN : SEQ_LEN + _n_ts]
            _m_ts = med_test[:_n_ts]; _ub_ts = ub_test[:_n_ts]; _lb_ts = lb_test[:_n_ts]
            _sa_ts = np.where(_p_ts >= 0.5,   1.0, -1.0)
            _sb_ts = np.where(_p_ts >= _m_ts, 1.0, -1.0)
            _sc_ts = np.where(_p_ts > _ub_ts, 1.0, np.where(_p_ts < _lb_ts, -1.0, 0.0))
            _sd_ts, _wd_ts = _compute_variant_d_weights(_p_ts, lower_bound=_lb_ts, upper_bound=_ub_ts)
            ts_signals = {"A": _sa_ts, "B": _sb_ts, "C": _sc_ts, "D": _sd_ts}
            ts_returns = {
                "A": _sa_ts * ts_btc_ret,
                "B": _sb_ts * ts_btc_ret,
                "C": _sc_ts * ts_btc_ret,
                "D": _sd_ts * _wd_ts * ts_btc_ret,
            }
        else:
            ts_btc_ret = ts_dates = ts_signals = ts_returns = None

        # ── STEP F6: Evaluate all variants ────────────────────────────────────
        log.info("[STEP 6] Evaluating all variants ...")

        # Training: Variant A only — no prior split exists to seed the rolling window
        evaluate_standard(y_train_aligned, prob_train, "Train",
                          df=train_df, sample_weight=train_weights)

        # Validation: all 4 variants with rolling bounds
        evaluate_standard(y_val_aligned, prob_val, "Val",
                          df=val_df, sample_weight=val_weights)
        evaluate_rolling_median(y_val_aligned, prob_val, med_val, "Val",
                                df=val_df, sample_weight=val_weights)
        evaluate_3state(y_val_aligned, prob_val, "Val",
                        df=val_df, lower_bound=lb_val, upper_bound=ub_val)
        evaluate_variant_d(val_df, prob_val, "Val",
                           lower_bound=lb_val, upper_bound=ub_val)

        # Test: all 4 variants with rolling bounds (if enabled)
        if prob_test is not None:
            evaluate_standard(y_test_aligned, prob_test, "Test", df=test_df)
            evaluate_rolling_median(y_test_aligned, prob_test, med_test, "Test", df=test_df)
            evaluate_3state(y_test_aligned, prob_test, "Test",
                            df=test_df, lower_bound=lb_test, upper_bound=ub_test)
            evaluate_variant_d(test_df, prob_test, "Test",
                               lower_bound=lb_test, upper_bound=ub_test)

        # ── STEP F7: Diagnostic visualizations ────────────────────────────────
        log.info("[STEP 7] Generating diagnostic visualizations ...")
        plot_training_history(history, out_dir=exp_dir)

        train_auc = roc_auc_score(y_train_aligned, prob_train, sample_weight=train_weights)
        plot_roc_curve(y_train_aligned, prob_train, train_auc, "Train", out_dir=exp_dir, sample_weight=train_weights)

        val_auc = roc_auc_score(y_val_aligned, prob_val, sample_weight=val_weights)
        plot_roc_curve(y_val_aligned, prob_val, val_auc, "Val", out_dir=exp_dir, sample_weight=val_weights)

        if prob_test is not None:
            test_auc = roc_auc_score(y_test_aligned, prob_test)
            plot_roc_curve(y_test_aligned, prob_test, test_auc, "Test", out_dir=exp_dir)

        # ── STEP F8: Advanced quant finance plots ─────────────────────────────
        log.info("[STEP 8] Generating advanced quantitative finance plots ...")
        plot_prob_distribution(prob_train, "Train", out_dir=exp_dir)
        plot_prob_distribution(prob_val,   "Val",   out_dir=exp_dir)

        # Equity curves — 4 PNGs per split; Train uses static fallback, Val/Test use rolling bounds
        plot_equity_curve(prob_train, train_df, "Train", out_dir=exp_dir,
                             q_star=q_star)
        plot_equity_curve(prob_val,   val_df,   "Val",   out_dir=exp_dir,
                             rolling_median=med_val, lower_bound=lb_val, upper_bound=ub_val,
                             q_star=q_star)

        # Trading signal overlays — variant A (baseline) and C (3-state) per split
        plot_trading_signals(train_df, prob_train, "Train", out_dir=exp_dir, variant="A",
                             q_star=q_star)
        plot_trading_signals(train_df, prob_train, "Train", out_dir=exp_dir, variant="C",
                             q_star=q_star)
        plot_trading_signals(val_df,   prob_val,   "Val",   out_dir=exp_dir, variant="A",
                             q_star=q_star)
        plot_trading_signals(val_df,   prob_val,   "Val",   out_dir=exp_dir, variant="B",
                             rolling_median=med_val, q_star=q_star)
        plot_trading_signals(val_df,   prob_val,   "Val",   out_dir=exp_dir, variant="C",
                             lower_bound=lb_val, upper_bound=ub_val, q_star=q_star)

        if prob_test is not None:
            plot_prob_distribution(prob_test, "Test", out_dir=exp_dir)
            plot_equity_curve(prob_test, test_df, "Test", out_dir=exp_dir,
                                 rolling_median=med_test, lower_bound=lb_test, upper_bound=ub_test,
                                 q_star=q_star)
            plot_trading_signals(test_df, prob_test, "Test", out_dir=exp_dir, variant="A",
                                 q_star=q_star)
            plot_trading_signals(test_df, prob_test, "Test", out_dir=exp_dir, variant="B",
                                 rolling_median=med_test, q_star=q_star)
            plot_trading_signals(test_df, prob_test, "Test", out_dir=exp_dir, variant="C",
                                 lower_bound=lb_test, upper_bound=ub_test, q_star=q_star)

        plot_threshold_sensitivity(y_val_aligned, prob_val, out_dir=exp_dir)

        # ── STEP F8b: Advanced tear sheet plots ───────────────────────────────
        log.info("[STEP 8b] Generating advanced tear sheet visualizations ...")

        # 1. Per-variant confusion matrices
        # Train: Variant A only
        plot_variant_confusion_matrix(
            tr_signals["A"], tr_btc_ret, "Train", "A", out_dir=exp_dir,
        )
        # Val: all 4 variants
        for _v in ("A", "B", "C", "D"):
            plot_variant_confusion_matrix(
                vl_signals[_v], vl_btc_ret, "Val", _v, out_dir=exp_dir,
            )
        # Test: all 4 variants
        if ts_signals is not None:
            for _v in ("A", "B", "C", "D"):
                plot_variant_confusion_matrix(
                    ts_signals[_v], ts_btc_ret, "Test", _v, out_dir=exp_dir,
                )

        # 2. Drawdown (underwater) curves
        plot_drawdown_curves(
            tr_btc_ret, tr_returns, "Train", tr_dates, out_dir=exp_dir, q_star=q_star,
        )
        plot_drawdown_curves(
            vl_btc_ret, vl_returns, "Val", vl_dates, out_dir=exp_dir, q_star=q_star,
        )
        if ts_returns is not None:
            plot_drawdown_curves(
                ts_btc_ret, ts_returns, "Test", ts_dates, out_dir=exp_dir, q_star=q_star,
            )

        # 3. Return distribution histograms
        plot_return_distributions(
            tr_btc_ret, tr_returns, "Train", out_dir=exp_dir, q_star=q_star,
        )
        plot_return_distributions(
            vl_btc_ret, vl_returns, "Val", out_dir=exp_dir, q_star=q_star,
        )
        if ts_returns is not None:
            plot_return_distributions(
                ts_btc_ret, ts_returns, "Test", out_dir=exp_dir, q_star=q_star,
            )

        # 4a. Rolling 30-day Sharpe ratio
        plot_rolling_sharpe(
            tr_btc_ret, tr_returns, "Train", tr_dates, out_dir=exp_dir, q_star=q_star,
        )
        plot_rolling_sharpe(
            vl_btc_ret, vl_returns, "Val", vl_dates, out_dir=exp_dir, q_star=q_star,
        )
        if ts_returns is not None:
            plot_rolling_sharpe(
                ts_btc_ret, ts_returns, "Test", ts_dates, out_dir=exp_dir, q_star=q_star,
            )

        # 4b. Monthly returns heatmap — Variant D (most thesis-relevant)
        plot_monthly_returns_heatmap(
            vl_returns["D"], vl_dates, "Val", "D", out_dir=exp_dir, q_star=q_star,
        )
        if ts_returns is not None:
            plot_monthly_returns_heatmap(
                ts_returns["D"], ts_dates, "Test", "D", out_dir=exp_dir, q_star=q_star,
            )

        # ── STEP F9: Permutation Feature Importance ────────────────────────────
        log.info("[STEP 9] Computing permutation feature importance (Val) ...")
        real_feat_cols = [
            c for c in train_df.columns
            if c != "target" and c not in PRUNED_FEATURES
        ]
        plot_feature_importance(
            model, X_val, y_val, y_val_aligned, prob_val, real_feat_cols,
            n_repeats=10, out_dir=exp_dir,
        )

        # ── STEP F10: CSV export (test set only) + Prediction Excel (all splits)
        if test_df is not None and prob_test is not None:
            export_predictions_to_csv(test_df, prob_test, SEQ_LEN, exp_dir,
                                      rolling_median=med_test, lower_bound=lb_test, upper_bound=ub_test)
            export_sanity_check_csv(test_df, prob_test, SEQ_LEN, exp_dir)
        export_simple_excel(train_df, prob_train, SEQ_LEN, exp_dir, "Train")
        export_simple_excel(val_df,   prob_val,   SEQ_LEN, exp_dir, "Val",  rolling_median=med_val)
        if test_df is not None and prob_test is not None:
            export_simple_excel(test_df, prob_test, SEQ_LEN, exp_dir, "Test", rolling_median=med_test)

        # ── STEP F11: Success banner ───────────────────────────────────────────
        log.info("Experiment '%s' complete. All outputs → %s", exp_name, exp_dir)
        print(f"\n{'═' * 65}")
        print(f"  ✓ {exp_name.upper()} experiment complete")
        print(f"  → {exp_dir}")
        print(f"{'═' * 65}\n")

    finally:
        log.removeHandler(exp_fh)
        exp_fh.close()


# =============================================================================
# MAIN — Orchestrate all pipeline steps
# =============================================================================

def main() -> None:
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    log.info("=" * 65)
    log.info("  02_lstm_model.py  —  DUAL-RUN PIPELINE START")
    log.info("  RUN_NAME: %s  |  BASE_OUT_DIR: %s", RUN_NAME, BASE_OUT_DIR)
    log.info("  Train : %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val   : %s → %s", VAL_START,   VAL_END)
    log.info("  Test  : %s → %s  [LOCKED=%s]",
             TEST_START, TEST_END, not EVALUATE_ON_TEST)
    log.info("  SEQ_LEN=%d  |  Pruned=%s", SEQ_LEN, sorted(PRUNED_FEATURES))
    log.info("=" * 65)

    # ── STEP 1: Load & Split (once — shared by both experiments) ─────────────
    log.info("[STEP 1] Loading and splitting data ...")
    train_df, val_df, test_df = load_and_split(INPUT_CSV)

    # ── STEP 2: Scale (once — scaler fitted identically for both runs) ────────
    log.info("[STEP 2] Scaling features (fit on Train only) ...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_cols) = scale_features(train_df, val_df, test_df)
    log.info("Feature count after pruning: %d", len(feature_cols))

    # ── RUN 1: Unweighted ─────────────────────────────────────────────────────
    log.info("=== STARTING UNWEIGHTED EXPERIMENT ===")
    run_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_df, val_df, test_df,
        use_weights=False, exp_name="unweighted", base_dir=BASE_OUT_DIR,
    )

    # ── RUN 2: Weighted ───────────────────────────────────────────────────────
    log.info("=== STARTING WEIGHTED EXPERIMENT ===")
    run_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_df, val_df, test_df,
        use_weights=True, exp_name="weighted", base_dir=BASE_OUT_DIR,
    )

    log.info("Both experiments complete. All outputs in: %s", BASE_OUT_DIR)
    log.info("02_lstm_model.py  —  DUAL-RUN PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
