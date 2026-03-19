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
#          plot_equity_curve_v2() computes and annotates four strategies:
#            • BTC Buy & Hold  (passive crypto benchmark)
#            • SPY Buy & Hold  (passive equities benchmark — second benchmark)
#            • Variant B  (3-state binary filter)
#            • Variant C  (dynamic position sizing)
#          Each strategy is annotated with three quant metrics:
#            • Annualised Sharpe Ratio  (rf = 0, 252-day basis)
#            • Annualised Volatility    (252-day basis)
#            • Maximum Drawdown        (peak-to-trough from running maximum)
#
# PIPELINE STEPS:
#   1. Load & Split  – Load v2 CSV; filter by date boundaries.
#   2. Scale         – StandardScaler fitted ONLY on Train (no leakage).
#   3. Window        – tf.keras.utils.timeseries_dataset_from_array.
#   4. Build & Train – 2-layer LSTM: L2 + Dropout + Recurrent Dropout.
#   5. Evaluate A    – Standard 0.5 threshold: Acc, Sens, Spec, AUC, Gini.
#   6. Evaluate B    – 3-state logic: Coverage + Conditional Win Rate.
#   7. Evaluate C    – Dynamic sizing: Weighted Return + quant metrics.
#   8. Visualize     – Training history, ROC, confusion matrices.
#   9. Advanced Plots:
#        • Probability distribution (Train, Val, [Test])
#        • Equity curves v2 with SPY + Variant C + Sharpe/Vol/MDD annotation
#        • Threshold sensitivity analysis (Val only)
#        • Permutation Feature Importance (Val only)
#
# OUTPUTS:
#   • training_history_v2.png
#   • roc_curve_v2_[val|test].png
#   • confusion_matrix_v2_[val|test].png
#   • prob_distribution_v2_[train|val|test].png
#   • equity_curve_v2_[train|val|test].png   ← SPY B&H + Variant C + metrics
#   • threshold_sensitivity_v2.png
#   • feature_importance_v2_val.png
#
# REQUIREMENTS:
#   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import sys

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
TRAIN_END:   str = "2023-12-31"
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"
TEST_START:  str = "2025-01-01"
TEST_END:    str = "2025-12-31"

# Test set toggle — same semantics as v1.
# False during all tuning phases. True only for the final reported run.
EVALUATE_ON_TEST: bool = True

# ── [V2-2] Sequence length ────────────────────────────────────────────────────
SEQ_LEN: int = 30   # 30 trading days

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 64
EPOCHS:     int = 150

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_UNITS_1:  int   = 512
LSTM_UNITS_2:  int   = 256
DROPOUT_RATE:  float = 0.25
L2_FACTOR:     float = 0.002
RECURRENT_DROPOUT_RATE: float = 0.25

# ── 3-state / dynamic-sizing confidence thresholds ───────────────────────────
UPPER_THRESHOLD: float = 0.52
LOWER_THRESHOLD: float = 0.48

# ── [V2-3] Variant C weight bounds ───────────────────────────────────────────
# Position weight at the band edge (minimum non-zero exposure).
# Position weight at maximum conviction (full exposure).
VC_MIN_WEIGHT: float = 0.25
VC_MAX_WEIGHT: float = 1

# ── Feature pruning ───────────────────────────────────────────────────────────
# gold_log_return and nvda_log_return showed negative permutation importance
# (permuting them improves val AUC) → toxic features, removed before scaling.
# MUST match 03_hyperparameter_tuning_v2.py exactly for reproducibility.
PRUNED_FEATURES: frozenset = frozenset({"gold_log_return", "nvda_log_return"})

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV: str = "processed_dataset_v2.csv"

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
        logging.FileHandler("run_log_v2.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — DATA LOADING & DATE-BASED SPLITTING
# =============================================================================

def load_and_split(csv_path: str) -> tuple:
    """
    Load the v2 processed CSV and partition it into strictly chronological
    subsets using exact calendar date boundaries.

    Chronological wall:
        TRAIN [2015–2023] → VAL [2024] → TEST [2025]
                          ▲           ▲
                 Hyperparameter    Reported
                   selection       results (locked behind EVALUATE_ON_TEST)

    Returns (train_df, val_df, test_df).  test_df is None if EVALUATE_ON_TEST=False.
    Raw DataFrames (pre-scaling) are returned so plot_equity_curve_v2() can
    access btc_log_return values and spy_log_return values directly.
    """
    log.info("Loading v2 dataset from '%s' ...", csv_path)
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)

    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df.index.min().date(), df.index.max().date(),
    )

    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    val_mask   = (df.index >= VAL_START)   & (df.index <= VAL_END)

    train_df = df[train_mask]
    val_df   = df[val_mask]

    for name, split in [("Train", train_df), ("Val", val_df)]:
        log.info(
            "  %-5s → %5d rows  [%s → %s]",
            name, len(split),
            split.index.min().date(), split.index.max().date(),
        )

    if EVALUATE_ON_TEST:
        test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
        test_df   = df[test_mask]
        log.info(
            "  %-5s → %5d rows  [%s → %s]",
            "Test", len(test_df),
            test_df.index.min().date(), test_df.index.max().date(),
        )
    else:
        test_df = None
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

    If USE_RAW_PRICES=True was used to generate the v2 CSV, the column
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
    X:          np.ndarray,
    y:          np.ndarray,
    seq_len:    int,
    batch_size: int,
    shuffle:    bool = False,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows.

    Window ↔ Target alignment (no look-ahead bias):
        Window i = features[i : i+L]      (L = seq_len)
        Target i = y[i + L - 1]           (direction of day i + L)

    Returns (dataset, y_aligned) where y_aligned = y[seq_len-1:].
    """
    N         = len(X)
    n_windows = N - seq_len + 1

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

    y_aligned = y[seq_len - 1:].astype(np.float32)
    return dataset, y_aligned


# =============================================================================
# STEP 4 — MODEL ARCHITECTURE
# =============================================================================

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Two-layer stacked LSTM with four anti-overfitting defences:
    L2 weight decay, inter-layer Dropout, Recurrent Dropout, EarlyStopping.
    """
    model = Sequential(
        [
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                kernel_regularizer=L2Reg(L2_FACTOR),
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                input_shape=(seq_len, n_features),
                name="lstm_layer_1",
            ),
            Dropout(DROPOUT_RATE, name="dropout_1"),
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                kernel_regularizer=L2Reg(L2_FACTOR),
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                name="lstm_layer_2",
            ),
            Dropout(DROPOUT_RATE, name="dropout_2"),
            Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="btc_direction_lstm_v2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss="binary_crossentropy",
        metrics=["accuracy"],
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
) -> tf.keras.callbacks.History:
    """
    Train with EarlyStopping(patience=7) and ReduceLROnPlateau(patience=5).
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    log.info(
        "Training complete. Best val_loss = %.5f at epoch %d / %d.",
        min(history.history["val_loss"]),
        best_epoch,
        len(history.history["val_loss"]),
    )
    return history


# =============================================================================
# STEP 6A — VARIANT A: STANDARD 0.5 THRESHOLD
# =============================================================================

def evaluate_standard(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
    df:         pd.DataFrame | None = None,
) -> dict:
    """Standard 0.5-threshold evaluation: Acc, Sensitivity, Specificity, AUC, Gini.

    If df is provided, also computes Variant A financial performance metrics
    (Sharpe, Volatility, MaxDD) using the same N−SEQ_LEN alignment as
    evaluate_variant_c.  Strategy: signal = +1 / −1 on every day (always in market).
    """
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc         = roc_auc_score(y_true, y_prob)
    gini        = 2.0 * auc - 1.0

    print(f"\n{'═' * 58}")
    print(f"  Variant A — Standard Evaluation  [{split_name} Set]")
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
        m            = _compute_quant_metrics(daily_ret_a)
        total_log_ret = float(np.sum(daily_ret_a))
        print(f"{'─' * 58}")
        print(f"  [Financial Performance — always in market, ±full size]")
        print(f"  Total log ret:  {total_log_ret:+.4f}")
        print(f"  Sharpe (ann):   {m['sharpe']:+.3f}   (rf=0, 252-day basis)")
        print(f"  Volatility  :   {m['volatility']:.3f}   (annualised)")
        print(f"  Max Drawdown:  {m['max_drawdown']:.3f}   ({m['max_drawdown']*100:.1f}%)")
    print(f"{'═' * 58}\n")

    return dict(accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, auc=auc, gini=gini)


# =============================================================================
# STEP 6B — VARIANT B: 3-STATE CONFIDENCE FILTER
# =============================================================================

def evaluate_3state(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
    df:         pd.DataFrame | None = None,
) -> dict:
    """
    3-state confidence filter: BUY if P>0.55, SELL if P<0.45, HOLD otherwise.
    Metrics reported on traded days only: Coverage + Conditional Win Rate.

    If df is provided, also computes Variant B financial performance metrics
    (Sharpe, Volatility, MaxDD). HOLD days contribute zero return.
    """
    trade_mask = (y_prob > UPPER_THRESHOLD) | (y_prob < LOWER_THRESHOLD)
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

    print(f"{'═' * 58}")
    print(f"  Variant B — 3-State Evaluation  [{split_name} Set]")
    print(f"  Thresholds:  UPPER={UPPER_THRESHOLD:.2f}  |  LOWER={LOWER_THRESHOLD:.2f}")
    print(f"{'─' * 58}")
    print(f"  Total days    :  {n_total}")
    print(f"  Traded days   :  {n_traded}  ({coverage * 100:.1f}% coverage)")
    print(f"  Hold days     :  {n_total - n_traded}  ({(1 - coverage) * 100:.1f}%)")
    print(f"  Cond. Win Rate:  {cond_win_rate:.4f}  (accuracy on traded days)")

    if df is not None:
        n_tradeable  = len(df) - SEQ_LEN
        prob_aligned = y_prob[:n_tradeable]
        btc_ret      = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
        signal_b     = np.where(
            prob_aligned > UPPER_THRESHOLD, 1.0,
            np.where(prob_aligned < LOWER_THRESHOLD, -1.0, 0.0),
        )
        daily_ret_b   = signal_b * btc_ret
        m             = _compute_quant_metrics(daily_ret_b)
        total_log_ret = float(np.sum(daily_ret_b))
        print(f"{'─' * 58}")
        print(f"  [Financial Performance — hold days contribute zero return]")
        print(f"  Total log ret:  {total_log_ret:+.4f}")
        print(f"  Sharpe (ann):   {m['sharpe']:+.3f}   (rf=0, 252-day basis)")
        print(f"  Volatility  :   {m['volatility']:.3f}   (annualised)")
        print(f"  Max Drawdown:  {m['max_drawdown']:.3f}   ({m['max_drawdown']*100:.1f}%)")
    print(f"{'═' * 58}\n")

    return dict(coverage=coverage, conditional_win_rate=cond_win_rate, n_traded=n_traded)


# =============================================================================
# STEP 6C — VARIANT C: DYNAMIC POSITION SIZING
# =============================================================================

def _compute_variant_c_weights(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Variant C signals and continuous position weights.

    [V2-3] Weight formula:
        HOLD zone  [0.45, 0.55]  →  signal=0,  weight=0.0
        Long       P > 0.55      →  signal=+1, weight = VC_MIN + (VC_MAX-VC_MIN)
                                                         × (P - 0.55) / (1.0 - 0.55)
        Short      P < 0.45      →  signal=-1, weight = VC_MIN + (VC_MAX-VC_MIN)
                                                         × (0.45 - P) / 0.45

    At the band edge (P=0.55/0.45): weight = VC_MIN_WEIGHT = 0.25.
    At max conviction  (P=1.0/0.0): weight = VC_MAX_WEIGHT = 1.00.
    The position changes continuously and proportionally with confidence,
    reducing full-size trades when the model is only marginally confident.

    Returns (signals, weights): both np.ndarray of shape (N,).
    """
    signals = np.zeros(len(probs), dtype=np.float32)
    weights = np.zeros(len(probs), dtype=np.float32)

    long_mask  = probs > UPPER_THRESHOLD
    short_mask = probs < LOWER_THRESHOLD

    signals[long_mask]  = 1.0
    signals[short_mask] = -1.0

    weight_range = VC_MAX_WEIGHT - VC_MIN_WEIGHT

    # Long weight: linearly from VC_MIN at P=0.55 to VC_MAX at P=1.0
    weights[long_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (probs[long_mask] - UPPER_THRESHOLD)
        / (1.0 - UPPER_THRESHOLD)
    )

    # Short weight: linearly from VC_MIN at P=0.45 to VC_MAX at P=0.0
    weights[short_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (LOWER_THRESHOLD - probs[short_mask])
        / LOWER_THRESHOLD
    )

    return signals, weights


def evaluate_variant_c(
    df:         pd.DataFrame,
    prob:       np.ndarray,
    split_name: str,
) -> dict:
    """
    [V2-3] Evaluate Variant C dynamic position sizing.

    Alignment fix
    ─────────────
    make_tf_dataset produces N − SEQ_LEN + 1 windows, so model.predict()
    returns N − SEQ_LEN + 1 probabilities. The realized return for window j
    lives at df row j + SEQ_LEN, so only N − SEQ_LEN returns are available.
    The last prediction is therefore trimmed before use:

        n_tradeable  = len(df) − SEQ_LEN
        prob_aligned = prob[:n_tradeable]           shape (N−SEQ_LEN,)
        btc_returns  = df["btc_log_return"][SEQ_LEN : SEQ_LEN + n_tradeable]

    This mirrors the alignment used in plot_equity_curve_v2().

    Metrics (on all days; HOLD days contribute zero return):
        Total log return     : sum of weighted daily returns
        Annualised Sharpe    : mean(daily_ret) / std(daily_ret) × √252  (rf=0)
        Annualised Volatility: std(daily_ret) × √252
        Maximum Drawdown     : max(1 − equity_t / running_max)
    """
    n_tradeable  = len(df) - SEQ_LEN
    prob_aligned = prob[:n_tradeable]
    btc_returns  = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]

    signals, weights = _compute_variant_c_weights(prob_aligned)
    daily_ret = signals * weights * btc_returns

    n_long  = int((signals > 0).sum())
    n_short = int((signals < 0).sum())
    n_hold  = int((signals == 0).sum())

    metrics = _compute_quant_metrics(daily_ret)
    total_log_ret = float(np.sum(daily_ret))

    print(f"{'═' * 58}")
    print(f"  Variant C — Dynamic Sizing  [{split_name} Set]")
    print(f"  HOLD zone [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}],  "
          f"weight ∈ [{VC_MIN_WEIGHT:.2f}, {VC_MAX_WEIGHT:.2f}]")
    print(f"{'─' * 58}")
    print(f"  Long days   :  {n_long}  |  Short days: {n_short}  |  Hold: {n_hold}")
    print(f"  Total log ret:  {total_log_ret:+.4f}")
    print(f"  Sharpe (ann):   {metrics['sharpe']:+.3f}   (rf=0, 252-day basis)")
    print(f"  Volatility  :   {metrics['volatility']:.3f}   (annualised)")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.3f}   ({metrics['max_drawdown']*100:.1f}%)")
    print(f"{'═' * 58}\n")

    return {**metrics, "total_log_ret": total_log_ret,
            "n_long": n_long, "n_short": n_short, "n_hold": n_hold}


# =============================================================================
# ADVANCED QUANT METRICS HELPER
# =============================================================================

def _compute_quant_metrics(daily_returns: np.ndarray, trading_days: int = 252) -> dict:
    """
    [V2-4] Compute annualised Sharpe, annualised volatility, and maximum
    drawdown for an array of daily log-returns.

    Parameters
    ----------
    daily_returns : array of per-day log-returns (signed, based on position).
    trading_days  : annualisation factor. 252 is standard (US equity basis);
                    crypto trades 365 days, but 252 is used here for
                    comparability with SPY and academic benchmarks.

    Returns
    -------
    dict with keys:
        sharpe       (float) – annualised Sharpe ratio (rf=0)
        volatility   (float) – annualised standard deviation
        max_drawdown (float) – maximum peak-to-trough fractional loss [0, 1]
    """
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    mean_ret = float(np.mean(daily_returns))
    std_ret  = float(np.std(daily_returns, ddof=1))

    sharpe     = (mean_ret / std_ret * np.sqrt(trading_days)) if std_ret > 1e-12 else 0.0
    volatility = std_ret * np.sqrt(trading_days)

    # Maximum Drawdown: peak-to-trough from a running maximum of the
    # portfolio equity curve (initialised at 1.0).
    equity      = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(equity)
    drawdowns   = 1.0 - equity / running_max
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "sharpe":       round(float(sharpe), 3),
        "volatility":   round(float(volatility), 3),
        "max_drawdown": round(float(max_drawdown), 3),
    }


# =============================================================================
# STEP 7 — DIAGNOSTIC VISUALIZATIONS
# =============================================================================

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save 2-panel BCE loss + accuracy figure for Train/Val splits."""
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"LSTM Training History v2 — BTC Daily Direction Classifier\n"
        f"Architecture: {LSTM_UNITS_1}→{LSTM_UNITS_2} units | "
        f"L2={L2_FACTOR} | Dropout={DROPOUT_RATE} | "
        f"RecDrop={RECURRENT_DROPOUT_RATE} | SeqLen={SEQ_LEN}",
        fontsize=11,
    )

    axes[0].plot(epochs_range, history.history["loss"],
                 lw=2, color="steelblue", label="Train Loss")
    axes[0].plot(epochs_range, history.history["val_loss"],
                 lw=2, color="tomato", ls="--", label="Validation Loss")
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history.history["accuracy"],
                 lw=2, color="steelblue", label="Train Accuracy")
    axes[1].plot(epochs_range, history.history["val_accuracy"],
                 lw=2, color="tomato", ls="--", label="Validation Accuracy")
    axes[1].set_title("Classification Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("training_history_v2.png", dpi=150)
    plt.close(fig)
    log.info("Training history saved → training_history_v2.png")


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auc_score: float, split_name: str = "Val"
) -> None:
    """ROC curve with AUC and Gini annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    gini = 2.0 * auc_score - 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="steelblue",
            label=f"LSTM  (AUC = {auc_score:.4f}  |  Gini = {gini:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier  (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
    ax.set_title(f"ROC Curve v2 — {split_name} Set", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_v2_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("ROC curve saved → %s", fname)


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, split_name: str
) -> None:
    """Seaborn confusion matrix with raw counts + row-normalised percentages."""
    y_pred  = (y_prob >= 0.5).astype(int)
    cm      = confusion_matrix(y_true.astype(int), y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annots = np.array(
        [[f"{cnt}\n({pct:.1%})" for cnt, pct in zip(row_c, row_p)]
         for row_c, row_p in zip(cm, cm_norm)]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=annots, fmt="", cmap="Blues",
        vmin=0.0, vmax=1.0, linewidths=0.6,
        xticklabels=["Pred: DOWN (0)", "Pred: UP (1)"],
        yticklabels=["True: DOWN (0)", "True: UP (1)"],
        annot_kws={"size": 12}, ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix v2 — {split_name} Set\n"
        "(Threshold = 0.50 | Colour = Row-Normalised Rate)",
        fontsize=12,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fname = f"confusion_matrix_v2_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", fname)


# =============================================================================
# STEP 8 — ADVANCED QUANTITATIVE FINANCE VISUALIZATIONS
# =============================================================================

def plot_prob_distribution(y_prob: np.ndarray, split_name: str) -> None:
    """Probability distribution KDE with Variant B zone annotations."""
    n_total = len(y_prob)
    n_short = int((y_prob <  LOWER_THRESHOLD).sum())
    n_noise = int(((y_prob >= LOWER_THRESHOLD) & (y_prob <= UPPER_THRESHOLD)).sum())
    n_long  = int((y_prob >  UPPER_THRESHOLD).sum())

    pct_short = 100.0 * n_short / n_total
    pct_noise = 100.0 * n_noise / n_total
    pct_long  = 100.0 * n_long  / n_total

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(
        LOWER_THRESHOLD, UPPER_THRESHOLD,
        color="lightgrey", alpha=0.70, zorder=0,
        label=(f"Noise Zone  [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]  "
               f"→  {pct_noise:.1f}% of {split_name} predictions"),
    )
    sns.histplot(
        y_prob, bins=40, kde=True, stat="density",
        color="steelblue", alpha=0.55, edgecolor="white", linewidth=0.4,
        label=f"LSTM Output Distribution — {split_name} Set",
        ax=ax, zorder=1,
    )
    ax.axvline(LOWER_THRESHOLD, color="crimson",  lw=2.0, ls="--", zorder=2,
               label=f"Lower Threshold = {LOWER_THRESHOLD:.2f}")
    ax.axvline(0.50,            color="black",    lw=1.8, ls=":", zorder=2,
               label="Decision Boundary = 0.50  (Variant A)")
    ax.axvline(UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
               label=f"Upper Threshold = {UPPER_THRESHOLD:.2f}")

    ymax  = ax.get_ylim()[1]
    box_y = ymax * 0.75
    for x_pos, text, color in [
        (LOWER_THRESHOLD / 2,
         f"SHORT zone\n{pct_short:.1f}%  ({n_short} days)", "crimson"),
        ((LOWER_THRESHOLD + UPPER_THRESHOLD) / 2,
         f"HOLD zone\n{pct_noise:.1f}%  ({n_noise} days)",  "dimgrey"),
        (UPPER_THRESHOLD + (1.0 - UPPER_THRESHOLD) / 2,
         f"LONG zone\n{pct_long:.1f}%  ({n_long} days)",    "seagreen"),
    ]:
        ax.text(
            x_pos, box_y, text, ha="center", va="top", fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, alpha=0.80, lw=1.2),
        )

    ax.set_xlabel("Predicted Probability  P(BTC closes higher tomorrow)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Probability Distribution of LSTM Output v2 — {split_name} Set\n"
        f"Grey band = Noise Zone [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]",
        fontsize=13, pad=14,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.30)
    plt.tight_layout()
    fname = f"prob_distribution_v2_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Probability distribution saved → %s", fname)


def _plot_single_equity_curve(
    variant_label:     str,
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
    pct_variant = (np.exp(np.cumsum(variant_daily_ret)) - 1.0) * 100.0
    pct_btc     = (np.exp(np.cumsum(btc_ret))           - 1.0) * 100.0
    m_variant   = _compute_quant_metrics(variant_daily_ret)
    m_btc       = _compute_quant_metrics(btc_ret)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        dates, pct_btc, lw=2.0, color="dimgrey", ls="-.", zorder=2,
        label=(f"BTC Buy & Hold  "
               f"(Sharpe: {m_btc['sharpe']:+.2f})  →  {pct_btc[-1]:+.1f}%"),
    )

    if has_spy and spy_ret is not None:
        pct_spy = (np.exp(np.cumsum(spy_ret)) - 1.0) * 100.0
        m_spy   = _compute_quant_metrics(spy_ret)
        ax.plot(
            dates, pct_spy, lw=1.8, color="olivedrab", ls="-.", zorder=2,
            label=(f"SPY Buy & Hold  "
                   f"(Sharpe: {m_spy['sharpe']:+.2f})  →  {pct_spy[-1]:+.1f}%"),
        )

    ax.plot(
        dates, pct_variant, lw=2.5, color=variant_color, ls="-", zorder=4,
        label=(f"{variant_label}  "
               f"(Sharpe: {m_variant['sharpe']:+.2f})  →  {pct_variant[-1]:+.1f}%"),
    )

    ax.fill_between(dates, 0, pct_btc, alpha=0.04, color="dimgrey", zorder=1)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.35)

    # ── Metrics annotation table ───────────────────────────────────────────────
    col_hdr  = f"{'Strategy':<20} {'Tot Ret':>8}  {'Sharpe':>7}  {'Vol':>6}  {'MDD':>6}"
    sep      = "─" * len(col_hdr)
    def _row(lbl: str, m: dict, pct: float) -> str:
        return (f"{lbl:<20} {pct:>+7.1f}%  "
                f"{m['sharpe']:>+7.3f}  {m['volatility']:>6.3f}  "
                f"{m['max_drawdown']*100:>5.1f}%")

    rows = [col_hdr, sep, _row("BTC B&H", m_btc, pct_btc[-1])]
    if has_spy and spy_ret is not None:
        rows.append(_row("SPY B&H", m_spy, pct_spy[-1]))
    rows.append(_row(variant_label, m_variant, pct_variant[-1]))

    ax.text(
        0.01, 0.02, "\n".join(rows),
        transform=ax.transAxes, fontsize=7.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="grey", alpha=0.88, lw=0.8),
        fontfamily="monospace",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8.5)

    ax.set_xlabel("Date  (day the return is realised)", fontsize=12)
    ax.set_ylabel("Cumulative Return  (%)", fontsize=12)
    ax.set_title(
        f"Simulated Equity Curve v2 — {split_name} Set  ·  {variant_label}\n"
        f"vs. {'SPY B&H  ·  ' if has_spy else ''}BTC B&H  |  rf=0, 252-day Sharpe\n"
        f"Period: {str(dates[0].date())}  →  {str(dates[-1].date())}",
        fontsize=12, pad=14,
    )
    ax.legend(fontsize=10, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.30)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Equity curve saved → %s", fname)


def plot_equity_curve_v2(
    prob:       np.ndarray,
    df:         pd.DataFrame,
    split_name: str,
) -> None:
    """
    [V2-4] Generate THREE separate equity curve PNGs for a given split:

        equity_curve_v2_{split_name}_VariantA.png  (always-in-market ±1 binary)
        equity_curve_v2_{split_name}_VariantB.png  (3-state confidence filter)
        equity_curve_v2_{split_name}_VariantC.png  (dynamic position sizing)

    Each plot shows:
        • The specific variant's equity curve
        • BTC Buy & Hold benchmark
        • SPY Buy & Hold benchmark (omitted if 'spy_log_return' not in df)

    Legend entries include Annualised Sharpe and Cumulative Return:
        "Variant C  (Sharpe: +0.85)  →  +24.5%"

    Prediction-to-return alignment:
        n_tradeable  = len(df) − SEQ_LEN
        prob_aligned = prob[:n_tradeable]
        btc_ret      = df["btc_log_return"][SEQ_LEN : SEQ_LEN + n_tradeable]
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
    # Variant A: always in market at full size (±1)
    signal_a = np.where(prob_aligned >= 0.5, 1.0, -1.0)
    ret_a    = signal_a * btc_ret

    # Variant B: 3-state confidence filter, HOLD zone → zero return
    signal_b = np.where(
        prob_aligned > UPPER_THRESHOLD, 1.0,
        np.where(prob_aligned < LOWER_THRESHOLD, -1.0, 0.0),
    )
    ret_b = signal_b * btc_ret

    # Variant C: dynamic position sizing
    signal_c, weight_c = _compute_variant_c_weights(prob_aligned)
    ret_c = signal_c * weight_c * btc_ret

    sname = split_name.lower()
    _plot_single_equity_curve(
        f"Variant A  (0.50 threshold)",
        ret_a, btc_ret, spy_ret, has_spy, dates, split_name,
        "steelblue",
        f"equity_curve_v2_{sname}_VariantA.png",
    )
    _plot_single_equity_curve(
        f"Variant B  (±{UPPER_THRESHOLD:.2f} filter)",
        ret_b, btc_ret, spy_ret, has_spy, dates, split_name,
        "tomato",
        f"equity_curve_v2_{sname}_VariantB.png",
    )
    _plot_single_equity_curve(
        f"Variant C  (dyn. sizing ±{UPPER_THRESHOLD:.2f})",
        ret_c, btc_ret, spy_ret, has_spy, dates, split_name,
        "darkorange",
        f"equity_curve_v2_{sname}_VariantC.png",
    )


def plot_threshold_sensitivity(
    y_true_val: np.ndarray, prob_val: np.ndarray
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
        "Threshold Sensitivity Analysis v2 — Validation Set\n"
        "Blue: Conditional Win Rate  |  Red: Coverage  |  Green: selected threshold",
        fontsize=11, pad=14,
    )
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(ax1_handles + [line_cov], ax1_labels + [line_cov.get_label()],
               fontsize=9, loc="best", framealpha=0.92)
    ax1.grid(axis="both", alpha=0.28)
    plt.tight_layout()
    fig.savefig("threshold_sensitivity_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Threshold sensitivity saved → threshold_sensitivity_v2.png")


# =============================================================================
# STEP 9 — PERMUTATION FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    model:             tf.keras.Model,
    X_val:             np.ndarray,
    y_val:             np.ndarray,
    y_val_aligned:     np.ndarray,
    prob_val_baseline: np.ndarray,
    feature_cols:      list,
    n_repeats:         int = 3,
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
        "btc_log_return":  "BTC Return",
        "spy_log_return":  "SPY Return",
        "gold_log_return": "Gold Return",
        "nvda_log_return": "NVDA Return",
        "dxy_log_return":  "DXY Return",
        "vix_log_return":  "VIX Return",
        "vix_level":       "VIX Level",
        "btc_volume_ratio":"BTC Vol. Ratio",
        "btc_rsi_14":      "BTC RSI(14)",
        "btc_roll_vol_21": "BTC RollVol(21d)",
        "is_weekend":      "Is Weekend",
        # Raw-price mode aliases (if present in feature_cols)
        "btc_close":       "BTC Price",
        "spy_close":       "SPY Price",
        "gold_close":      "Gold Price",
        "nvda_close":      "NVDA Price",
        "dxy_close":       "DXY Price",
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
        importances.append((col, mean_drop))
        log.info(
            "  Feature %-22s  AUC drop = %+.4f (mean over %d repeats)",
            col, mean_drop, n_repeats,
        )

    importances.sort(key=lambda x: x[1], reverse=True)
    labels = [label_map.get(c, c) for c, _ in importances]
    values = [v for _, v in importances]
    colors = ["steelblue" if v >= 0 else "tomato" for v in values]

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)
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
        f"AUC Drop  (baseline − permuted)  |  n_repeats={n_repeats}\n"
        "Blue = positive contribution (important)  |  Red = toxic (AUC drops when kept)",
        fontsize=10,
    )
    ax.set_title(
        f"Permutation Feature Importance v2 — Validation Set\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  SEQ_LEN={SEQ_LEN}",
        fontsize=12, pad=14,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.30)
    plt.tight_layout()
    fig.savefig("feature_importance_v2_val.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance saved → feature_importance_v2_val.png")


# =============================================================================
# MAIN — Orchestrate all pipeline steps
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("  02_lstm_model_v2.py  —  PIPELINE START")
    log.info("  Train : %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val   : %s → %s", VAL_START,   VAL_END)
    log.info("  Test  : %s → %s  [LOCKED=%s]",
             TEST_START, TEST_END, not EVALUATE_ON_TEST)
    log.info("  SEQ_LEN=%d  |  Pruned=%s", SEQ_LEN, sorted(PRUNED_FEATURES))
    log.info("=" * 65)

    # ── STEP 1: Load & Split ──────────────────────────────────────────────────
    log.info("[STEP 1] Loading and splitting data ...")
    train_df, val_df, test_df = load_and_split(INPUT_CSV)

    # ── STEP 2: Scale ─────────────────────────────────────────────────────────
    log.info("[STEP 2] Scaling features (fit on Train only) ...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_cols) = scale_features(train_df, val_df, test_df)

    n_features = len(feature_cols)
    log.info("Feature count after pruning: %d", n_features)

    # ── STEP 3: Build windowed datasets ──────────────────────────────────────
    log.info("[STEP 3] Building sliding-window tf.data.Datasets ...")
    train_ds, y_train_aligned = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=True
    )
    val_ds, y_val_aligned = make_tf_dataset(
        X_val, y_val, SEQ_LEN, BATCH_SIZE, shuffle=False
    )

    if X_test is not None:
        test_ds, y_test_aligned = make_tf_dataset(
            X_test, y_test, SEQ_LEN, BATCH_SIZE, shuffle=False
        )
    else:
        test_ds = y_test_aligned = None

    # ── STEP 4: Build & Train ─────────────────────────────────────────────────
    log.info("[STEP 4] Building and training LSTM model ...")
    model   = build_model(SEQ_LEN, n_features)
    history = train_model(model, train_ds, val_ds)

    # ── STEP 5: Generate predictions ─────────────────────────────────────────
    log.info("[STEP 5] Generating predictions ...")
    prob_train = model.predict(train_ds, verbose=0).ravel()
    prob_val   = model.predict(val_ds,   verbose=0).ravel()
    prob_test  = model.predict(test_ds,  verbose=0).ravel() if test_ds else None

    # ── STEP 6: Evaluate all variants ────────────────────────────────────────
    log.info("[STEP 6] Evaluating all variants ...")
    for split_name, y_al, prob, df_split in [
        ("Train", y_train_aligned, prob_train, train_df),
        ("Val",   y_val_aligned,   prob_val,   val_df),
    ]:
        evaluate_standard(y_al, prob, split_name, df=df_split)
        evaluate_3state(y_al, prob, split_name, df=df_split)
        evaluate_variant_c(df_split, prob, split_name)

    if prob_test is not None:
        evaluate_standard(y_test_aligned, prob_test, "Test", df=test_df)
        evaluate_3state(y_test_aligned, prob_test, "Test", df=test_df)
        evaluate_variant_c(test_df, prob_test, "Test")

    # ── STEP 7: Diagnostic visualizations ────────────────────────────────────
    log.info("[STEP 7] Generating diagnostic visualizations ...")
    plot_training_history(history)

    val_metrics = evaluate_standard.__wrapped__ if hasattr(evaluate_standard, "__wrapped__") \
        else None
    val_auc = roc_auc_score(y_val_aligned, prob_val)
    plot_roc_curve(y_val_aligned, prob_val, val_auc, "Val")
    plot_confusion_matrix(y_val_aligned, prob_val, "Val")

    if prob_test is not None:
        test_auc = roc_auc_score(y_test_aligned, prob_test)
        plot_roc_curve(y_test_aligned, prob_test, test_auc, "Test")
        plot_confusion_matrix(y_test_aligned, prob_test, "Test")

    # ── STEP 8: Advanced quant finance plots ─────────────────────────────────
    log.info("[STEP 8] Generating advanced quantitative finance plots ...")

    for split_name, prob in [
        ("Train", prob_train),
        ("Val",   prob_val),
    ]:
        plot_prob_distribution(prob, split_name)

    plot_equity_curve_v2(prob_train, train_df, "Train")
    plot_equity_curve_v2(prob_val,   val_df,   "Val")

    if prob_test is not None:
        plot_prob_distribution(prob_test, "Test")
        plot_equity_curve_v2(prob_test, test_df, "Test")

    # Threshold sensitivity on Val only (no test snooping)
    plot_threshold_sensitivity(y_val_aligned, prob_val)

    # ── STEP 9: Permutation Feature Importance (Val only) ─────────────────────
    log.info("[STEP 9] Computing permutation feature importance (Val) ...")
    plot_feature_importance(
        model, X_val, y_val, y_val_aligned, prob_val, feature_cols, n_repeats=3,
    )

    log.info("02_lstm_model_v2.py  —  PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
