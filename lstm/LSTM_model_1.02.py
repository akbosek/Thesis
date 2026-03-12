#!/usr/bin/env python3
# =============================================================================
# LSTM_model_1.02.py
# =============================================================================
# PURPOSE:
#   Load the processed dataset, split it chronologically by EXACT DATE
#   boundaries, train a regularised two-layer LSTM classifier, evaluate
#   it with both a standard 0.5 threshold and a 3-state confidence filter,
#   and save publication-quality diagnostic and quantitative-finance plots.
#
# CHANGES FROM v1.01  (tagged [CHG n] inline):
#   [CHG 1] RECURRENT DROPOUT — Added RECURRENT_DROPOUT_RATE hyperparameter.
#           Applied recurrent_dropout= to both LSTM layers.
#           NOTE: cuDNN kernel is disabled when recurrent_dropout > 0.
#           Training is ~3-5x slower per epoch; offset by shorter runs.
#   [CHG 2] TEST SET TOGGLE — EVALUATE_ON_TEST = False completely skips
#           test-set loading, scaling, windowing, prediction, evaluation,
#           and plotting. Safe to leave False during all tuning phases.
#   [CHG 3] AGGRESSIVE EARLY STOPPING — EarlyStopping patience 20 → 7.
#           Halts training the moment val_loss diverges persistently.
#   [CHG 4] SPLIT-AWARE PLOTS — plot_prob_distribution() and
#           plot_equity_curve() now accept a split_name parameter and
#           generate per-split files (e.g., equity_curve_val.png).
#           Called for Train + Val always; Test only if [CHG 2] toggle = True.
#   [CHG 5] FEATURE IMPORTANCE — New plot_feature_importance() function.
#           Uses Permutation Feature Importance on the Validation set:
#           shuffles each feature independently, measures the AUC drop,
#           and saves a ranked bar chart to feature_importance_val.png.
#
# PIPELINE STEPS:
#   1. Load & Split  – Load CSV; filter into Train / Val / [Test] by date.
#   2. Scale         – StandardScaler fitted ONLY on Train (no leakage).
#   3. Window        – tf.keras.utils.timeseries_dataset_from_array.
#   4. Build & Train – 2-layer LSTM: L2 + Dropout + Recurrent Dropout.  [CHG 1]
#   5. Evaluate A    – Standard 0.5 threshold: Acc, Sens, Spec, AUC, Gini.
#   6. Evaluate B    – 3-state logic: Coverage + Conditional Win Rate.
#   7. Visualize     – Training history, ROC curve, confusion matrices.
#   8. Advanced Plots:
#        • Probability distribution KDE  (Train, Val, [Test])           [CHG 4]
#        • Simulated equity curves       (Train, Val, [Test])           [CHG 4]
#        • Threshold sensitivity analysis (Val only — no snooping)
#        • Permutation Feature Importance (Val only)                    [CHG 5]
#
# OUTPUTS (existing):
#   • training_history.png
#   • roc_curve_[test|val].png
#   • confusion_matrix_[val|test].png
#
# OUTPUTS (quantitative finance thesis visuals):
#   • prob_distribution_train.png / prob_distribution_val.png / [_test]
#   • equity_curve_train.png      / equity_curve_val.png      / [_test]
#   • threshold_sensitivity.png
#   • feature_importance_val.png                                        [CHG 5]
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

matplotlib.use("Agg")   # Non-interactive backend — safe for headless/server use.
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
# Edit ONLY this block to change the model configuration.
# =============================================================================

# ── Date-based chronological splits ──────────────────────────────────────────
# Using exact date boundaries (instead of percentage splits) is MANDATORY for
# time-series data. Percentage splits risk contaminating future information
# into training data whenever rows are not perfectly ordered, and they make
# the split boundaries model-version-dependent and hard to reproduce.
TRAIN_START: str = "2016-01-01"
TRAIN_END:   str = "2022-12-31"
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"
TEST_START:  str = "2025-01-01"
TEST_END:    str = "2025-11-30"

# ── [CHG 2] TEST SET TOGGLE ───────────────────────────────────────────────────
# Set to False during ALL tuning / experimentation phases to keep the test
# set locked away. Flip to True ONLY for the final, single, reported run.
#
#   False → test set is never loaded, scaled, windowed, predicted, or plotted.
#   True  → full pipeline including test evaluation and test plots.
EVALUATE_ON_TEST: bool = True

# ── Sequence (look-back window) length ───────────────────────────────────────
# 20 days ≈ 4 trading weeks; captures short-to-medium momentum.
SEQ_LEN: int = 20

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 64    # Powers of 2 are cache-efficient.
EPOCHS:     int = 150   # Hard ceiling; EarlyStopping fires well before this.

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_UNITS_1:  int   = 512    # First LSTM layer (capacity).
LSTM_UNITS_2:  int   = 256    # Second LSTM layer (compression).
DROPOUT_RATE:  float = 0.25   # Inter-layer dropout fraction.
L2_FACTOR:     float = 1e-4   # Kernel weight decay.

# ── [CHG 1] RECURRENT DROPOUT ─────────────────────────────────────────────────
# Drops connections in the recurrent kernel (h_{t-1} gates) — directly
# regularises the temporal memory mechanism where financial overfitting lives.
# IMPORTANT: any value > 0 disables the cuDNN LSTM kernel, making each epoch
# ~3-5× slower. With patience=7 the total run time remains acceptable.
# Recommended range for financial time series: 0.15–0.25.
RECURRENT_DROPOUT_RATE: float = 0.20

# ── 3-state confidence threshold logic ───────────────────────────────────────
UPPER_THRESHOLD: float = 0.55
LOWER_THRESHOLD: float = 0.45

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV: str = "processed_dataset.csv"

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

    Design principle: Temporal holdout
    ────────────────────────────────────
    Financial time series cannot be randomly shuffled across splits.
    We enforce a hard chronological wall:

        TRAIN [2016–2022] → VAL [2024] → TEST [2025]
                          ▲              ▲
                  Hyperparameter      Reported
                    selection         results (locked behind EVALUATE_ON_TEST)

    [CHG 2] If EVALUATE_ON_TEST=False the test slice is never extracted.
            test_df is returned as None; all downstream functions handle None.

    Returns
    -------
    (train_df, val_df, test_df)
        test_df is None when EVALUATE_ON_TEST=False.
        Raw DataFrames (pre-scaling) are returned so that plot_equity_curve
        can access the original btc_log_return values and DatetimeIndex.
    """
    log.info("Loading dataset from '%s' ...", csv_path)
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

    # [CHG 2] Only extract test data when the toggle is on.
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
    Apply StandardScaler to features, fitting STRICTLY on training data.

    Why fit on training data only?  (Data-leakage prevention)
    ──────────────────────────────────────────────────────────
    StandardScaler computes μ and σ for each feature. Fitting on the full
    dataset would embed future volatility regime statistics into the
    training transform — a subtle but real form of look-ahead bias.

        scaler.fit(X_train)     ← only past observations used
        scaler.transform(X_val) ← applies training μ/σ (no refit)
        scaler.transform(X_test)← applies training μ/σ (no refit)

    [CHG 2] test_df=None (when EVALUATE_ON_TEST=False) is handled safely:
            X_test and y_test are returned as None without crashing.

    Returns
    -------
    X_train, X_val, X_test   : np.ndarray or None
    y_train, y_val, y_test   : np.ndarray or None
    scaler                   : fitted StandardScaler
    feature_cols             : list[str]
    """
    feature_cols = [c for c in train_df.columns if c != "target"]
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

    # [CHG 2] Skip test transform entirely if test is locked.
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
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    batch_size: int,
    shuffle: bool = False,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows using the official Keras
    utility. No manual Python for-loops are used.

    ── Window ↔ Target alignment (critical — no look-ahead bias) ─────────────

    Given N rows and look-back length L:

      Window i  covers  features[i : i+L]
                         └── last day of window = row index i + L - 1

      y[t] encodes direction of btc_log_return on day t+1 (from data_generator).

      Target for window i  =  y[i + L - 1]
        → uses all information through day (i + L - 1)
        → predicts BTC movement on day (i + L)   ← strictly future

    The target array is pre-shifted so the API's native pairing is correct.
    The padded tail (targets_aligned[n_windows:]) is never consumed.

    Returns
    -------
    dataset   : tf.data.Dataset
    y_aligned : np.ndarray — y[seq_len-1:], mirrors dataset label order.
                             Use for sklearn metric functions outside TF graph.
    """
    N         = len(X)
    n_windows = N - seq_len + 1

    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[seq_len - 1:]
    targets_aligned[n_windows:] = 0.0   # padding tail — never read by the API

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
    Construct and compile a two-layer stacked LSTM with four stacked
    anti-overfitting defences.

    Architecture
    ────────────
    Input  →  LSTM(LSTM_UNITS_1, return_sequences=True)
           →  Dropout(DROPOUT_RATE)
           →  LSTM(LSTM_UNITS_2, return_sequences=False)
           →  Dropout(DROPOUT_RATE)
           →  Dense(1, sigmoid)   →  P(BTC closes higher tomorrow) ∈ (0, 1)

    Anti-overfitting defences (four layers)
    ────────────────────────────────────────
    1. L2 weight decay        – penalises large kernel weights.
    2. Inter-layer Dropout    – zeros output activations between LSTM layers;
                                regularises feed-forward connections.
    3. Recurrent Dropout      – zeros connections in the recurrent kernel     [CHG 1]
                                (h_{t-1} gates); directly regularises the
                                temporal memory — the primary site of sequence
                                overfitting in financial LSTMs.
                                ⚠ Disables cuDNN: ~3-5× slower per epoch.
    4. EarlyStopping          – halts before val_loss minimum is passed.

    Parameters
    ----------
    seq_len    : int – time-steps per input sequence (SEQ_LEN).
    n_features : int – feature dimension per time-step.

    Returns
    -------
    model : compiled tf.keras.Model
    """
    model = Sequential(
        [
            # ── Layer 1: Return full sequence for Layer 2 ─────────────────────
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                kernel_regularizer=L2Reg(L2_FACTOR),
                # [CHG 1] Regularise the recurrent (h_{t-1}) weight matrix.
                # Equivalent to applying dropout to the hidden state at each
                # time step inside the LSTM cell — not just at layer output.
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                input_shape=(seq_len, n_features),
                name="lstm_layer_1",
            ),
            Dropout(DROPOUT_RATE, name="dropout_1"),

            # ── Layer 2: Compress sequence → single fixed-size vector ──────────
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                kernel_regularizer=L2Reg(L2_FACTOR),
                # [CHG 1] Same recurrent regularisation on the second layer.
                recurrent_dropout=RECURRENT_DROPOUT_RATE,
                name="lstm_layer_2",
            ),
            Dropout(DROPOUT_RATE, name="dropout_2"),

            # ── Output: P(up) ∈ (0, 1) via sigmoid ───────────────────────────
            Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="btc_direction_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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
    Train the LSTM with two stacked callbacks.

    EarlyStopping  (patience = 7)                                       [CHG 3]
        Monitors val_loss. Halts training if it does not strictly improve
        for 7 consecutive epochs and restores weights from the best epoch.
        Reduced from 20 → 7 to terminate immediately when the train/val
        loss curves diverge — the primary symptom of the observed overfitting.
        Rationale: with severe overfitting (train=0.2, val=1.6) we need the
        brakes applied quickly; patience=20 allowed the model to overfit for
        ~13 extra epochs before the kill signal.

    ReduceLROnPlateau  (patience = 5, factor = 0.5)
        Halves the learning rate after 5 stagnant epochs. Fires BEFORE
        EarlyStopping, giving the optimizer one chance to escape a plateau
        with a finer step size before training is terminated.
        Patience aligned to roughly 70% of EarlyStopping patience.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,             # [CHG 3] was 20; now aggressively 7
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,             # fires before EarlyStopping triggers
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
# STEP 6A — VARIANT A: STANDARD 0.5 THRESHOLD EVALUATION
# =============================================================================

def evaluate_standard(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
) -> dict:
    """
    Evaluate model performance using a hard 0.5 classification threshold.

    Metrics
    ───────
    Accuracy    = (TP + TN) / total
    Sensitivity = TP / (TP + FN)   (True Positive Rate / Recall)
    Specificity = TN / (TN + FP)   (True Negative Rate)
    ROC AUC     = threshold-independent ranking quality  (0.5 → random)
    Gini        = 2 × AUC − 1      (normalised to [−1, 1])
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
    print(f"{'═' * 58}\n")

    return dict(accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, auc=auc, gini=gini)


# =============================================================================
# STEP 6B — VARIANT B: 3-STATE CONFIDENCE-FILTER EVALUATION
# =============================================================================

def evaluate_3state(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
) -> dict:
    """
    Evaluate model performance using a 3-state confidence filter.

    Logic
    ─────
        P > UPPER_THRESHOLD  →  BUY  (long BTC)
        P < LOWER_THRESHOLD  →  SELL (short BTC / exit)
        otherwise            →  HOLD (no trade — model abstains)

    Metrics (computed on TRADED days only)
    ───────────────────────────────────────
    Coverage          = n_traded / n_total
    Conditional Win Rate = Accuracy(traded days)
    """
    trade_mask = (y_prob > UPPER_THRESHOLD) | (y_prob < LOWER_THRESHOLD)
    n_traded   = int(trade_mask.sum())
    n_total    = len(y_prob)
    coverage   = n_traded / n_total if n_total > 0 else 0.0

    if n_traded == 0:
        log.warning(
            "[%s] Zero days exceed threshold bands (UPPER=%.2f, LOWER=%.2f). "
            "Consider widening the gap.",
            split_name, UPPER_THRESHOLD, LOWER_THRESHOLD,
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
    print(f"{'═' * 58}\n")

    return dict(coverage=coverage, conditional_win_rate=cond_win_rate,
                n_traded=n_traded)


# =============================================================================
# STEP 7 — DIAGNOSTIC VISUALIZATIONS
# =============================================================================

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Save a 2-panel figure: BCE loss and accuracy for Train + Val splits.

    Overfitting diagnosis guide
    ────────────────────────────
    Healthy  : Train and Val curves converge and remain close.
    Overfit  : Val loss rises while train loss keeps falling.
    Underfit : Both plateau at a high loss value.
    EarlyStop: Curves end before EPOCHS ceiling — look for when they stop.
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"LSTM Training History — BTC Daily Direction Classifier\n"
        f"Architecture: {LSTM_UNITS_1}→{LSTM_UNITS_2} units | "
        f"L2={L2_FACTOR} | Dropout={DROPOUT_RATE} | "
        f"RecDrop={RECURRENT_DROPOUT_RATE} | SeqLen={SEQ_LEN}",  # [CHG 1]
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
    fig.savefig("training_history.png", dpi=150)
    plt.close(fig)
    log.info("Training history saved → training_history.png")


def plot_roc_curve(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    auc_score: float,
    split_name: str = "Test",
) -> None:
    """
    Save a ROC curve with AUC and Gini annotation.

    Interpretation
    ───────────────
    Diagonal (AUC=0.50) → random. Top-left → perfect.
    AUC is threshold-independent: measures probability ranking quality.
    ROC is intentionally reported on the validation set during tuning, and
    on the test set only when EVALUATE_ON_TEST=True.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    gini = 2.0 * auc_score - 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="steelblue",
            label=f"LSTM  (AUC = {auc_score:.4f}  |  Gini = {gini:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2,
            label="Random Classifier  (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
    ax.set_title(f"ROC Curve — {split_name} Set", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("ROC curve saved → %s", fname)


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
) -> None:
    """
    Save a seaborn confusion matrix heatmap with raw counts and
    row-normalised percentages.

    Row-normalisation converts counts to per-class recall rates:
        Row 0 (True DOWN) → Specificity
        Row 1 (True UP)   → Sensitivity
    """
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
        f"Confusion Matrix — {split_name} Set\n"
        "(Threshold = 0.50 | Colour = Row-Normalised Rate)",
        fontsize=12,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fname = f"confusion_matrix_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", fname)


# =============================================================================
# STEP 8 — ADVANCED QUANTITATIVE FINANCE VISUALIZATIONS
# =============================================================================


def plot_prob_distribution(y_prob: np.ndarray, split_name: str) -> None:
    """
    [CHG 4] Plot the probability density of LSTM outputs for a given split,
    overlaid with threshold bands that define the 3-state trading logic.

    Thesis rationale
    ────────────────
    A well-calibrated classifier should produce bimodal outputs — probability
    mass near 0 (high short conviction) and near 1 (high long conviction).
    Predictions near 0.50 have maximum uncertainty — the grey 'Noise Zone'
    directly and visually justifies Variant B's abstention logic.

    The annotated zone counts directly support the Coverage metric reported
    in the Variant B table: "X% of {split_name} days fell in the noise zone."

    [CHG 4] Previously hardcoded to the test set. Now accepts split_name and
    saves to prob_distribution_{split_name.lower()}.png.

    Parameters
    ----------
    y_prob      : np.ndarray — LSTM sigmoid outputs, shape (N_windows,).
    split_name  : str        — 'Train', 'Val', or 'Test'; used in title + filename.
    """
    n_total = len(y_prob)

    n_short = int((y_prob <  LOWER_THRESHOLD).sum())
    n_noise = int(((y_prob >= LOWER_THRESHOLD) & (y_prob <= UPPER_THRESHOLD)).sum())
    n_long  = int((y_prob >  UPPER_THRESHOLD).sum())

    pct_short = 100.0 * n_short / n_total
    pct_noise = 100.0 * n_noise / n_total
    pct_long  = 100.0 * n_long  / n_total

    fig, ax = plt.subplots(figsize=(11, 6))

    # Noise Zone drawn first so it sits behind all other plot elements.
    ax.axvspan(
        LOWER_THRESHOLD, UPPER_THRESHOLD,
        color="lightgrey", alpha=0.70, zorder=0,
        label=(f"Noise Zone  [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]  "
               f"→  {pct_noise:.1f}% of {split_name} predictions"),
    )

    # Histogram with KDE overlay.
    sns.histplot(
        y_prob,
        bins=40,
        kde=True,
        stat="density",
        color="steelblue",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label=f"LSTM Output Distribution — {split_name} Set",
        ax=ax,
        zorder=1,
    )

    # Vertical threshold markers.
    ax.axvline(LOWER_THRESHOLD, color="crimson",  lw=2.0, ls="--", zorder=2,
               label=f"Lower Threshold = {LOWER_THRESHOLD:.2f}  (Short boundary)")
    ax.axvline(0.50,            color="black",    lw=1.8, ls=":", zorder=2,
               label="Decision Boundary = 0.50  (Variant A cut-off)")
    ax.axvline(UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
               label=f"Upper Threshold = {UPPER_THRESHOLD:.2f}  (Long boundary)")

    # Zone annotation boxes — placed at 75% of the y-axis peak.
    ymax  = ax.get_ylim()[1]
    box_y = ymax * 0.75

    zone_specs = [
        (LOWER_THRESHOLD / 2,
         f"SHORT zone\n{pct_short:.1f}%  ({n_short} days)", "crimson"),
        ((LOWER_THRESHOLD + UPPER_THRESHOLD) / 2,
         f"HOLD zone\n{pct_noise:.1f}%  ({n_noise} days)",  "dimgrey"),
        (UPPER_THRESHOLD + (1.0 - UPPER_THRESHOLD) / 2,
         f"LONG zone\n{pct_long:.1f}%  ({n_long} days)",    "seagreen"),
    ]
    for x_pos, text, color in zone_specs:
        ax.text(
            x_pos, box_y, text,
            ha="center", va="top", fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, alpha=0.80, lw=1.2),
        )

    ax.set_xlabel("Predicted Probability  P(BTC closes higher tomorrow)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Probability Distribution of LSTM Output — {split_name} Set\n"
        f"Grey band = Noise Zone [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]  "
        "where Variant B abstains from trading",
        fontsize=13, pad=14,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.30)

    plt.tight_layout()
    fname = f"prob_distribution_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Probability distribution saved → %s", fname)


def plot_equity_curve(
    prob:       np.ndarray,
    df:         pd.DataFrame,
    split_name: str,
) -> None:
    """
    [CHG 4] Simulate and plot three cumulative return series for a given split.

    This is the central quantitative result for the thesis defense. It converts
    abstract metrics (AUC, accuracy) into tangible economic information: did
    the model generate alpha over the passive benchmark?

    ── Strategy definitions ──────────────────────────────────────────────────

    Buy & Hold (benchmark)
        Return[t] = btc_log_return[t].  Always long. Complexity-free baseline.

    Variant A — Always in Market
        Signal = +1 (LONG)  if  P ≥ 0.50
        Signal = −1 (SHORT) if  P <  0.50
        Return[t] = signal[t] × btc_log_return[t]

    Variant B — 3-State Confidence Filter
        Signal =  0 (FLAT)  if  LOWER ≤ P ≤ UPPER  → capital sits in cash
        Signal = +1 (LONG)  if  P > UPPER
        Signal = −1 (SHORT) if  P < LOWER
        Return[t] = signal[t] × btc_log_return[t]   (0 on flat days)

    ── Prediction-to-return alignment ───────────────────────────────────────
        prob[j] is generated from a window ending at df row (j + SEQ_LEN − 1).
        It predicts the sign of btc_log_return at row (j + SEQ_LEN).
        earned_return[j] = btc_log_return[j + SEQ_LEN]
        The last prediction is trimmed — its corresponding return is out of
        df bounds.

    ── Thesis note on the Train equity curve ────────────────────────────────
        Severe overfitting should be visually obvious here: the model's signals
        will nearly perfectly track training prices, producing an unrealistically
        smooth equity curve. Contrast with the Val/Test curves to demonstrate
        the generalisation gap to the committee.

    [CHG 4] Previously hardcoded to the test set. Now accepts split_name and
    saves to equity_curve_{split_name.lower()}.png.

    Parameters
    ----------
    prob       : np.ndarray — LSTM predictions (N_windows,).
    df         : pd.DataFrame — raw (un-scaled) split DataFrame with
                 DatetimeIndex and 'btc_log_return' column.
    split_name : str — 'Train', 'Val', or 'Test'.
    """
    n_rows      = len(df)
    n_tradeable = n_rows - SEQ_LEN   # number of valid trade days

    if n_tradeable <= 0:
        log.warning(
            "[%s] Rows (%d) ≤ SEQ_LEN (%d). Skipping equity curve.",
            split_name, n_rows, SEQ_LEN,
        )
        return

    # Actual BTC log-returns on each trade day (row SEQ_LEN → N-1).
    btc_returns  = df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]
    # Drop the last prediction whose corresponding return is out of df bounds.
    prob_aligned = prob[:n_tradeable]
    # Calendar dates on which returns are realised.
    dates = df.index[SEQ_LEN : SEQ_LEN + n_tradeable]

    # ── Strategy signals ──────────────────────────────────────────────────────
    signal_a = np.where(prob_aligned >= 0.5, 1.0, -1.0)
    signal_b = np.where(
        prob_aligned > UPPER_THRESHOLD,   1.0,
        np.where(prob_aligned < LOWER_THRESHOLD, -1.0, 0.0),
    )

    # ── Cumulative log-returns → total % gain/loss ────────────────────────────
    pct_bh = (np.exp(np.cumsum(btc_returns))            - 1.0) * 100.0
    pct_a  = (np.exp(np.cumsum(signal_a * btc_returns)) - 1.0) * 100.0
    pct_b  = (np.exp(np.cumsum(signal_b * btc_returns)) - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, pct_bh, lw=2.2, color="dimgrey",   ls="-", zorder=2,
            label=f"Buy & Hold  (Benchmark)                 →  {pct_bh[-1]:+.1f}%")
    ax.plot(dates, pct_a,  lw=2.0, color="steelblue", ls="-", zorder=3,
            label=f"Variant A  (Always Invested, thr=0.50) →  {pct_a[-1]:+.1f}%")
    ax.plot(dates, pct_b,  lw=2.5, color="tomato",    ls="-", zorder=4,
            label=(f"Variant B  (3-State ±{UPPER_THRESHOLD:.2f})               "
                   f"→  {pct_b[-1]:+.1f}%"))

    ax.fill_between(dates, 0, pct_bh, alpha=0.06, color="dimgrey", zorder=1)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.35)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)

    ax.set_xlabel("Date  (day the return is realised)", fontsize=12)
    ax.set_ylabel("Cumulative Return  (%)", fontsize=12)
    ax.set_title(
        f"Simulated Equity Curves — {split_name} Set\n"
        f"Buy & Hold  ·  Variant A (Always Invested)  ·  "
        f"Variant B (3-State Confidence Filter ±{UPPER_THRESHOLD:.2f})\n"
        f"Period: {str(dates[0].date())}  →  {str(dates[-1].date())}",
        fontsize=12, pad=14,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.30)

    plt.tight_layout()
    fname = f"equity_curve_{split_name.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Equity curve saved → %s", fname)


def plot_threshold_sensitivity(
    y_true_val: np.ndarray,
    prob_val:   np.ndarray,
) -> None:
    """
    Sweep symmetric confidence thresholds from 0.50 to 0.70 and visualise
    the Conditional Win Rate and Coverage trade-off on the VALIDATION set.

    This chart is the quantitative justification for the chosen threshold.
    It answers: "Why UPPER = 0.55 specifically?"

    ── Why Validation only — never Test ─────────────────────────────────────
    Using the Test set here would constitute implicit threshold snooping:
    UPPER/LOWER would be implicitly optimised on the same data they are
    reported on, inflating Variant B metrics dishonestly.

    ── Symmetric threshold convention ───────────────────────────────────────
        For each tested upper value u:  lower l = 1 − u
            u = 0.50 → 100% coverage (all days traded)
            u = 0.55 → moderate filter (our selected point)
            u = 0.70 → strict filter (very low coverage)
    """
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
            "  u=%.2f / l=%.2f  →  Coverage=%5.1f%%  WinRate=%5.1f%%  "
            "(%d / %d traded)",
            upper, lower, coverage, win_rate, n_traded, n_total,
        )

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    line_wr, = ax1.plot(
        upper_thresholds, win_rates,
        color="steelblue", lw=2.5, ls="-", marker="o", ms=7, zorder=3,
        label="Conditional Win Rate  (left axis, %)",
    )
    line_cov, = ax2.plot(
        upper_thresholds, coverages,
        color="tomato", lw=2.5, ls="-", marker="s", ms=7, zorder=3,
        label="Coverage  (right axis, % days traded)",
    )

    ax1.axvline(UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
                label=f"Selected threshold  =  {UPPER_THRESHOLD:.2f}")
    ax1.axhline(50.0, color="black", lw=0.9, ls=":", alpha=0.50, zorder=1,
                label="50% Win Rate  (random baseline)")

    ax1.set_xlabel(
        "Upper Threshold  u   (Lower  l = 1 − u,  symmetric around 0.50)",
        fontsize=11,
    )
    ax1.set_ylabel("Conditional Win Rate  (%)", fontsize=11, color="steelblue")
    ax2.set_ylabel("Coverage  (% of Validation Days Traded)", fontsize=11,
                   color="tomato")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_xticks(upper_thresholds)
    ax1.set_xticklabels([f"{x:.2f}" for x in upper_thresholds], fontsize=9)

    ax1.set_title(
        "Threshold Sensitivity Analysis — Validation Set\n"
        "Blue (left): Conditional Win Rate  |  Red (right): Coverage\n"
        "Green dashed: operationally selected threshold",
        fontsize=11, pad=14,
    )

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(
        ax1_handles + [line_cov],
        ax1_labels  + [line_cov.get_label()],
        fontsize=9, loc="best", framealpha=0.92,
    )
    ax1.grid(axis="both", alpha=0.28)

    plt.tight_layout()
    fig.savefig("threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Threshold sensitivity saved → threshold_sensitivity.png")


# =============================================================================
# [CHG 5] STEP 9 — PERMUTATION FEATURE IMPORTANCE
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
    [CHG 5] Compute and visualise Permutation Feature Importance on the
    Validation set, saving a ranked bar chart to feature_importance_val.png.

    ── Method: Permutation Feature Importance ────────────────────────────────
    For each feature column i:
        1. Randomly permute the values of column i across all rows of X_val.
           This destroys the signal in feature i while preserving the
           marginal distribution of all other features — no data is fabricated.
        2. Re-create the windowed tf.data.Dataset from the permuted X_val.
        3. Generate predictions with the trained model.
        4. Compute ROC AUC on the permuted predictions vs. y_val_aligned.
        5. Importance = baseline_AUC − permuted_AUC.
           A large positive drop means the model relied heavily on feature i.
           A value near 0 means the model barely uses that feature.

    Repeat n_repeats times (different random permutations) and average the
    results to reduce Monte Carlo variance. A single permutation can produce
    a lucky (high or low) arrangement; averaging over 3–5 repeats is standard.

    ── Why ROC AUC and not accuracy? ────────────────────────────────────────
    AUC is more informative than accuracy for importance estimation:
        • It is threshold-independent — we are measuring signal in the raw
          probability output, not after a hard 0.5 cut.
        • It is less sensitive to class imbalance.
        • Small permutation effects are detectable in AUC but may be invisible
          in accuracy when the model is near a classification boundary.

    ── Why Validation set — not Train? ──────────────────────────────────────
    Importance on the training set reflects what the model memorized, not
    what actually generalises. Validation set importance tells us which
    features the model relies on in production-like conditions — the
    economically meaningful question.

    ── Computational note ───────────────────────────────────────────────────
    Each of the n_features × n_repeats iterations runs a full forward pass
    through the validation dataset. With recurrent_dropout > 0 (cuDNN
    disabled), each pass is slower than in v1.01. For n_features=6 and
    n_repeats=3 this is 18 passes — acceptable for a thesis workflow.

    Parameters
    ----------
    model             : trained tf.keras.Model.
    X_val             : np.ndarray (N_val, n_features) — scaled val features.
    y_val             : np.ndarray (N_val,) — unaligned val targets.
                        Needed to reconstruct the windowed dataset.
    y_val_aligned     : np.ndarray (N_val - SEQ_LEN + 1,) — aligned targets.
                        Used as ground truth for AUC computation.
    prob_val_baseline : np.ndarray — LSTM predictions on the unperturbed
                        validation set. Pre-computed in main() and passed
                        here to avoid a redundant forward pass.
    feature_cols      : list[str] — feature column names in column order.
    n_repeats         : int — number of random permutations per feature.

    Saves
    ─────
    'feature_importance_val.png' at 150 DPI.
    """
    rng = np.random.default_rng(seed=SEED)

    # ── Baseline AUC (unperturbed validation set) ─────────────────────────────
    baseline_auc = roc_auc_score(y_val_aligned, prob_val_baseline)
    log.info("Feature importance — baseline Val AUC: %.4f", baseline_auc)

    importances_mean = np.zeros(len(feature_cols), dtype=np.float64)
    importances_std  = np.zeros(len(feature_cols), dtype=np.float64)

    for feat_idx, feat_name in enumerate(feature_cols):
        auc_drops = []

        for repeat in range(n_repeats):
            # Copy X_val so we never mutate the original.
            X_permuted = X_val.copy()

            # Permute only column feat_idx — destroy its predictive signal.
            perm_indices = rng.permutation(len(X_permuted))
            X_permuted[:, feat_idx] = X_permuted[perm_indices, feat_idx]

            # Reconstruct the windowed tf.data.Dataset from permuted features.
            # We must re-use make_tf_dataset because the sliding-window logic
            # is inseparable from the LSTM's input format.
            ds_permuted, _ = make_tf_dataset(
                X_permuted, y_val, SEQ_LEN, BATCH_SIZE, shuffle=False
            )

            # Forward pass on permuted data — verbose=0 suppresses per-batch output.
            prob_permuted = model.predict(ds_permuted, verbose=0).flatten()

            permuted_auc = roc_auc_score(y_val_aligned, prob_permuted)
            auc_drop     = baseline_auc - permuted_auc
            auc_drops.append(auc_drop)

            log.info(
                "  [%s] repeat %d/%d — permuted AUC=%.4f  drop=%.4f",
                feat_name, repeat + 1, n_repeats, permuted_auc, auc_drop,
            )

        importances_mean[feat_idx] = np.mean(auc_drops)
        importances_std[feat_idx]  = np.std(auc_drops)

    # ── Sort features by mean importance (descending) ─────────────────────────
    sort_order = np.argsort(importances_mean)[::-1]
    sorted_names = [feature_cols[i] for i in sort_order]
    sorted_means = importances_mean[sort_order]
    sorted_stds  = importances_std[sort_order]

    # Map internal column names to readable thesis-friendly labels.
    label_map = {
        "btc_log_return":  "BTC Return",
        "spy_log_return":  "SPY (S&P 500)",
        "gold_log_return": "Gold",
        "nvda_log_return": "NVDA",
        "dxy_log_return":  "DXY (USD Index)",
        "is_weekend":      "Is Weekend",
    }
    display_names = [label_map.get(n, n) for n in sorted_names]

    # ── Build bar chart ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colour bars: green for positive importance (model relied on the feature),
    # red for negative (permuting the feature accidentally helped — indicates
    # the model may be using this feature in a noisy or counter-productive way).
    colors = ["seagreen" if v >= 0 else "tomato" for v in sorted_means]

    bars = ax.barh(
        y=display_names,
        width=sorted_means,
        xerr=sorted_stds,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=5,
        error_kw={"ecolor": "dimgrey", "elinewidth": 1.5},
        zorder=2,
    )

    # Annotate each bar with its mean importance value.
    for bar, mean_val in zip(bars, sorted_means):
        x_offset = 0.001 if mean_val >= 0 else -0.001
        ha = "left" if mean_val >= 0 else "right"
        ax.text(
            mean_val + x_offset, bar.get_y() + bar.get_height() / 2,
            f"{mean_val:+.4f}",
            va="center", ha=ha, fontsize=9, color="black",
        )

    # Zero baseline — features to the right of this line increase AUC when
    # kept intact; to the left means permuting them improved AUC (suspicious).
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6, zorder=1)

    ax.set_xlabel(
        "Mean AUC Drop when Feature is Permuted\n"
        "(Baseline Val AUC − Permuted Val AUC;  larger = more important)",
        fontsize=11,
    )
    ax.set_title(
        f"Permutation Feature Importance — Validation Set\n"
        f"Baseline Val AUC = {baseline_auc:.4f}  |  "
        f"{n_repeats} repeats per feature  |  Error bars = ±1 std",
        fontsize=12, pad=14,
    )
    ax.invert_yaxis()   # Highest importance at the top.
    ax.grid(axis="x", alpha=0.30, zorder=0)

    plt.tight_layout()
    fig.savefig("feature_importance_val.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance saved → feature_importance_val.png")


# =============================================================================
# MAIN — Orchestrate the full model pipeline
# =============================================================================

def main() -> None:
    """
    Run the complete LSTM training, evaluation, and visualisation pipeline.

    Gate structure for EVALUATE_ON_TEST=False  [CHG 2]
    ─────────────────────────────────────────────────
    All test-set operations are wrapped in `if EVALUATE_ON_TEST:` blocks.
    When False, the script produces identical outputs for Train and Val,
    then terminates without ever reading the test slice from the CSV.
    """
    log.info("=" * 65)
    log.info("  LSTM_model_1.02.py  —  PIPELINE START")
    log.info("  Train:  %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val:    %s → %s", VAL_START,   VAL_END)
    if EVALUATE_ON_TEST:
        log.info("  Test:   %s → %s  [UNLOCKED]", TEST_START, TEST_END)
    else:
        log.info("  Test:   LOCKED  (EVALUATE_ON_TEST=False)")
    log.info(
        "  SEQ_LEN=%d  BATCH=%d  EPOCHS=%d",
        SEQ_LEN, BATCH_SIZE, EPOCHS,
    )
    log.info(
        "  LSTM units: %d → %d  |  Dropout: %.2f  |  "
        "RecDrop: %.2f  |  L2: %g",             # [CHG 1]
        LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE,
        RECURRENT_DROPOUT_RATE, L2_FACTOR,
    )
    log.info("=" * 65)

    # ── STEP 1: Load & Split ──────────────────────────────────────────────────
    log.info("[STEP 1] Loading and splitting dataset ...")
    train_df, val_df, test_df = load_and_split(INPUT_CSV)   # [CHG 2]

    # ── STEP 2: Scale ─────────────────────────────────────────────────────────
    log.info("[STEP 2] Scaling features (fit on Train only) ...")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, feature_cols,
    ) = scale_features(train_df, val_df, test_df)   # [CHG 2] test_df may be None

    n_features = X_train.shape[1]

    # ── STEP 3: Build tf.data.Datasets ───────────────────────────────────────
    log.info("[STEP 3] Building windowed tf.data.Datasets (SEQ_LEN=%d) ...", SEQ_LEN)

    # Shuffled training dataset — used during model.fit().
    train_ds_fit,  _               = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=True)

    # Un-shuffled training dataset — used only for metric computation.
    # Temporal order must be preserved so predictions align with y_aligned.
    train_ds_eval, y_train_aligned = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=False)

    val_ds,        y_val_aligned   = make_tf_dataset(
        X_val, y_val, SEQ_LEN, BATCH_SIZE)

    # [CHG 2] Only build test dataset when the toggle is on.
    if EVALUATE_ON_TEST:
        test_ds, y_test_aligned = make_tf_dataset(
            X_test, y_test, SEQ_LEN, BATCH_SIZE)
    else:
        test_ds = y_test_aligned = None

    log.info(
        "Windowed set sizes → Train: %d | Val: %d | Test: %s windows",
        len(y_train_aligned), len(y_val_aligned),
        str(len(y_test_aligned)) if y_test_aligned is not None else "LOCKED",
    )

    # ── STEP 4: Build & Train Model ───────────────────────────────────────────
    log.info("[STEP 4] Building model architecture ...")
    model = build_model(SEQ_LEN, n_features)   # [CHG 1] recurrent_dropout added

    log.info("[STEP 4] Training LSTM (EarlyStopping patience=7) ...")  # [CHG 3]
    history = train_model(model, train_ds_fit, val_ds)

    # ── STEP 5: Generate Predictions ─────────────────────────────────────────
    log.info("[STEP 5] Generating predictions on Train and Val ...")
    prob_train = model.predict(train_ds_eval, verbose=0).flatten()
    prob_val   = model.predict(val_ds,        verbose=0).flatten()

    if EVALUATE_ON_TEST:   # [CHG 2]
        log.info("[STEP 5] Generating predictions on Test ...")
        prob_test = model.predict(test_ds, verbose=0).flatten()
    else:
        prob_test = None

    # ── STEP 6A: Variant A — Standard Evaluation ──────────────────────────────
    log.info("[STEP 6A] Variant A (standard 0.5 threshold) ...")
    all_metrics_a = {}

    eval_splits = [("Train", y_train_aligned, prob_train),
                   ("Val",   y_val_aligned,   prob_val)]
    if EVALUATE_ON_TEST:   # [CHG 2]
        eval_splits.append(("Test", y_test_aligned, prob_test))

    for split_name, y_true, y_prob in eval_splits:
        all_metrics_a[split_name] = evaluate_standard(y_true, y_prob, split_name)

    # ── STEP 6B: Variant B — 3-State Evaluation ───────────────────────────────
    log.info("[STEP 6B] Variant B (3-state confidence filter) ...")
    all_metrics_b = {}
    for split_name, y_true, y_prob in eval_splits:
        all_metrics_b[split_name] = evaluate_3state(y_true, y_prob, split_name)

    # ── STEP 7: Diagnostic Visualizations ─────────────────────────────────────
    log.info("[STEP 7] Generating diagnostic plots ...")
    plot_training_history(history)

    # ROC curve: use Val during tuning; switch to Test only on final run.
    if EVALUATE_ON_TEST:   # [CHG 2]
        plot_roc_curve(y_test_aligned, prob_test,
                       all_metrics_a["Test"]["auc"], "Test")
    else:
        plot_roc_curve(y_val_aligned, prob_val,
                       all_metrics_a["Val"]["auc"], "Val")

    # Confusion matrices for all active splits.
    plot_confusion_matrix(y_train_aligned, prob_train, "Train")
    plot_confusion_matrix(y_val_aligned,   prob_val,   "Val")
    if EVALUATE_ON_TEST:   # [CHG 2]
        plot_confusion_matrix(y_test_aligned, prob_test, "Test")

    # ── STEP 8: Advanced Quantitative Finance Plots ───────────────────────────
    log.info("[STEP 8] Generating advanced finance plots ...")

    # [CHG 4] Probability distribution for Train + Val (and Test if unlocked).
    plot_prob_distribution(prob_train, "Train")
    plot_prob_distribution(prob_val,   "Val")
    if EVALUATE_ON_TEST:   # [CHG 2]
        plot_prob_distribution(prob_test, "Test")

    # [CHG 4] Equity curves for Train + Val (and Test if unlocked).
    # train_df and val_df are raw DataFrames retained for btc_log_return access.
    plot_equity_curve(prob_train, train_df, "Train")
    plot_equity_curve(prob_val,   val_df,   "Val")
    if EVALUATE_ON_TEST:   # [CHG 2]
        plot_equity_curve(prob_test, test_df, "Test")

    # Threshold sensitivity — always on Validation (never Test, avoids snooping).
    plot_threshold_sensitivity(y_val_aligned, prob_val)

    # ── STEP 9: Permutation Feature Importance ────────────────────────────────
    log.info(
        "[STEP 9] Computing permutation feature importance "
        "(Val set, %d repeats) ...", 3,          # [CHG 5]
    )
    plot_feature_importance(
        model             = model,
        X_val             = X_val,
        y_val             = y_val,           # unaligned — for dataset reconstruction
        y_val_aligned     = y_val_aligned,   # aligned  — for AUC computation
        prob_val_baseline = prob_val,        # avoids a redundant forward pass
        feature_cols      = feature_cols,
        n_repeats         = 3,
    )

    # ── Final summary printout ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 65)
    print(f"  {'Split':<8}  {'Accuracy':>9}  {'AUC':>7}  {'Gini':>7}  "
          f"{'Sens.':>7}  {'Spec.':>7}")
    print("  " + "─" * 55)
    for split_name in [k for k in ("Train", "Val", "Test") if k in all_metrics_a]:
        m = all_metrics_a[split_name]
        print(
            f"  {split_name:<8}  {m['accuracy']:>9.4f}  {m['auc']:>7.4f}  "
            f"{m['gini']:>7.4f}  {m['sensitivity']:>7.4f}  "
            f"{m['specificity']:>7.4f}"
        )
    print("  " + "─" * 55)
    print(f"\n  3-State (Variant B) — Val Set:")
    mb = all_metrics_b["Val"]
    print(f"    Coverage        : {mb['coverage']:.4f} "
          f"({mb['coverage']*100:.1f}% of val days traded)")
    print(f"    Cond. Win Rate  : {mb['conditional_win_rate']:.4f}")
    if EVALUATE_ON_TEST and "Test" in all_metrics_b:
        mb_t = all_metrics_b["Test"]
        print(f"\n  3-State (Variant B) — Test Set:")
        print(f"    Coverage        : {mb_t['coverage']:.4f} "
              f"({mb_t['coverage']*100:.1f}% of test days traded)")
        print(f"    Cond. Win Rate  : {mb_t['conditional_win_rate']:.4f}")
    print("═" * 65 + "\n")

    log.info("LSTM_model_1.02.py  —  PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
