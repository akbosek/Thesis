#!/usr/bin/env python3
# =============================================================================
# 02_lstm_model.py
# =============================================================================
# PURPOSE:
#   Load the processed dataset, split it chronologically by EXACT DATE
#   boundaries, train a regularised two-layer LSTM classifier, evaluate
#   it with both a standard 0.5 threshold and a 3-state confidence filter,
#   and save publication-quality diagnostic and quantitative-finance plots.
#
# PIPELINE STEPS:
#   1. Load & Split   – Load CSV; filter into Train / Val / Test by date.
#   2. Scale          – StandardScaler fitted ONLY on Train data (no leakage).
#   3. Window         – tf.keras.utils.timeseries_dataset_from_array.
#   4. Build & Train  – 2-layer LSTM with L2 regularisation + Dropout.
#   5. Evaluate A     – Standard 0.5 threshold: Accuracy, Sensitivity,
#                       Specificity, ROC AUC, Gini on all three splits.
#   6. Evaluate B     – 3-state logic: Coverage + Conditional Win Rate.
#   7. Visualize      – Training history, ROC curve, confusion matrices.
#   8. Advanced Plots – Probability distribution KDE           [NEW]
#                       Simulated equity curves                 [NEW]
#                       Threshold sensitivity analysis          [NEW]
#
# OUTPUTS  (existing):
#   • training_history.png        – Loss & accuracy curves over epochs
#   • roc_curve_test.png          – ROC curve with AUC for the Test set
#   • confusion_matrix_val.png    – Seaborn CM heatmap for the Val set
#   • confusion_matrix_test.png   – Seaborn CM heatmap for the Test set
#
# OUTPUTS  (new — quantitative finance thesis visuals):
#   • prob_distribution.png       – KDE of LSTM output + threshold zone bands
#   • equity_curve_test.png       – Buy & Hold vs Variant A vs Variant B PnL
#   • threshold_sensitivity.png   – Win rate & coverage across threshold sweep
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

matplotlib.use("Agg")   # Non-interactive backend – safe for headless/server usage.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   # Clean date tick formatting for equity curve.
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2 as L2Reg
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
TRAIN_END:   str = "2022-12-31"   # ~5-year training window
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"   # 2-year validation window (hyperparameter tuning)
TEST_START:  str = "2025-01-01"
TEST_END:    str = "2025-11-30"   # ~6-month test window (final evaluation, touch once)

# ── Sequence (look-back window) length ───────────────────────────────────────
# The LSTM will see SEQ_LEN consecutive days of features before making a
# prediction. 20 days ≈ 4 trading weeks, capturing short-term momentum.
SEQ_LEN: int = 20

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int   = 64     # Mini-batch size; powers of 2 are cache-efficient.
EPOCHS: int       = 150    # Upper bound; EarlyStopping will halt training early.

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_UNITS_1: int   = 512    # Units in first LSTM layer (higher capacity).
LSTM_UNITS_2: int   = 256    # Units in second LSTM layer (compression).
DROPOUT_RATE: float = 0.25   # Fraction of activations zeroed each batch.
L2_FACTOR:    float = 1e-4   # Kernel regularisation weight; penalises large weights.

# ── 3-state confidence threshold logic ───────────────────────────────────────
# The model only signals a trade when it is sufficiently confident.
#   P > UPPER_THRESHOLD → predict UP   (long signal)
#   P < LOWER_THRESHOLD → predict DOWN (short/exit signal)
#   LOWER ≤ P ≤ UPPER   → HOLD         (model abstains; day not traded)
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
        logging.FileHandler("run_log.txt", mode="w", encoding="utf-8") 
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — DATA LOADING & DATE-BASED SPLITTING
# =============================================================================

def load_and_split(csv_path: str) -> tuple:
    """
    Load the processed CSV and partition it into three non-overlapping,
    strictly chronological subsets using exact calendar date boundaries.

    Design principle: Temporal holdout
    ────────────────────────────────────
    Unlike cross-validation on i.i.d. data, financial time series cannot be
    randomly shuffled across splits because future information would leak
    into training. We enforce a hard chronological wall:

        TRAIN [2018–2022] → VAL [2023–2024] → TEST [2025]
                          ▲                  ▲
                  Hyperparameter          Reported
                    selection             results

    The test set is touched EXACTLY ONCE — after all hyperparameter choices
    are finalised using the validation set.

    Parameters
    ----------
    csv_path : str  –  Path to 'processed_dataset.csv'.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrames
        Each DataFrame retains its DatetimeIndex and all feature/target columns.
        The raw DataFrames are returned here (BEFORE scaling) so that downstream
        functions (e.g. plot_equity_curve) can access the original un-scaled
        btc_log_return values for economic performance simulation.
    """
    log.info("Loading dataset from '%s' ...", csv_path)
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)   # Ensure chronological order after loading.

    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df.index.min().date(), df.index.max().date(),
    )

    # Boolean masks — both start and end dates are INCLUSIVE.
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    val_mask   = (df.index >= VAL_START)   & (df.index <= VAL_END)
    test_mask  = (df.index >= TEST_START)  & (df.index <= TEST_END)

    train_df = df[train_mask]
    val_df   = df[val_mask]
    test_df  = df[test_mask]

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        log.info(
            "  %-5s → %5d rows  [%s → %s]",
            name, len(split),
            split.index.min().date(), split.index.max().date(),
        )

    return train_df, val_df, test_df


# =============================================================================
# STEP 2 — FEATURE SCALING
# =============================================================================

def scale_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple:
    """
    Apply StandardScaler to features, fitting STRICTLY on training data.

    Why fit on training data only?  (Data-leakage prevention)
    ──────────────────────────────────────────────────────────
    StandardScaler computes the mean (μ) and std (σ) of each feature.
    If fitted on the entire dataset those statistics would embed statistical
    properties of future data — a subtle form of look-ahead bias.

    Correct protocol:
        scaler.fit(X_train)          ← uses only past observations
        scaler.transform(X_val)      ← applies μ/σ from training
        scaler.transform(X_test)     ← applies μ/σ from training

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrames with features + 'target'.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray (N_split, n_features)
    y_train, y_val, y_test : np.ndarray (N_split,)
    scaler                 : fitted StandardScaler instance
    feature_cols           : list[str] of feature column names
    """
    feature_cols = [c for c in train_df.columns if c != "target"]
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    X_test_raw  = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df["target"].values.astype(np.float32)
    y_val   = val_df["target"].values.astype(np.float32)
    y_test  = test_df["target"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform training set
    X_val   = scaler.transform(X_val_raw)          # only transform (no fit)
    X_test  = scaler.transform(X_test_raw)         # only transform (no fit)

    log.info(
        "Scaler fitted on training set. Feature means (rounded): %s",
        dict(zip(feature_cols, np.round(scaler.mean_, 4))),
    )
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
    Build a tf.data.Dataset of non-overlapping sliding windows using the
    official Keras utility. No manual Python for-loops are used.

    ── Window ↔ Target alignment (critical — no look-ahead bias) ─────────────

    Given N rows and look-back length L:

      Window i  covers  features[i : i+L]
                         └── last window day = row index i + L - 1

      y[t] = direction of btc_log_return on day t+1  (from data generator).

      Target for window i  =  y[i + L - 1]
        → uses all information up to row (i + L - 1) inclusive
        → predicts BTC movement on row (i + L)  ← strictly future

    Parameters
    ----------
    X         : np.ndarray (N, n_features) – scaled feature matrix.
    y         : np.ndarray (N,) – binary target labels (0 or 1).
    seq_len   : int  – number of look-back time-steps per window.
    batch_size: int  – windows per batch.
    shuffle   : bool – True only for the training dataset.

    Returns
    -------
    dataset   : tf.data.Dataset yielding (window_batch, label_batch).
    y_aligned : np.ndarray  –  y[seq_len-1:]  matching dataset label order.
    """
    N         = len(X)
    n_windows = N - seq_len + 1

    # Construct target array aligned with window start indices.
    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[seq_len - 1:]   # y[L-1], y[L], ..., y[N-1]
    targets_aligned[n_windows:] = 0.0               # padding tail — never read

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
    Construct and compile a two-layer stacked LSTM with L2 + Dropout
    regularisation for binary direction classification.

    Architecture overview
    ─────────────────────
    Input  →  LSTM(LSTM_UNITS_1, return_sequences=True)
           →  Dropout(DROPOUT_RATE)
           →  LSTM(LSTM_UNITS_2, return_sequences=False)
           →  Dropout(DROPOUT_RATE)
           →  Dense(1, sigmoid)   → P(BTC closes higher tomorrow) ∈ (0, 1)

    Anti-overfitting measures (stacked triple defence)
    ────────────────────────────────────────────────────
    1. L2 weight decay  – penalises large kernel weights → smoother boundaries.
    2. Dropout          – randomly zeros activations → ensemble-like behaviour.
    3. EarlyStopping    – halts before val_loss minimum is passed (see train_model).

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
            # Layer 1: Return full sequence so Layer 2 can learn higher-level patterns.
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                kernel_regularizer=L2Reg(L2_FACTOR),
                input_shape=(seq_len, n_features),
                name="lstm_layer_1",
            ),
            Dropout(DROPOUT_RATE, name="dropout_1"),

            # Layer 2: Compress the temporal sequence into a single fixed-size vector.
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                kernel_regularizer=L2Reg(L2_FACTOR),
                name="lstm_layer_2",
            ),
            Dropout(DROPOUT_RATE, name="dropout_2"),

            # Output: sigmoid maps z → (0, 1), interpreted as P(up tomorrow).
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
    Train the LSTM with two stacked early-stopping callbacks.

    EarlyStopping  (patience = 20 epochs)
        Monitors val_loss. Halts if it does not improve for 20 consecutive
        epochs and restores weights from the best epoch.

    ReduceLROnPlateau  (patience = 8 epochs, factor = 0.5)
        Halves the learning rate after 8 stagnant epochs, enabling finer
        convergence before EarlyStopping fires.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
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

    Metrics computed
    ────────────────
    Accuracy    = (TP + TN) / total
    Sensitivity = TP / (TP + FN)   (True Positive Rate / Recall)
    Specificity = TN / (TN + FP)   (True Negative Rate)
    ROC AUC     = threshold-independent ranking quality (0.5 → random)
    Gini        = 2 × AUC − 1      (normalised to [-1, 1])
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
        otherwise            →  HOLD (no trade taken)

    Metrics (computed on TRADED days only)
    ───────────────────────────────────────
    Coverage          = n_traded / n_total
    Conditional Win Rate = Accuracy(traded days)
        When the model commits to a trade, how often is it correct?
        A high CWR with reasonable coverage is the ideal outcome.
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
# STEP 7 — EXISTING DIAGNOSTIC VISUALIZATIONS
# =============================================================================

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Save a 2-panel figure: binary cross-entropy loss (left) and classification
    accuracy (right) for both Train and Validation splits across all epochs.

    Overfitting diagnosis guide
    ────────────────────────────
    Healthy  : Train and Val curves converge and remain close.
    Overfit  : Val loss diverges upward while train loss keeps falling.
    Underfit : Both curves plateau at a high loss / low accuracy value.
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"LSTM Training History — BTC Daily Direction Classifier\n"
        f"Architecture: {LSTM_UNITS_1}→{LSTM_UNITS_2} units | "
        f"L2={L2_FACTOR} | Dropout={DROPOUT_RATE} | SeqLen={SEQ_LEN}",
        fontsize=11,
    )

    # Panel 1 — Loss
    axes[0].plot(epochs_range, history.history["loss"],
                 lw=2, color="steelblue", label="Train Loss")
    axes[0].plot(epochs_range, history.history["val_loss"],
                 lw=2, color="tomato", ls="--", label="Validation Loss")
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2 — Accuracy
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
) -> None:
    """
    Save a ROC curve with AUC and Gini coefficient annotation for the Test set.

    Interpretation
    ───────────────
    Diagonal (AUC=0.50) → random guessing.
    Top-left corner     → perfect classifier.
    The shaded area between the curve and the diagonal = Gini / 2.
    AUC is threshold-independent: it measures probability ranking quality
    regardless of where the 0.5 decision boundary is placed.
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
    ax.set_title("ROC Curve — Test Set", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig("roc_curve_test.png", dpi=150)
    plt.close(fig)
    log.info("ROC curve saved → roc_curve_test.png")


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    split_name: str,
) -> None:
    """
    Save a seaborn confusion matrix heatmap with raw counts and
    row-normalised percentages in each cell.

    Row-normalisation converts counts to per-class recall rates:
        Row 0 (True DOWN) → Specificity
        Row 1 (True UP)   → Sensitivity
    This is more informative than raw counts when classes are imbalanced.
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
# STEP 8 — NEW: ADVANCED QUANTITATIVE FINANCE VISUALIZATIONS
# =============================================================================
#
# These three new functions are the primary visual contribution added for the
# thesis defense. Each is independently interpretable by a committee with
# quantitative finance (rather than purely machine-learning) expertise.
#
#   plot_prob_distribution()     – Raw LSTM output density + 3-zone overlay.
#   plot_equity_curve()          – Cumulative PnL: B&H vs Variant A vs Variant B.
#   plot_threshold_sensitivity() – Win rate & coverage across a threshold sweep.
#
# =============================================================================


def plot_prob_distribution(y_prob_test: np.ndarray) -> None:
    """
    Plot the probability density of LSTM outputs on the Test set, overlaid
    with threshold bands that define the 3-state trading logic.

    Thesis rationale
    ────────────────
    A well-calibrated binary classifier produces a bimodal distribution, with
    probability mass concentrated near 0 (high short conviction) and near 1
    (high long conviction). Predictions concentrated near 0.50 carry maximum
    uncertainty — the model is effectively "flipping a coin" on those days.

    The shaded grey 'Noise Zone' between LOWER_THRESHOLD and UPPER_THRESHOLD
    directly and visually justifies Variant B's abstention logic, answering
    the natural committee question: "Why not trade every day?"
    The annotated percentages showing how many test days fall in each zone
    directly support the Coverage metric reported in the Variant B table.

    Plot elements
    ─────────────
    • Grey shaded band              – Noise Zone (LOWER to UPPER threshold).
    • Blue histogram (stat=density) – empirical distribution of P(up) values.
    • Overlaid KDE curve            – smooth density estimate (via seaborn).
    • Dashed vertical lines         – LOWER (red), 0.50 (black), UPPER (green).
    • Annotated zone boxes          – zone name + % of test predictions.

    Parameters
    ----------
    y_prob_test : np.ndarray
        Raw LSTM sigmoid outputs for the Test set,
        shape = (N_test_windows,), dtype = float32, values ∈ (0, 1).

    Saves
    ─────
    'prob_distribution.png' at 150 DPI.
    """
    n_total = len(y_prob_test)

    # ── Compute per-zone counts for the annotation boxes ──────────────────────
    n_short = int((y_prob_test <  LOWER_THRESHOLD).sum())
    n_noise = int(
        ((y_prob_test >= LOWER_THRESHOLD) & (y_prob_test <= UPPER_THRESHOLD)).sum()
    )
    n_long  = int((y_prob_test >  UPPER_THRESHOLD).sum())

    pct_short = 100.0 * n_short / n_total
    pct_noise = 100.0 * n_noise / n_total
    pct_long  = 100.0 * n_long  / n_total

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    # Draw the Noise Zone FIRST so it sits behind all other plot elements.
    ax.axvspan(
        LOWER_THRESHOLD, UPPER_THRESHOLD,
        color="lightgrey", alpha=0.70, zorder=0,
        label=(f"Noise Zone  [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]  "
               f"→  {pct_noise:.1f}% of test predictions"),
    )

    # Histogram with KDE overlay.
    # stat='density' normalises histogram area to 1.0, making the y-axis
    # directly comparable to the KDE curve (both are probability densities).
    sns.histplot(
        y_prob_test,
        bins=40,
        kde=True,
        stat="density",
        color="steelblue",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label="LSTM Output Distribution  (histogram + KDE)",
        ax=ax,
        zorder=1,
    )

    # Vertical threshold markers — drawn on top of the histogram (zorder=2).
    ax.axvline(
        LOWER_THRESHOLD, color="crimson", lw=2.0, ls="--", zorder=2,
        label=f"Lower Threshold = {LOWER_THRESHOLD:.2f}  (Short signal boundary)",
    )
    ax.axvline(
        0.50, color="black", lw=1.8, ls=":", zorder=2,
        label="Decision Boundary = 0.50  (Variant A cut-off)",
    )
    ax.axvline(
        UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
        label=f"Upper Threshold = {UPPER_THRESHOLD:.2f}  (Long signal boundary)",
    )

    # ── Zone annotation boxes ─────────────────────────────────────────────────
    # Query the finalised y-range to place the text boxes at 82% of peak height.
    ymax  = ax.get_ylim()[1]
    box_y = ymax * 0.75

    zone_specs = [
        # (x_centre,                          label_text,                          color)
        (LOWER_THRESHOLD / 2,
         f"SHORT zone\n{pct_short:.1f}%  ({n_short} days)",
         "crimson"),
        ((LOWER_THRESHOLD + UPPER_THRESHOLD) / 2,
         f"HOLD zone\n{pct_noise:.1f}%  ({n_noise} days)",
         "dimgrey"),
        (UPPER_THRESHOLD + (1.0 - UPPER_THRESHOLD) / 2,
         f"LONG zone\n{pct_long:.1f}%  ({n_long} days)",
         "seagreen"),
    ]
    for x_pos, text, color in zone_specs:
        ax.text(
            x_pos, box_y, text,
            ha="center", va="top", fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=color, alpha=0.80, lw=1.2),
        )

    # ── Labels, title, legend ─────────────────────────────────────────────────
    ax.set_xlabel("Predicted Probability  P(BTC closes higher tomorrow)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Probability Distribution of LSTM Output — Test Set\n"
        f"Grey band = Noise Zone [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]  "
        "where Variant B abstains from trading",
        fontsize=13, pad=14,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.30)

    plt.tight_layout()
    fig.savefig("prob_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Probability distribution saved → prob_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────


def plot_equity_curve(
    prob_test: np.ndarray,
    test_df:   pd.DataFrame,
) -> None:
    """
    Simulate and plot three cumulative return series over the Test set period.

    This is the central quantitative result for the thesis defense. It
    converts abstract metrics (accuracy, AUC) into tangible economic
    information: did the model generate alpha over the passive benchmark?

    ── Strategy definitions ──────────────────────────────────────────────────

    Buy & Hold (benchmark)
        Always long BTC.  Return[t] = btc_log_return[t].
        Every active strategy must beat this to justify its complexity.

    Variant A — Always in Market
        Signals are generated every day; no abstention.
        Signal = +1 (LONG)  if  P ≥ 0.50
        Signal = −1 (SHORT) if  P <  0.50
        Return[t] = signal[t] × btc_log_return[t]
        A short position earns the NEGATIVE of the BTC return.

    Variant B — 3-State Confidence Filter
        Signal =  0 (FLAT)  if  LOWER ≤ P ≤ UPPER  → capital sits in cash
        Signal = +1 (LONG)  if  P > UPPER
        Signal = −1 (SHORT) if  P < LOWER
        Return[t] = signal[t] × btc_log_return[t]   (0 on flat days)
        Key thesis question: does filtering low-conviction predictions
        improve the cumulative return profile vs. Buy & Hold?

    ── Prediction-to-return alignment ───────────────────────────────────────
        prob_test[j] is generated from a window that ENDS at test_df row
        (j + SEQ_LEN − 1). It predicts the sign of btc_log_return at row
        (j + SEQ_LEN). The return actually earned is therefore:

            earned_return[j] = btc_log_return_test[j + SEQ_LEN]

        The last prediction (j = N − SEQ_LEN) would require test_df row N,
        which is out of bounds. We trim it safely — a negligible 1-sample
        correction that has no practical effect on the curve shape.

    ── Cumulative return conversion ──────────────────────────────────────────
        cum[t]   = Σ_{s=0}^{t}  strategy_return[s]   (sum of log-returns)
        pct[t]   = ( exp(cum[t]) − 1 ) × 100           (total % gain | loss)
        Adding 0 % at the origin is implied by the empty cumulative sum.

    Parameters
    ----------
    prob_test : np.ndarray
        LSTM output probabilities for the Test set,
        shape = (N_test − SEQ_LEN + 1,).
    test_df   : pd.DataFrame
        Raw (un-scaled) test split with DatetimeIndex and 'btc_log_return'.
        Passed as a raw DataFrame (not X_test) so that the original log-return
        values and the DatetimeIndex are both directly accessible.

    Saves
    ─────
    'equity_curve_test.png' at 150 DPI.
    """
    # ── Alignment: trim both arrays to in-bounds trade days ───────────────────
    n_test      = len(test_df)
    n_tradeable = n_test - SEQ_LEN   # max valid trade days (last pred dropped)

    if n_tradeable <= 0:
        log.warning(
            "Test set (%d rows) ≤ SEQ_LEN (%d). Skipping equity_curve_test.png.",
            n_test, SEQ_LEN,
        )
        return

    # Actual BTC log-returns realised on each trade day (row SEQ_LEN → N-1).
    btc_returns  = test_df["btc_log_return"].values[SEQ_LEN : SEQ_LEN + n_tradeable]

    # Drop the last prediction whose corresponding return is out of test_df bounds.
    prob_aligned = prob_test[:n_tradeable]

    # Date axis: the calendar dates on which returns are realised (trade day).
    dates = test_df.index[SEQ_LEN : SEQ_LEN + n_tradeable]

    # ── Strategy signals ──────────────────────────────────────────────────────
    # Variant A: binary direction, always deployed.
    signal_a = np.where(prob_aligned >= 0.5, 1.0, -1.0)

    # Variant B: ternary — long / short / flat (cash).
    signal_b = np.where(
        prob_aligned > UPPER_THRESHOLD,  1.0,
        np.where(prob_aligned < LOWER_THRESHOLD, -1.0, 0.0),
    )

    # ── Cumulative log-returns → total percentage return ──────────────────────
    # cumsum of log-returns = log(V_t / V_0) where V is portfolio value.
    # (exp(·) − 1) × 100 converts to the familiar % gain / loss scale.
    pct_bh = (np.exp(np.cumsum(btc_returns))            - 1.0) * 100.0
    pct_a  = (np.exp(np.cumsum(signal_a * btc_returns)) - 1.0) * 100.0
    pct_b  = (np.exp(np.cumsum(signal_b * btc_returns)) - 1.0) * 100.0

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    # Embed final cumulative return directly in the legend label.
    # This is cleaner than on-chart annotations and avoids right-boundary clipping.
    ax.plot(
        dates, pct_bh, lw=2.2, color="dimgrey", ls="-", zorder=2,
        label=f"Buy & Hold  (BTC Benchmark)                  →  {pct_bh[-1]:+.1f}%",
    )
    ax.plot(
        dates, pct_a, lw=2.0, color="steelblue", ls="-", zorder=3,
        label=f"Variant A  (Always in Market, thr=0.50)  →  {pct_a[-1]:+.1f}%",
    )
    ax.plot(
        dates, pct_b, lw=2.5, color="tomato", ls="-", zorder=4,
        label=(f"Variant B  (3-State Filter ±{UPPER_THRESHOLD:.2f})             "
               f"→  {pct_b[-1]:+.1f}%"),
    )

    # Subtle shading under the B&H curve as a visual reference area.
    ax.fill_between(dates, 0, pct_bh, alpha=0.06, color="dimgrey", zorder=1)

    # Zero baseline — visual break-even reference.
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.35)

    # ── Date axis formatting ──────────────────────────────────────────────────
    # Monthly ticks with abbreviated month + year format (e.g. "Jan '25").
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)

    # ── Labels, title, legend ─────────────────────────────────────────────────
    ax.set_xlabel("Date  (day the return is realised)", fontsize=12)
    ax.set_ylabel("Cumulative Return  (%)", fontsize=12)
    ax.set_title(
        "Simulated Equity Curves — Test Set\n"
        f"Buy & Hold  ·  Variant A (Always Invested, thr=0.50)  ·  "
        f"Variant B (3-State Confidence Filter, ±{UPPER_THRESHOLD:.2f})\n"
        f"Period: {str(dates[0].date())}  →  {str(dates[-1].date())}",
        fontsize=12, pad=14,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.30)

    plt.tight_layout()
    fig.savefig("equity_curve_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Equity curve saved → equity_curve_test.png")


# ─────────────────────────────────────────────────────────────────────────────


def plot_threshold_sensitivity(
    y_true_val: np.ndarray,
    prob_val:   np.ndarray,
) -> None:
    """
    Sweep symmetric confidence thresholds from 0.50 to 0.70 and visualise how
    Conditional Win Rate and Coverage change on the VALIDATION set.

    This chart is the quantitative justification for the selected threshold.
    It answers the committee question: "Why UPPER = 0.55 specifically, and
    not 0.60 or 0.65?" by making the trade-off curve explicit.

    ── Symmetric threshold convention ───────────────────────────────────────
        For each tested upper value u, the lower is defined as l = 1 − u,
        preserving symmetry around the 0.50 decision boundary:
            u = 0.50 → l = 0.50  →  all days traded (100% coverage)
            u = 0.55 → l = 0.45  →  moderate filter  (our selected point)
            u = 0.70 → l = 0.30  →  strict filter (very low coverage)

    ── Metrics ───────────────────────────────────────────────────────────────
        Coverage (%)           = (n_traded / n_total) × 100
            Fraction of val days where the model is willing to trade.

        Conditional Win Rate (%) = Accuracy(traded days) × 100
            Of the days the model commits to, what % were correct?
            Ideally this RISES as the threshold widens — indicating the model
            genuinely produces higher-quality signals when it is more selective.
            If it stays flat, the model's probability ranking is poorly calibrated.

    ── Why Validation — NOT Test set? ────────────────────────────────────────
        Using the Test set here would constitute implicit threshold snooping:
        the UPPER/LOWER values would be implicitly tuned on the same data they
        are reported on, inflating the Variant B metrics dishonestly.
        The Validation set is the theoretically correct holdout for all
        hyperparameter and threshold selection decisions.

    Parameters
    ----------
    y_true_val : np.ndarray  –  True labels for the Validation set.
    prob_val   : np.ndarray  –  LSTM probabilities for the Validation set.

    Saves
    ─────
    'threshold_sensitivity.png' at 150 DPI.
    """
    # ── Parameter sweep configuration ─────────────────────────────────────────
    # Step 0.02 produces 11 test points: [0.50, 0.52, 0.54, ..., 0.70]
    upper_thresholds = np.arange(0.50, 0.71, 0.02)

    win_rates: list = []    # Conditional Win Rate at each threshold (%)
    coverages: list = []    # Coverage at each threshold (%)

    for upper in upper_thresholds:
        lower      = 1.0 - upper            # symmetric lower threshold
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
            # No days qualify at this threshold; win rate is mathematically undefined.
            win_rate = float("nan")

        win_rates.append(win_rate)
        coverages.append(coverage)

        log.info(
            "  u=%.2f / l=%.2f  →  Coverage=%5.1f%%  WinRate=%5.1f%%  "
            "(%d / %d traded days)",
            upper, lower, coverage, win_rate, n_traded, n_total,
        )

    # ── Build dual-axis figure ────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()   # Secondary Y-axis; shares the same X-axis.

    # ── Primary axis (left, blue): Conditional Win Rate ───────────────────────
    line_wr, = ax1.plot(
        upper_thresholds, win_rates,
        color="steelblue", lw=2.5, ls="-", marker="o", ms=7, zorder=3,
        label="Conditional Win Rate  (left axis, %)",
    )

    # ── Secondary axis (right, red): Coverage ────────────────────────────────
    line_cov, = ax2.plot(
        upper_thresholds, coverages,
        color="tomato", lw=2.5, ls="-", marker="s", ms=7, zorder=3,
        label="Coverage  (right axis, % days traded)",
    )

    # ── Reference lines ───────────────────────────────────────────────────────
    # Green dashed vertical line marking the operationally selected threshold.
    ax1.axvline(
        UPPER_THRESHOLD, color="seagreen", lw=2.0, ls="--", zorder=2,
        label=f"Selected threshold  =  {UPPER_THRESHOLD:.2f}",
    )
    # Horizontal dotted line at 50% — the random-guess win rate baseline.
    ax1.axhline(
        50.0, color="black", lw=0.9, ls=":", alpha=0.50, zorder=1,
        label="50% Win Rate  (random baseline)",
    )

    # ── Axis labels and tick styling ──────────────────────────────────────────
    ax1.set_xlabel(
        "Upper Threshold  u   (Lower  l = 1 − u,  symmetric around 0.50)",
        fontsize=11,
    )
    ax1.set_ylabel("Conditional Win Rate  (%)", fontsize=11, color="steelblue")
    ax2.set_ylabel(
        "Coverage  (% of Validation Days Traded)", fontsize=11, color="tomato"
    )

    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")

    # Display actual threshold values on X-axis (e.g. 0.50, 0.52, ..., 0.70).
    ax1.set_xticks(upper_thresholds)
    ax1.set_xticklabels([f"{x:.2f}" for x in upper_thresholds], fontsize=9)

    ax1.set_title(
        "Threshold Sensitivity Analysis — Validation Set\n"
        "Blue  (left)  : Conditional Win Rate — accuracy on high-conviction days\n"
        "Red   (right) : Coverage — fraction of days the model chooses to trade\n"
        "Green dashed  : operationally selected threshold",
        fontsize=11, pad=14,
    )

    # ── Unified legend (merge handles from both axes) ─────────────────────────
    # ax1.get_legend_handles_labels() returns: [line_wr, axvline, axhline]
    # We append line_cov from ax2 manually.
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
# MAIN — Orchestrate the full model pipeline  (updated: 7 → 8 steps)
# =============================================================================

def main() -> None:
    """
    Run the complete LSTM training, evaluation, and visualisation pipeline.

    Pipeline extended from 7 to 8 steps:
        Steps 1–7 are identical to the original script.
        Step 8 adds the three advanced quantitative-finance plots.
    """
    log.info("=" * 65)
    log.info("  02_lstm_model.py  —  PIPELINE START")
    log.info("  Train:  %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val:    %s → %s", VAL_START,   VAL_END)
    log.info("  Test:   %s → %s", TEST_START,  TEST_END)
    log.info("  SEQ_LEN=%d  BATCH=%d  EPOCHS=%d", SEQ_LEN, BATCH_SIZE, EPOCHS)
    log.info("  LSTM units: %d → %d  |  Dropout: %.2f  |  L2: %g",
             LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE, L2_FACTOR)
    log.info("=" * 65)

    # ── STEP 1: Load & Split ──────────────────────────────────────────────────
    log.info("[STEP 1/8] Loading and splitting dataset ...")
    train_df, val_df, test_df = load_and_split(INPUT_CSV)

    # ── STEP 2: Scale ─────────────────────────────────────────────────────────
    log.info("[STEP 2/8] Scaling features (fit on Train only) ...")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, feature_cols,
    ) = scale_features(train_df, val_df, test_df)

    n_features = X_train.shape[1]

    # ── STEP 3: Build tf.data.Datasets ───────────────────────────────────────
    log.info("[STEP 3/8] Building windowed tf.data.Datasets (SEQ_LEN=%d) ...", SEQ_LEN)
    train_ds_fit, _ = make_tf_dataset(X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=True)
    train_ds_eval, y_train_aligned = make_tf_dataset(X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=False)
    val_ds,  y_val_aligned  = make_tf_dataset(X_val,  y_val,  SEQ_LEN, BATCH_SIZE)
    test_ds, y_test_aligned = make_tf_dataset(X_test, y_test, SEQ_LEN, BATCH_SIZE)

    # ── STEP 4: Build & Train Model ───────────────────────────────────────────
    log.info("[STEP 4/8] Building model architecture ...")
    model = build_model(SEQ_LEN, n_features)

    log.info("[STEP 4/8] Training LSTM (EarlyStopping active) ...")
    history = train_model(model, train_ds_fit, val_ds)

    # ── STEP 5: Generate Predictions ─────────────────────────────────────────
    log.info("[STEP 5/8] Generating predictions on all splits ...")
    prob_train = model.predict(train_ds_eval, verbose=0).flatten()
    prob_val   = model.predict(val_ds,        verbose=0).flatten()
    prob_test  = model.predict(test_ds,       verbose=0).flatten()

    # ── STEP 6A: Variant A — Standard Evaluation ──────────────────────────────
    log.info("[STEP 6/8] Running Variant A (standard 0.5 threshold) ...")
    all_metrics_a = {}
    for split_name, y_true, y_prob in [
        ("Train", y_train_aligned, prob_train),
        ("Val",   y_val_aligned,   prob_val),
        ("Test",  y_test_aligned,  prob_test),
    ]:
        all_metrics_a[split_name] = evaluate_standard(y_true, y_prob, split_name)

    # ── STEP 6B: Variant B — 3-State Evaluation ───────────────────────────────
    log.info("[STEP 7/8] Running Variant B (3-state confidence filter) ...")
    all_metrics_b = {}
    for split_name, y_true, y_prob in [
        ("Train", y_train_aligned, prob_train),
        ("Val",   y_val_aligned,   prob_val),
        ("Test",  y_test_aligned,  prob_test),
    ]:
        all_metrics_b[split_name] = evaluate_3state(y_true, y_prob, split_name)

    # ── STEP 8: Visualizations (Including New Thesis Plots) ───────────────────
    log.info("[STEP 8/8] Generating and saving diagnostic plots ...")

    # Original ML plots
    plot_training_history(history)
    test_auc = all_metrics_a["Test"]["auc"]
    plot_roc_curve(y_test_aligned, prob_test, test_auc)
    plot_confusion_matrix(y_val_aligned,  prob_val,  "Val")
    plot_confusion_matrix(y_test_aligned, prob_test, "Test")

    # New Quantitative Finance plots
    plot_prob_distribution(prob_test)
    plot_equity_curve(prob_test, test_df)
    plot_threshold_sensitivity(y_val_aligned, prob_val)

    log.info("02_lstm_model.py  —  PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
