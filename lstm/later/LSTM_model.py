#!/usr/bin/env python3
# =============================================================================
# 02_lstm_model.py
# =============================================================================
# PURPOSE:
#   Load the processed dataset, split it chronologically by EXACT DATE
#   boundaries, train a regularised two-layer LSTM classifier, evaluate
#   it with both a standard 0.5 threshold and a 3-state confidence filter,
#   and save publication-quality diagnostic plots.
#
# PIPELINE STEPS:
#   1. Load & Split  – Load CSV; filter into Train / Val / Test by date.
#   2. Scale         – StandardScaler fitted ONLY on Train data (no leakage).
#   3. Window        – tf.keras.utils.timeseries_dataset_from_array (no loops).
#   4. Build & Train – 2-layer LSTM with L2 regularisation + Dropout.
#   5. Evaluate A    – Standard 0.5 threshold: Accuracy, Sensitivity,
#                      Specificity, ROC AUC, Gini on all three splits.
#   6. Evaluate B    – 3-state logic: Coverage + Conditional Win Rate.
#   7. Visualize     – Training history, ROC curve, confusion matrices.
#
# OUTPUTS:
#   • training_history.png       – Loss & accuracy curves over epochs
#   • roc_curve_test.png         – ROC curve with AUC for the Test set
#   • confusion_matrix_val.png   – Seaborn CM heatmap for the Val set
#   • confusion_matrix_test.png  – Seaborn CM heatmap for the Test set
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
TRAIN_START: str = "2018-01-01"
TRAIN_END:   str = "2022-12-31"   # ~8-year training window
VAL_START:   str = "2023-01-01"
VAL_END:     str = "2024-12-31"   # 1-year validation window (hyperparameter tuning)
TEST_START:  str = "2025-01-01"
TEST_END:    str = "2025-06-30"   # ~1-year test window (final evaluation, touch once)

# ── Sequence (look-back window) length ───────────────────────────────────────
# The LSTM will see SEQ_LEN consecutive days of features before making a
# prediction. 15 days ≈ 3 trading weeks, capturing short-term momentum.
SEQ_LEN: int = 20

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int   = 64     # Mini-batch size; powers of 2 are cache-efficient.
EPOCHS: int       = 150    # Upper bound; EarlyStopping will halt training early.

# ── LSTM architecture ─────────────────────────────────────────────────────────
LSTM_UNITS_1: int  = 256    # Units in the first LSTM layer (higher capacity).
LSTM_UNITS_2: int  = 128    # Units in the second LSTM layer (compression).
DROPOUT_RATE: float = 0.25 # Randomly zero 40% of activations → reduces co-adaptation.
L2_FACTOR: float   = 1e-4  # Kernel regularisation weight; penalises large weights.

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
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — DATA LOADING & DATE-BASED SPLITTING
# =============================================================================

def load_and_split(
    csv_path: str,
) -> tuple:
    """
    Load the processed CSV and partition it into three non-overlapping,
    strictly chronological subsets using exact calendar date boundaries.

    Design principle: Temporal holdout
    -----------------------------------
    Unlike cross-validation on i.i.d. data, financial time series cannot be
    randomly shuffled across splits because future information would leak
    into training. We enforce a hard chronological wall:

        TRAIN [2015-2022]  →  VAL [2023]  →  TEST [2024-present]
                           ▲                ▲
                     Hyperparameter      Reported
                       selection         results

    The test set is touched EXACTLY ONCE — after all hyperparameter choices
    are finalised using the validation set.

    Parameters
    ----------
    csv_path : str  –  Path to 'processed_dataset.csv'.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrames
        Each DataFrame retains its DatetimeIndex and all feature/target columns.
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
    ----------------------------------------------------------
    StandardScaler computes the mean (μ) and std (σ) of each feature.
    If we fitted on the entire dataset (or on Val/Test), those statistics
    would embed statistical properties of future data into the training
    transform — a subtle form of look-ahead bias. The model would benefit
    from knowing future volatility regimes, inflating evaluation metrics.

    Correct protocol:
        scaler.fit(X_train)          ← uses only past observations
        scaler.transform(X_val)      ← applies μ/σ from training
        scaler.transform(X_test)     ← applies μ/σ from training

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrames with features + 'target'.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray of shape (N_split, n_features)
    y_train, y_val, y_test : np.ndarray of shape (N_split,)
    scaler                 : fitted StandardScaler instance
    feature_names          : list[str] of feature column names
    """
    feature_cols = [c for c in train_df.columns if c != "target"]
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # Separate features from target
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    X_test_raw  = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df["target"].values.astype(np.float32)
    y_val   = val_df["target"].values.astype(np.float32)
    y_test  = test_df["target"].values.astype(np.float32)

    # Fit on training, transform all three splits.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    log.info(
        "Scaler fitted on training set. "
        "Feature means (rounded): %s",
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
    official Keras utility.  No manual Python for-loops are used.

    ── Window ↔ Target alignment (critical for no look-ahead bias) ──────────

    Given an array of N rows and a look-back length L:

      Window  i  covers  features[i  : i+L]   (L consecutive day-rows)
                         └── last day of window = day index  i + L - 1

      y[t] encodes the direction of BTC's return on day t+1
           (defined in 01_data_generator.py as shift(-1) > 0).

      The target for window i therefore is y[i + L - 1]:
        → uses all information up to and including day (i + L - 1)
        → predicts price movement on day (i + L)   ← strictly FUTURE

    ── API constraint & workaround ──────────────────────────────────────────

    tf.keras.utils.timeseries_dataset_from_array(data, targets, ...) pairs:
        window i  ←→  targets[i]

    We need  targets[i] = y[i + L - 1], i.e. y left-shifted by (L - 1).
    The API requires len(targets) == len(data) == N.

    Solution: build 'targets_aligned' of length N:
        targets_aligned[i]       = y[i + L - 1]   for i in [0, N-L]
        targets_aligned[N-L+1:]  = 0.0             ← padding, NEVER READ
    The function creates only (N - L + 1) windows, so the padded tail is
    silently ignored — this is mathematically verified in the docstring.

    Parameters
    ----------
    X         : np.ndarray (N, n_features) – scaled feature matrix.
    y         : np.ndarray (N,) – binary target labels (0 or 1).
    seq_len   : int  – number of look-back time-steps per window.
    batch_size: int  – number of windows per batch.
    shuffle   : bool – True only for the training dataset (random mini-batches).

    Returns
    -------
    dataset   : tf.data.Dataset yielding (window_batch, label_batch) tuples.
    y_aligned : np.ndarray of the targets actually used by the dataset,
                i.e. y[seq_len - 1:] — needed for metric computation outside
                the TF graph.
    """
    N         = len(X)
    n_windows = N - seq_len + 1     # number of valid windows the API will create

    # Construct the target array aligned with window start indices.
    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[seq_len - 1:]   # y[L-1], y[L], ..., y[N-1]
    targets_aligned[n_windows:] = 0.0               # unused padding (padded tail)

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X.astype(np.float32),
        targets=targets_aligned,
        sequence_length=seq_len,
        sequence_stride=1,                           # step = 1 → dense windows
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )

    # y_aligned mirrors the labels that will be yielded by the dataset —
    # use it for scikit-learn metric functions outside the TF graph.
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
    ---------------------
    Input  :  (batch_size, SEQ_LEN, n_features)  – one window per sample

    LSTM 1 :  64 units, return_sequences=True
              → outputs shape (batch_size, SEQ_LEN, 64)
              → return_sequences=True passes the full hidden-state sequence
                to the second LSTM layer, allowing it to learn higher-level
                temporal patterns from the representations of the first layer.

    Dropout 1: 40 % drop rate
              → Randomly zeros neuron outputs during training, forcing the
                network to learn redundant representations (Srivastava 2014).

    LSTM 2 :  32 units, return_sequences=False
              → outputs shape (batch_size, 32) — the final hidden state
              → compresses the temporal sequence into a fixed-size vector.

    Dropout 2: 40 % drop rate

    Dense  :  1 unit, sigmoid activation
              → outputs P(BTC closes higher tomorrow) ∈ (0, 1)

    Anti-overfitting measures (stacked)
    ------------------------------------
    1. L2 weight decay (kernel_regularizer=L2Reg(L2_FACTOR)):
       Adds λ·Σ|w|² to the loss. Penalises large weights → smaller, smoother
       decision boundaries → reduces variance.

    2. Dropout (DROPOUT_RATE = 0.40):
       Approximately trains an ensemble of 2^N subnetworks on each batch.
       The geometric average of their outputs approximates Bayesian inference.

    3. EarlyStopping (configured in train_model):
       Halts training when val_loss stops improving → avoids over-training
       past the bias-variance optimum.

    4. ReduceLROnPlateau (configured in train_model):
       Decays the learning rate when training plateaus → finer convergence
       in flat loss regions without re-introducing overfitting momentum.

    Parameters
    ----------
    seq_len    : int – number of time-steps in each input sequence (SEQ_LEN).
    n_features : int – dimensionality of each time-step's feature vector.

    Returns
    -------
    model : tf.keras.Model — compiled, ready to call model.fit().
    """
    model = Sequential(
        [
            # ── Layer 1: Stacked LSTM with sequence output ────────────────────
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,          # pass sequence to next LSTM layer
                kernel_regularizer=L2Reg(L2_FACTOR),
                input_shape=(seq_len, n_features),
                name="lstm_layer_1",
            ),
            Dropout(DROPOUT_RATE, name="dropout_1"),

            # ── Layer 2: Final LSTM with single vector output ─────────────────
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,         # only output the last hidden state
                kernel_regularizer=L2Reg(L2_FACTOR),
                name="lstm_layer_2",
            ),
            Dropout(DROPOUT_RATE, name="dropout_2"),

            # ── Output layer: sigmoid for binary probability ──────────────────
            # sigmoid(z) = 1 / (1 + e^{-z})  maps z → (0, 1)
            # Interpreted as → P(BTC return on day t+1 > 0 | features[t-L+1..t])
            Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="btc_direction_lstm",
    )

    # Binary cross-entropy is the canonical loss for Bernoulli targets.
    # Adam adaptive learning rate is the industry standard for deep learning.
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
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
) -> tf.keras.callbacks.History:
    """
    Train the LSTM with two stacked early-stopping callbacks.

    EarlyStopping  (patience = 20 epochs)
        Monitors 'val_loss'. If it does not improve for 20 consecutive
        epochs, training stops and the best weights are restored.
        This is the primary guard against overfitting.

    ReduceLROnPlateau  (patience = 8 epochs, factor = 0.5)
        Halves the learning rate after 8 non-improving epochs.
        Smaller lr → smaller gradient steps → finer convergence in
        flat regions without oscillating into a poor local minimum.
        Triggers BEFORE EarlyStopping, gracefully slowing descent first.

    Parameters
    ----------
    model    : compiled tf.keras.Model.
    train_ds : tf.data.Dataset for training (shuffle=True).
    val_ds   : tf.data.Dataset for validation (shuffle=False).

    Returns
    -------
    history : tf.keras.callbacks.History – per-epoch train/val metrics.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,   # revert to the epoch with lowest val_loss
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
    best_val   = min(history.history["val_loss"])
    log.info(
        "Training complete. Best val_loss = %.5f at epoch %d / %d.",
        best_val, best_epoch, len(history.history["val_loss"]),
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

    Metrics calculated
    ------------------
    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
                  Overall fraction of correct predictions.

    Sensitivity = TP / (TP + FN)   [also: Recall, True Positive Rate]
                  Of all actual UP days, what fraction did we correctly
                  predict as UP? High sensitivity → few missed long signals.

    Specificity = TN / (TN + FP)   [also: True Negative Rate]
                  Of all actual DOWN days, what fraction did we correctly
                  predict as DOWN? High specificity → few false alarms.

    ROC AUC     = Area Under the ROC Curve.
                  Threshold-independent measure of ranking quality.
                  AUC = 0.5 → random guessing; AUC = 1.0 → perfect model.

    Gini        = 2 × AUC − 1
                  Normalises AUC to [−1, 1]; used widely in credit scoring
                  and quantitative finance to compare discriminatory power.

    Parameters
    ----------
    y_true     : Ground-truth binary labels (0 or 1).
    y_prob     : Predicted probabilities for class 1 (BTC up).
    split_name : 'Train' / 'Val' / 'Test' — printed to console.

    Returns
    -------
    dict with keys: accuracy, sensitivity, specificity, auc, gini.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc         = roc_auc_score(y_true, y_prob)
    gini        = 2.0 * auc - 1.0

    metrics = {
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc":         auc,
        "gini":        gini,
    }

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

    return metrics


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

    Rationale
    ---------
    A binary classifier forced to predict every day will inevitably place
    low-confidence trades. In a real trading system, you only act when the
    edge is clear. The 3-state filter implements this by requiring the model
    to exceed a probability threshold before signalling a position:

        P(up) > UPPER_THRESHOLD  →  BUY  (long BTC)
        P(up) < LOWER_THRESHOLD  →  SELL (short BTC / exit)
        otherwise                →  HOLD (no trade taken)

    Metrics computed on traded days only
    ─────────────────────────────────────
    Coverage          = n_traded / n_total
                        Fraction of days the model is willing to trade.
                        A very low coverage may mean the thresholds are too
                        wide and the model rarely acts.

    Conditional Win Rate = Accuracy(traded days)
                        Accuracy restricted to days where the model had
                        high confidence. Tells us: "when the model commits
                        to a trade, how often is it right?"
                        A high CWR with reasonable coverage is the ideal
                        outcome — it means the model knows what it knows.

    Parameters
    ----------
    y_true     : Ground-truth binary labels (0 or 1).
    y_prob     : Predicted probabilities for class 1 (BTC up).
    split_name : 'Train' / 'Val' / 'Test'.

    Returns
    -------
    dict with keys: coverage, conditional_win_rate, n_traded.
    """
    # A 'traded' day is one where the model has sufficient conviction.
    trade_mask = (y_prob > UPPER_THRESHOLD) | (y_prob < LOWER_THRESHOLD)
    n_traded   = int(trade_mask.sum())
    n_total    = len(y_prob)
    coverage   = n_traded / n_total if n_total > 0 else 0.0

    if n_traded == 0:
        log.warning(
            "[%s] Zero days exceed the threshold bands "
            "(UPPER=%.2f, LOWER=%.2f). "
            "Consider widening the threshold gap.",
            split_name, UPPER_THRESHOLD, LOWER_THRESHOLD,
        )
        return {"coverage": 0.0, "conditional_win_rate": float("nan"), "n_traded": 0}

    y_pred_traded = (y_prob[trade_mask] >= 0.5).astype(int)
    y_true_traded = y_true[trade_mask].astype(int)
    cond_win_rate = accuracy_score(y_true_traded, y_pred_traded)

    metrics = {
        "coverage":             coverage,
        "conditional_win_rate": cond_win_rate,
        "n_traded":             n_traded,
    }

    print(f"{'═' * 58}")
    print(f"  Variant B — 3-State Evaluation  [{split_name} Set]")
    print(f"  Thresholds:  UPPER={UPPER_THRESHOLD:.2f}  |  LOWER={LOWER_THRESHOLD:.2f}")
    print(f"{'─' * 58}")
    print(f"  Total days    :  {n_total}")
    print(f"  Traded days   :  {n_traded}  ({coverage * 100:.1f}% coverage)")
    print(f"  Hold days     :  {n_total - n_traded}  ({(1-coverage) * 100:.1f}%)")
    print(f"  Cond. Win Rate:  {cond_win_rate:.4f}  (accuracy on traded days)")
    print(f"{'═' * 58}\n")

    return metrics


# =============================================================================
# STEP 7 — VISUALIZATIONS
# =============================================================================

def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Save a 2-panel figure displaying loss and accuracy curves for both
    the training and validation sets across all completed epochs.

    How to diagnose overfitting from this plot
    -------------------------------------------
    •  Healthy fit  :  Train and val curves converge and stay close.
    •  Overfitting  :  Train loss keeps falling; val loss diverges upward.
                       The gap between them is the 'generalisation gap'.
    •  Early stop ✓ :  The val curve plateaus, then EarlyStopping fires —
                       look for where the curves end vs. EPOCHS maximum.
    •  Underfitting :  Both curves plateau at a high loss value; the model
                       lacks capacity to capture the signal.
    """
    # Epoch axis (1-indexed for readability)
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"LSTM Training History — BTC Daily Direction Classifier\n"
        f"Architecture: {LSTM_UNITS_1}→{LSTM_UNITS_2} units | "
        f"L2={L2_FACTOR} | Dropout={DROPOUT_RATE} | "
        f"SeqLen={SEQ_LEN}",
        fontsize=11,
    )

    # ── Panel 1: Binary Cross-Entropy Loss ────────────────────────────────────
    axes[0].plot(
        epochs_range, history.history["loss"],
        lw=2, color="steelblue", label="Train Loss",
    )
    axes[0].plot(
        epochs_range, history.history["val_loss"],
        lw=2, color="tomato", ls="--", label="Validation Loss",
    )
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Panel 2: Accuracy ─────────────────────────────────────────────────────
    # TF2 Keras stores accuracy under the key 'accuracy' (not 'acc').
    acc_key     = "accuracy"
    val_acc_key = "val_accuracy"

    axes[1].plot(
        epochs_range, history.history[acc_key],
        lw=2, color="steelblue", label="Train Accuracy",
    )
    axes[1].plot(
        epochs_range, history.history[val_acc_key],
        lw=2, color="tomato", ls="--", label="Validation Accuracy",
    )
    axes[1].set_title("Classification Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("training_history.png", dpi=150)
    plt.close(fig)
    log.info("Training history plot saved → training_history.png")


def plot_roc_curve(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    auc_score: float,
) -> None:
    """
    Save a ROC curve with AUC annotation evaluated on the TEST set.

    What the ROC curve shows
    ------------------------
    The curve plots True Positive Rate (Sensitivity) on the y-axis against
    False Positive Rate (1 − Specificity) on the x-axis, sweeping across
    ALL possible classification thresholds simultaneously.

    •  Diagonal line  →  AUC = 0.5  →  model is indistinguishable from
                         random guessing (coin flip).
    •  Top-left corner →  AUC = 1.0  →  perfect classifier.
    •  Our model curve →  area between the curve and the diagonal = Gini/2.

    The AUC is a threshold-independent metric: it measures how well the
    model ranks UP days above DOWN days in probability, regardless of where
    exactly the 0.5 decision boundary sits.

    Parameters
    ----------
    y_true    : Ground-truth binary labels for the test set.
    y_prob    : Predicted probabilities for class 1 on the test set.
    auc_score : Pre-computed AUC for the annotation text.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gini = 2.0 * auc_score - 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr, tpr,
        lw=2.5, color="steelblue",
        label=f"LSTM  (AUC = {auc_score:.4f}  |  Gini = {gini:.4f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.50)")
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
    Save a seaborn confusion matrix heatmap with both raw counts and
    row-normalised percentages in each cell.

    Layout
    ------
                   Predicted: DOWN   Predicted: UP
    True: DOWN  [  TN (row %)    |   FP (row %)  ]
    True: UP    [  FN (row %)    |   TP (row %)  ]

    Row-normalisation (dividing each row by the row total) converts raw
    counts to per-class recall percentages. This is more informative than
    absolute counts when classes are imbalanced, because it directly shows:
        Row 0 → Specificity (how often we correctly label DOWN days)
        Row 1 → Sensitivity (how often we correctly label UP days)

    Parameters
    ----------
    y_true     : Ground-truth binary labels.
    y_prob     : Predicted probabilities.
    split_name : 'Val' or 'Test' — used in the title and filename.
    """
    y_pred  = (y_prob >= 0.5).astype(int)
    cm      = confusion_matrix(y_true.astype(int), y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalise

    # Build a string annotation array: "count\n(percentage)"
    annots = np.array(
        [
            [f"{count}\n({pct:.1%})" for count, pct in zip(row_c, row_p)]
            for row_c, row_p in zip(cm, cm_norm)
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,                            # colour-code by row-normalised %
        annot=annots,                       # show "count (pct)" in each cell
        fmt="",                             # suppress default number formatting
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        xticklabels=["Pred: DOWN (0)", "Pred: UP (1)"],
        yticklabels=["True: DOWN (0)", "True: UP (1)"],
        annot_kws={"size": 12},
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix — {split_name} Set\n"
        f"(Threshold = 0.50 | Colour = Row-Normalised Rate)",
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
# MAIN — Orchestrate the full model pipeline
# =============================================================================

def main() -> None:
    """Run the complete LSTM training, evaluation, and visualisation pipeline."""
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
    log.info("[STEP 1/7] Loading and splitting dataset ...")
    train_df, val_df, test_df = load_and_split(INPUT_CSV)

    # ── STEP 2: Scale ─────────────────────────────────────────────────────────
    log.info("[STEP 2/7] Scaling features (fit on Train only) ...")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, feature_cols,
    ) = scale_features(train_df, val_df, test_df)

    n_features = X_train.shape[1]
    log.info("n_features = %d  |  feature list: %s", n_features, feature_cols)

    # ── STEP 3: Build tf.data.Datasets ───────────────────────────────────────
    log.info("[STEP 3/7] Building windowed tf.data.Datasets (SEQ_LEN=%d) ...", SEQ_LEN)

    # Shuffled training dataset — used for gradient updates during model.fit().
    # Shuffling window start-indices prevents the model from over-fitting to
    # the temporal ordering of batches.
    train_ds_fit, _ = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=True
    )

    # UN-shuffled training dataset — used ONLY for metric computation after
    # training. Keeping temporal order ensures predictions align with y_aligned.
    train_ds_eval, y_train_aligned = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE, shuffle=False
    )
    val_ds,  y_val_aligned  = make_tf_dataset(X_val,  y_val,  SEQ_LEN, BATCH_SIZE)
    test_ds, y_test_aligned = make_tf_dataset(X_test, y_test, SEQ_LEN, BATCH_SIZE)

    log.info(
        "Windowed set sizes → Train: %d | Val: %d | Test: %d windows",
        len(y_train_aligned), len(y_val_aligned), len(y_test_aligned),
    )

    # ── STEP 4: Build & Train Model ───────────────────────────────────────────
    log.info("[STEP 4/7] Building model architecture ...")
    model = build_model(SEQ_LEN, n_features)

    log.info("[STEP 4/7] Training LSTM (EarlyStopping active) ...")
    history = train_model(model, train_ds_fit, val_ds)

    # ── STEP 5: Generate Predictions ─────────────────────────────────────────
    log.info("[STEP 5/7] Generating predictions on all splits ...")
    # model.predict returns shape (N, 1); flatten to (N,) for sklearn metrics.
    prob_train = model.predict(train_ds_eval, verbose=0).flatten()
    prob_val   = model.predict(val_ds,        verbose=0).flatten()
    prob_test  = model.predict(test_ds,       verbose=0).flatten()

    # ── STEP 6A: Variant A — Standard Evaluation ──────────────────────────────
    log.info("[STEP 6/7] Running Variant A (standard 0.5 threshold) ...")

    all_metrics_a = {}
    for split_name, y_true, y_prob in [
        ("Train", y_train_aligned, prob_train),
        ("Val",   y_val_aligned,   prob_val),
        ("Test",  y_test_aligned,  prob_test),
    ]:
        all_metrics_a[split_name] = evaluate_standard(y_true, y_prob, split_name)

    # ── STEP 6B: Variant B — 3-State Evaluation ───────────────────────────────
    log.info("[STEP 6/7] Running Variant B (3-state confidence filter) ...")

    all_metrics_b = {}
    for split_name, y_true, y_prob in [
        ("Train", y_train_aligned, prob_train),
        ("Val",   y_val_aligned,   prob_val),
        ("Test",  y_test_aligned,  prob_test),
    ]:
        all_metrics_b[split_name] = evaluate_3state(y_true, y_prob, split_name)

    # ── STEP 7: Visualizations ────────────────────────────────────────────────
    log.info("[STEP 7/7] Generating and saving diagnostic plots ...")

    plot_training_history(history)

    # ROC curve computed only on the Test set (Val is used for model selection,
    # so its ROC would be optimistically biased).
    test_auc = all_metrics_a["Test"]["auc"]
    plot_roc_curve(y_test_aligned, prob_test, test_auc)

    plot_confusion_matrix(y_val_aligned,  prob_val,  "Val")
    plot_confusion_matrix(y_test_aligned, prob_test, "Test")

    # ── Final summary printout ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 65)
    print(f"  {'Split':<8}  {'Accuracy':>9}  {'AUC':>7}  {'Gini':>7}  "
          f"{'Sens.':>7}  {'Spec.':>7}")
    print("  " + "─" * 55)
    for split_name in ["Train", "Val", "Test"]:
        m = all_metrics_a[split_name]
        print(
            f"  {split_name:<8}  {m['accuracy']:>9.4f}  {m['auc']:>7.4f}  "
            f"{m['gini']:>7.4f}  {m['sensitivity']:>7.4f}  "
            f"{m['specificity']:>7.4f}"
        )
    print("  " + "─" * 55)
    print(f"\n  3-State (Variant B) — Test Set:")
    mb = all_metrics_b["Test"]
    print(f"    Coverage        : {mb['coverage']:.4f} "
          f"({mb['coverage']*100:.1f}% of test days traded)")
    print(f"    Cond. Win Rate  : {mb['conditional_win_rate']:.4f}")
    print("═" * 65 + "\n")

    log.info("02_lstm_model.py  —  PIPELINE COMPLETE")
    log.info(
        "Saved artefacts: training_history.png | roc_curve_test.png | "
        "confusion_matrix_val.png | confusion_matrix_test.png"
    )


# =============================================================================
if __name__ == "__main__":
    main()
