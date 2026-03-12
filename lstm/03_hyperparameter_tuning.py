#!/usr/bin/env python3
# =============================================================================
# 03_hyperparameter_tuning.py
# =============================================================================
# PURPOSE:
#   Run an Optuna overnight hyperparameter search over a constrained LSTM
#   architecture space for the BTC daily direction classifier.
#   Objective: MAXIMIZE Val ROC AUC across N_TRIALS trials.
#   All Variant A + Variant B metrics for both Train and Val are tracked
#   per trial via trial.set_user_attr() and exported at the end.
#
# DESIGN PRINCIPLES:
#   • Data loaded ONCE in main() and injected via closure — avoids N_TRIALS
#     redundant CSV reads and StandardScaler refits.
#   • NO test set is ever loaded — strict holdout is preserved for the final
#     reported run in LSTM_model_1.02.py.
#   • units_2 is enforced ≤ units_1 inside the objective to prevent
#     architecturally nonsensical expanding bottlenecks.
#   • EarlyStopping patience=5 aggressively kills bad trials early, which
#     partially compensates for the cuDNN slowdown from recurrent_dropout.
#   • SQLite storage provides crash-safe persistence: if the process is killed
#     mid-run (power cut, OOM, etc.), all completed trials are preserved.
#     Re-running the script resumes the same study automatically.
#   • KeyboardInterrupt is handled at two levels for robustness:
#       (a) catch=(KeyboardInterrupt,) in study.optimize — marks the
#           interrupted trial as FAILED and allows the study to proceed
#           cleanly to the export step.
#       (b) outer try/except KeyboardInterrupt — catches any interrupt that
#           escapes the Optuna loop (e.g., signal received between trials).
#
# OUTPUTS:
#   • optuna_study.db      — SQLite database (all trials, crash-safe)
#   • tuning_results.csv   — full trials dataframe (params + all metrics)
#   • best_params.json     — best trial's hyperparameters
#   • run_log_tuning.txt   — full session log (mirrors stdout)
#
# REQUIREMENTS:
#   pip install optuna tensorflow pandas numpy scikit-learn
#
# OVERNIGHT USAGE:
#   python 03_hyperparameter_tuning.py
#   → Press Ctrl+C at any point to stop cleanly and trigger export.
#   → Re-run to resume from where Optuna left off (SQLite stores all trials).
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import sys
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import optuna

import tensorflow as tf
from tensorflow.keras import Sequential          # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.regularizers import l2 as L2Reg     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Suppress TensorFlow's verbose startup messages so the trial log is readable.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =============================================================================
# ███  GLOBAL CONFIGURATION  ███
# Edit ONLY this block to change the study configuration.
# =============================================================================

# ── Chronological date splits ─────────────────────────────────────────────────
# STRICTLY no test set. Dates match LSTM_model_1.02.py to ensure comparability.
TRAIN_START: str = "2016-01-01"
TRAIN_END:   str = "2022-12-31"
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"
# TEST set intentionally absent. Load it ONLY in 02_lstm_model.py at final run.

# ── Data paths ────────────────────────────────────────────────────────────────
INPUT_CSV:          str = "processed_dataset.csv"
STUDY_DB_PATH:      str = "sqlite:///optuna_study.db"   # persistent storage URI
RESULTS_CSV_PATH:   str = "tuning_results.csv"
BEST_PARAMS_PATH:   str = "best_params.json"
LOG_FILE_PATH:      str = "run_log_tuning.txt"

# ── Study configuration ───────────────────────────────────────────────────────
STUDY_NAME:  str = "btc_lstm_v1"   # fixed name enables resume-from-DB on rerun
N_TRIALS:    int = 200             # total budget; Ctrl+C stops early gracefully

# ── Fixed architectural constants (not tuned) ─────────────────────────────────
# These are fixed across all trials to reduce the search space dimensionality.
SEQ_LEN:    int = 20    # look-back window length (days); tuned separately if needed
BATCH_SIZE: int = 64    # mini-batch size

# ── Per-trial training limits ─────────────────────────────────────────────────
# EPOCHS is a hard ceiling; EarlyStopping(patience=5) fires well before this.
# Kept low (50) so that mediocre trials fail fast and the overnight budget is
# distributed across a wider hyperparameter coverage.
EPOCHS_PER_TRIAL: int = 50

# ── Variant B fixed thresholds (same as production script) ───────────────────
UPPER_THRESHOLD: float = 0.55
LOWER_THRESHOLD: float = 0.45

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# LOGGING SETUP
# Mirrors stdout to a file so the overnight session is fully auditable.
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8"),
        # Append mode ('a'): resuming the study appends to the existing log,
        # making the full overnight history readable in one file.
    ],
)
log = logging.getLogger(__name__)

# Suppress Optuna's default per-trial INFO messages — we use a custom callback
# that is more informative (includes all metrics, not just the objective value).
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# STEP 1 — DATA PIPELINE
# Loaded ONCE in main() and passed to objective via closure.
# This avoids N_TRIALS × CSV read / StandardScaler fit overhead.
# =============================================================================

def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Load the processed dataset, slice Train and Val by exact date boundaries,
    apply StandardScaler (fit on Train only), and build windowed tf.data
    Datasets ready for model.fit().

    ── Strict holdout guarantee ─────────────────────────────────────────────
    The test slice is never extracted here. The boolean mask only covers
    TRAIN_START–TRAIN_END and VAL_START–VAL_END.

    ── StandardScaler protocol (no leakage) ─────────────────────────────────
    scaler.fit(X_train)    → computes μ/σ from training observations only.
    scaler.transform(X_val) → applies training μ/σ; no future info embedded.

    ── Data returned ─────────────────────────────────────────────────────────
    Returns both the raw numpy arrays (for manual metric computation) and
    the pre-built tf.data.Datasets (for model.fit / model.predict).

    Returns
    -------
    X_train, X_val          : np.ndarray — scaled feature matrices
    y_train, y_val          : np.ndarray — binary targets (unaligned, length N)
    train_ds_fit            : tf.data.Dataset — shuffled, for model.fit()
    train_ds_eval           : tf.data.Dataset — ordered, for metric computation
    val_ds                  : tf.data.Dataset — ordered, for val_loss + metrics
    y_train_aligned         : np.ndarray — y_train[SEQ_LEN-1:], mirrors train_ds_eval
    y_val_aligned           : np.ndarray — y_val[SEQ_LEN-1:],   mirrors val_ds
    feature_cols            : list[str]
    scaler                  : fitted StandardScaler (for inspection / export)
    """
    log.info("Loading dataset from '%s' ...", csv_path)
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)

    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df.index.min().date(), df.index.max().date(),
    )

    # ── Date-based slicing ────────────────────────────────────────────────────
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    val_mask   = (df.index >= VAL_START)   & (df.index <= VAL_END)

    train_df = df[train_mask]
    val_df   = df[val_mask]

    for label, split in [("Train", train_df), ("Val", val_df)]:
        log.info(
            "  %-5s → %5d rows  [%s → %s]",
            label, len(split),
            split.index.min().date(), split.index.max().date(),
        )

    # ── Feature / target separation ───────────────────────────────────────────
    feature_cols = [c for c in train_df.columns if c != "target"]
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    y_train     = train_df["target"].values.astype(np.float32)
    y_val       = val_df["target"].values.astype(np.float32)

    # ── StandardScaler — fit on Train only ────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)    # NO refit — preserves leakage-free protocol

    log.info(
        "Scaler fitted on Train. Feature means: %s",
        dict(zip(feature_cols, np.round(scaler.mean_, 5))),
    )

    # ── Windowed tf.data.Datasets ─────────────────────────────────────────────
    train_ds_fit,  _               = _make_dataset(X_train, y_train, shuffle=True)
    train_ds_eval, y_train_aligned = _make_dataset(X_train, y_train, shuffle=False)
    val_ds,        y_val_aligned   = _make_dataset(X_val,   y_val,   shuffle=False)

    log.info(
        "Windows — Train: %d | Val: %d",
        len(y_train_aligned), len(y_val_aligned),
    )

    return (
        X_train, X_val, y_train, y_val,
        train_ds_fit, train_ds_eval, val_ds,
        y_train_aligned, y_val_aligned,
        feature_cols, scaler,
    )


def _make_dataset(
    X:       np.ndarray,
    y:       np.ndarray,
    shuffle: bool = False,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows with correct label alignment.

    ── Window ↔ Target alignment (no look-ahead bias) ───────────────────────
    Window i  covers  features[i : i+SEQ_LEN]
    Target for window i  =  y[i + SEQ_LEN - 1]
        → uses info through day (i + SEQ_LEN - 1)
        → predicts BTC return on day (i + SEQ_LEN)  ← strictly future

    targets_aligned[i] = y[i + SEQ_LEN - 1] ensures the API's default pairing
    (targets[i] ↔ window i) produces the correct label without any offset.

    Returns
    -------
    dataset   : tf.data.Dataset
    y_aligned : np.ndarray — y[SEQ_LEN-1:], mirrors dataset label order.
    """
    N         = len(X)
    n_windows = N - SEQ_LEN + 1

    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[SEQ_LEN - 1:]
    targets_aligned[n_windows:] = 0.0   # padding tail — never consumed by the API

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X.astype(np.float32),
        targets=targets_aligned,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )

    y_aligned = y[SEQ_LEN - 1:].astype(np.float32)
    return dataset, y_aligned


# =============================================================================
# STEP 2 — MODEL BUILDER
# Constructs a fresh model for each trial using trial-suggested parameters.
# =============================================================================

def build_trial_model(
    units_1:            int,
    units_2:            int,
    dropout:            float,
    recurrent_dropout:  float,
    l2_factor:          float,
    lr:                 float,
    n_features:         int,
) -> tf.keras.Model:
    """
    Build and compile a two-layer stacked LSTM with the given hyperparameters.

    Note on cuDNN
    ─────────────
    Any recurrent_dropout > 0 disables TensorFlow's cuDNN LSTM kernel,
    falling back to the slower unoptimised implementation. This is an
    unavoidable trade-off for the regularisation benefit. EarlyStopping
    patience=5 kills low-quality trials before the speed cost compounds.

    Parameters
    ----------
    units_1, units_2   : LSTM hidden-state sizes for layers 1 and 2.
    dropout            : Inter-layer dropout fraction (after each LSTM layer).
    recurrent_dropout  : Recurrent kernel dropout fraction (within LSTM cells).
    l2_factor          : L2 weight decay applied to LSTM kernel matrices.
    lr                 : Adam initial learning rate.
    n_features         : Input feature dimensionality (determined by data).

    Returns
    -------
    model : compiled tf.keras.Model, ready for model.fit().
    """
    model = Sequential([
        LSTM(
            units=units_1,
            return_sequences=True,          # pass full sequence to layer 2
            kernel_regularizer=L2Reg(l2_factor),
            recurrent_dropout=recurrent_dropout,
            input_shape=(SEQ_LEN, n_features),
            name="lstm_1",
        ),
        Dropout(dropout, name="drop_1"),

        LSTM(
            units=units_2,
            return_sequences=False,         # compress to single vector
            kernel_regularizer=L2Reg(l2_factor),
            recurrent_dropout=recurrent_dropout,
            name="lstm_2",
        ),
        Dropout(dropout, name="drop_2"),

        Dense(1, activation="sigmoid", name="output"),
    ], name=f"lstm_{units_1}_{units_2}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# STEP 3 — VARIANT METRIC HELPERS
# Isolated helpers so the objective function stays clean and readable.
# =============================================================================

def _variant_a_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute Variant A (standard 0.5 threshold) metrics.

    Returns
    -------
    dict with keys: auc, gini, accuracy, sensitivity, specificity.
    """
    from sklearn.metrics import confusion_matrix as cm_fn

    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    acc    = accuracy_score(y_true.astype(int), y_pred)

    tn, fp, fn, tp = cm_fn(y_true.astype(int), y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return dict(
        auc         = float(auc),
        gini        = float(2.0 * auc - 1.0),
        accuracy    = float(acc),
        sensitivity = float(sensitivity),
        specificity = float(specificity),
    )


def _variant_b_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    upper:  float = UPPER_THRESHOLD,
    lower:  float = LOWER_THRESHOLD,
) -> dict:
    """
    Compute Variant B (3-state confidence filter) metrics.

    Returns
    -------
    dict with keys: coverage, conditional_win_rate, n_traded.
    """
    trade_mask = (y_prob > upper) | (y_prob < lower)
    n_traded   = int(trade_mask.sum())
    n_total    = len(y_prob)
    coverage   = float(n_traded / n_total) if n_total > 0 else 0.0

    if n_traded > 0:
        cwr = float(accuracy_score(
            y_true[trade_mask].astype(int),
            (y_prob[trade_mask] >= 0.5).astype(int),
        ))
    else:
        cwr = float("nan")

    return dict(
        coverage             = coverage,
        conditional_win_rate = cwr,
        n_traded             = n_traded,
    )


# =============================================================================
# STEP 4 — OPTUNA OBJECTIVE FUNCTION
# Called once per trial. Returns the value to MAXIMISE (Val AUC).
# =============================================================================

def make_objective(
    X_train:         np.ndarray,
    X_val:           np.ndarray,
    train_ds_fit:    "tf.data.Dataset",
    train_ds_eval:   "tf.data.Dataset",
    val_ds:          "tf.data.Dataset",
    y_train_aligned: np.ndarray,
    y_val_aligned:   np.ndarray,
):
    """
    Factory that builds and returns the Optuna objective function.

    Using a factory (closure) injects the pre-computed data into the
    objective without global variables, keeping the function pure and
    testable. Optuna calls objective(trial) for each trial; the datasets
    and aligned labels are captured from the enclosing scope.

    Parameters
    ----------
    All pre-processed data arrays and Datasets from load_and_prepare_data().

    Returns
    -------
    objective : Callable[[optuna.Trial], float]
        Takes a trial, suggests hyperparameters, trains a model, logs all
        metrics as user attributes, and returns Val AUC as the objective.
    """
    n_features = X_train.shape[1]

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: suggest → build → train → evaluate → log → return.

        ── Hyperparameter search space ───────────────────────────────────────

        units_1: {128, 256, 512}
            First LSTM layer size. Higher capacity → more potential signal
            extraction, but also more overfitting risk on small financial datasets.

        units_2: {64, 128, 256}  (enforced ≤ units_1)
            Second LSTM layer size. The constraint prevents expanding bottlenecks
            (e.g., 128 → 256), which are architecturally nonsensical for a
            compression-then-classify pattern and tend to overfit.

        dropout: [0.1, 0.2, 0.3, 0.4]
            Inter-layer dropout. Step=0.1 keeps the search discrete and
            interpretable; finer resolution yields diminishing returns here.

        recurrent_dropout: [0.1, 0.2, 0.3, 0.4]
            Recurrent kernel dropout. Regularises h_{t-1} connections.
            Higher values → stronger temporal regularisation, slower training.

        l2_factor: {1e-5, 1e-4, 1e-3}
            Kernel weight decay. Log-scale coverage from light to heavy.

        lr: {5e-4, 1e-3, 2e-3}
            Adam initial learning rate. Categorical rather than continuous
            to avoid wasting budget on minutely different LR values.

        ── Return value ──────────────────────────────────────────────────────
        Val AUC (float) — Optuna maximises this.
        Returns -1.0 on any exception to mark the trial as failed without
        crashing the study.
        """
        # ── 4A: Suggest hyperparameters ───────────────────────────────────────
        units_1           = trial.suggest_categorical("units_1",           [128, 256, 512])
        units_2           = trial.suggest_categorical("units_2",           [64, 128, 256])
        dropout           = trial.suggest_float(      "dropout",           0.1, 0.4, step=0.1)
        recurrent_dropout = trial.suggest_float(      "recurrent_dropout", 0.1, 0.4, step=0.1)
        l2_factor         = trial.suggest_categorical("l2_factor",         [1e-5, 1e-4, 1e-3])
        lr                = trial.suggest_categorical("lr",                [5e-4, 1e-3, 2e-3])

        # ── 4B: Enforce units_2 ≤ units_1 (no expanding bottleneck) ──────────
        # Optuna's sampler is unaware of cross-parameter constraints, so it
        # will occasionally suggest units_2 > units_1. We fix it inline rather
        # than using optuna.create_trial constraints (which require sampling
        # rejection and waste trial budget).
        #
        #   units_1=128, units_2=256 → units_2 forced to 64  (units_1 // 2)
        #   units_1=256, units_2=256 → unchanged (equal is fine)
        #   units_1=512, units_2=128 → unchanged (valid compression)
        if units_2 > units_1:
            units_2 = max(64, units_1 // 2)   # halve; floor at 64 to avoid trivial size

        # Log the (possibly corrected) architecture before training.
        trial.set_user_attr("units_2_effective", int(units_2))
        log.debug(
            "  Trial %d — arch: [%d→%d]  drop=%.1f  recdrop=%.1f  "
            "l2=%g  lr=%g",
            trial.number, units_1, units_2, dropout,
            recurrent_dropout, l2_factor, lr,
        )

        # ── 4C: Build and train model ─────────────────────────────────────────
        # Wrap in try/except so a single trial failure (OOM, NaN loss, etc.)
        # is logged as a failed trial without aborting the entire study.
        try:
            # Fresh model per trial — no weight sharing across trials.
            model = build_trial_model(
                units_1           = units_1,
                units_2           = units_2,
                dropout           = dropout,
                recurrent_dropout = recurrent_dropout,
                l2_factor         = l2_factor,
                lr                = lr,
                n_features        = n_features,
            )

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,                  # tight: fail bad trials fast
                    restore_best_weights=True,
                    verbose=0,                   # silent — no terminal spam
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,                  # fires before EarlyStopping
                    min_lr=1e-7,
                    verbose=0,
                ),
            ]

            history = model.fit(
                train_ds_fit,
                epochs=EPOCHS_PER_TRIAL,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=0,                       # silent per-epoch output
            )

            actual_epochs = len(history.history["val_loss"])
            best_val_loss = float(min(history.history["val_loss"]))

            # ── 4D: Generate predictions ──────────────────────────────────────
            prob_train = model.predict(train_ds_eval, verbose=0).flatten()
            prob_val   = model.predict(val_ds,        verbose=0).flatten()

            # ── 4E: Compute Variant A metrics ─────────────────────────────────
            m_train_a = _variant_a_metrics(y_train_aligned, prob_train)
            m_val_a   = _variant_a_metrics(y_val_aligned,   prob_val)

            # ── 4F: Compute Variant B metrics ─────────────────────────────────
            m_train_b = _variant_b_metrics(y_train_aligned, prob_train)
            m_val_b   = _variant_b_metrics(y_val_aligned,   prob_val)

            # ── 4G: Compute overfitting gap ───────────────────────────────────
            # A large positive AUC gap is the primary diagnostic for the
            # overfitting we observed. Storing it per-trial makes it trivial
            # to sort/filter the results CSV by generalisation quality.
            auc_overfit_gap = m_train_a["auc"] - m_val_a["auc"]

            # ── 4H: Log all metrics as trial user attributes ──────────────────
            # trial.set_user_attr() persists values in the Optuna DB and in
            # the trials_dataframe() export. Prefix convention:
            #   train_ / val_ for split; _a_ / _b_ for variant.
            #
            # Training diagnostics
            trial.set_user_attr("actual_epochs",       actual_epochs)
            trial.set_user_attr("best_val_loss",       round(best_val_loss,    5))
            trial.set_user_attr("auc_overfit_gap",     round(auc_overfit_gap,  4))

            # Variant A — Train
            trial.set_user_attr("train_a_auc",         round(m_train_a["auc"],         4))
            trial.set_user_attr("train_a_gini",        round(m_train_a["gini"],        4))
            trial.set_user_attr("train_a_accuracy",    round(m_train_a["accuracy"],    4))
            trial.set_user_attr("train_a_sensitivity", round(m_train_a["sensitivity"], 4))
            trial.set_user_attr("train_a_specificity", round(m_train_a["specificity"], 4))

            # Variant A — Val  (the PRIMARY tracked metrics for thesis)
            trial.set_user_attr("val_a_auc",           round(m_val_a["auc"],           4))
            trial.set_user_attr("val_a_gini",          round(m_val_a["gini"],          4))
            trial.set_user_attr("val_a_accuracy",      round(m_val_a["accuracy"],      4))
            trial.set_user_attr("val_a_sensitivity",   round(m_val_a["sensitivity"],   4))
            trial.set_user_attr("val_a_specificity",   round(m_val_a["specificity"],   4))

            # Variant B — Train
            trial.set_user_attr("train_b_coverage",    round(m_train_b["coverage"],    4))
            trial.set_user_attr("train_b_cond_wr", (
                round(m_train_b["conditional_win_rate"], 4)
                if not np.isnan(m_train_b["conditional_win_rate"]) else -1.0
            ))
            trial.set_user_attr("train_b_n_traded",    int(m_train_b["n_traded"]))

            # Variant B — Val
            trial.set_user_attr("val_b_coverage",      round(m_val_b["coverage"],      4))
            trial.set_user_attr("val_b_cond_wr", (
                round(m_val_b["conditional_win_rate"], 4)
                if not np.isnan(m_val_b["conditional_win_rate"]) else -1.0
            ))
            trial.set_user_attr("val_b_n_traded",      int(m_val_b["n_traded"]))

            # ── 4I: Return the MAXIMISATION objective ─────────────────────────
            # Val AUC is the primary objective because:
            #   (a) It is threshold-independent — not artificially inflated by
            #       a lucky decision boundary at 0.5.
            #   (b) It is robust to the ~53/47 class imbalance.
            #   (c) It directly measures how well the model ranks UP vs DOWN
            #       days in probability space — the core quantitative skill.
            return float(m_val_a["auc"])

        except Exception as exc:
            # Log the exception but return -1.0 so the study continues.
            # Common causes: numerical instability (NaN loss), GPU OOM on
            # large architectures, or a rare tf.data pipeline error.
            log.warning(
                "Trial %d raised %s: %s — marking as FAILED.",
                trial.number, type(exc).__name__, str(exc),
            )
            return -1.0

    return objective


# =============================================================================
# STEP 5 — OPTUNA PROGRESS CALLBACK
# Prints a concise per-trial summary without replacing the default Optuna
# handler (which we silenced to avoid duplicate output).
# =============================================================================

def make_progress_callback(study_start_time: float):
    """
    Build an Optuna callback that logs a per-trial summary after each trial.

    The callback receives the study and the just-completed FrozenTrial.
    It extracts the key user attributes and prints a single structured line,
    making the overnight log easy to scan for promising trials.

    Parameters
    ----------
    study_start_time : float — time.perf_counter() at study start, used to
                               compute elapsed wall-clock time per trial.
    Returns
    -------
    callback : Callable[[optuna.Study, optuna.trial.FrozenTrial], None]
    """
    def callback(
        study:  optuna.Study,
        trial:  optuna.trial.FrozenTrial,
    ) -> None:
        # Skip logging for FAILED or PRUNED trials (value is None).
        if trial.value is None:
            log.warning(
                "Trial %3d  FAILED/PRUNED  —  "
                "arch: [%s→%s]  drop=%.1f  recdrop=%.1f  l2=%g  lr=%g",
                trial.number,
                trial.params.get("units_1", "?"),
                trial.params.get("units_2", "?"),
                trial.params.get("dropout", float("nan")),
                trial.params.get("recurrent_dropout", float("nan")),
                trial.params.get("l2_factor", float("nan")),
                trial.params.get("lr", float("nan")),
            )
            return

        elapsed = time.perf_counter() - study_start_time

        # Retrieve all user attrs with safe fallbacks for failed trials.
        ua = trial.user_attrs
        val_auc      = trial.value
        train_auc    = ua.get("train_a_auc",         float("nan"))
        gap          = ua.get("auc_overfit_gap",      float("nan"))
        val_b_cov    = ua.get("val_b_coverage",       float("nan"))
        val_b_cwr    = ua.get("val_b_cond_wr",        float("nan"))
        epochs_used  = ua.get("actual_epochs",        "?")
        units_2_eff  = ua.get("units_2_effective",
                               trial.params.get("units_2", "?"))

        # Is this the new best trial?
        is_best = (trial.number == study.best_trial.number)
        star    = "  ★ NEW BEST" if is_best else ""

        log.info(
            "Trial %3d  "
            "ValAUC=%.4f  TrainAUC=%.4f  Gap=%.4f  "
            "VarB_Cov=%.2f  VarB_CWR=%.4f  "
            "arch=[%d→%d]  drop=%.1f  recdrop=%.1f  l2=%g  lr=%g  "
            "epochs=%s  elapsed=%.0fs%s",
            trial.number,
            val_auc, train_auc, gap,
            val_b_cov, val_b_cwr,
            trial.params.get("units_1", 0), units_2_eff,
            trial.params.get("dropout", 0.0),
            trial.params.get("recurrent_dropout", 0.0),
            trial.params.get("l2_factor", 0.0),
            trial.params.get("lr", 0.0),
            epochs_used,
            elapsed,
            star,
        )

    return callback


# =============================================================================
# STEP 6 — RESULTS EXPORT
# Always called after the study ends (normally or via interrupt).
# =============================================================================

def export_results(study: optuna.Study) -> None:
    """
    Export the full trials dataframe to CSV and the best parameters to JSON.

    Called unconditionally from main() — whether the study ran to N_TRIALS
    completion, was interrupted by Ctrl+C, or stopped after a resumed run.

    ── trials_dataframe() structure ─────────────────────────────────────────
    Each row is one trial. Columns include:
        • params_*       – all suggested hyperparameters
        • user_attrs_*   – all set_user_attr() values (metrics, diagnostics)
        • value          – the objective value (Val AUC)
        • state          – COMPLETE / FAIL / PRUNED / RUNNING

    The CSV is the primary artefact for thesis analysis:
    sort by 'value' (Val AUC) descending to find the best configuration,
    sort by 'user_attrs_auc_overfit_gap' ascending to find the most
    generalisable architecture.

    Parameters
    ----------
    study : optuna.Study — the completed (or interrupted) study object.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log.info(
        "Study ended — %d total trials  (%d COMPLETE, %d FAIL/PRUNED).",
        len(study.trials),
        len(completed),
        len(study.trials) - len(completed),
    )

    if not completed:
        log.warning("No completed trials to export. Exiting without writing files.")
        return

    # ── Export full trials dataframe ─────────────────────────────────────────
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df_trials.to_csv(RESULTS_CSV_PATH, index=False)
    log.info("Full results exported → %s  (%d rows)", RESULTS_CSV_PATH, len(df_trials))

    # ── Export best parameters to JSON ───────────────────────────────────────
    best = study.best_trial
    best_params = {
        "trial_number":             best.number,
        "val_auc":                  round(best.value, 4),
        # Hyperparameters — as suggested (pre-correction).
        # units_2_effective holds the post-correction value if it was adjusted.
        "units_1":                  best.params["units_1"],
        "units_2":                  best.params["units_2"],
        "units_2_effective":        best.user_attrs.get("units_2_effective",
                                                         best.params["units_2"]),
        "dropout":                  best.params["dropout"],
        "recurrent_dropout":        best.params["recurrent_dropout"],
        "l2_factor":                best.params["l2_factor"],
        "lr":                       best.params["lr"],
        # Key metrics for the best trial.
        "train_a_auc":              best.user_attrs.get("train_a_auc"),
        "val_a_auc":                best.user_attrs.get("val_a_auc"),
        "auc_overfit_gap":          best.user_attrs.get("auc_overfit_gap"),
        "val_b_coverage":           best.user_attrs.get("val_b_coverage"),
        "val_b_cond_win_rate":      best.user_attrs.get("val_b_cond_wr"),
        "actual_epochs":            best.user_attrs.get("actual_epochs"),
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    log.info("Best params saved → %s", BEST_PARAMS_PATH)

    # ── Print a human-readable summary to console ─────────────────────────────
    print("\n" + "═" * 65)
    print("  OPTUNA STUDY COMPLETE — BEST TRIAL SUMMARY")
    print("═" * 65)
    print(f"  Trial number      :  {best.number}")
    print(f"  Val AUC (obj)     :  {best.value:.4f}")
    print(f"  Train AUC         :  {best.user_attrs.get('train_a_auc', 'N/A')}")
    print(f"  AUC Overfit Gap   :  {best.user_attrs.get('auc_overfit_gap', 'N/A')}")
    print(f"  Val B Coverage    :  {best.user_attrs.get('val_b_coverage', 'N/A')}")
    print(f"  Val B Cond WR     :  {best.user_attrs.get('val_b_cond_wr', 'N/A')}")
    print(f"  Actual epochs     :  {best.user_attrs.get('actual_epochs', 'N/A')}")
    print("─" * 65)
    print("  Best hyperparameters:")
    print(f"    units_1             :  {best.params['units_1']}")
    print(f"    units_2 (effective) :  {best.user_attrs.get('units_2_effective', best.params['units_2'])}")
    print(f"    dropout             :  {best.params['dropout']}")
    print(f"    recurrent_dropout   :  {best.params['recurrent_dropout']}")
    print(f"    l2_factor           :  {best.params['l2_factor']}")
    print(f"    lr                  :  {best.params['lr']}")
    print("─" * 65)
    print(f"  Full results CSV  →  {RESULTS_CSV_PATH}")
    print(f"  Best params JSON  →  {BEST_PARAMS_PATH}")
    print(f"  Optuna DB         →  {STUDY_DB_PATH}")
    print("═" * 65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Orchestrate the full overnight hyperparameter tuning run.

    Execution flow
    ──────────────
    1. Load and pre-process data ONCE (no repeated I/O in the objective).
    2. Create (or load from DB) the Optuna study.
    3. Register a progress callback.
    4. Run study.optimize() with KeyboardInterrupt catching at two levels.
    5. Export results to CSV and JSON unconditionally.

    Resume behaviour
    ────────────────
    If optuna_study.db already contains trials from a previous run, Optuna
    automatically resumes the study (same STUDY_NAME). Completed trials are
    not re-run. This means:
        • A power cut at trial 47 of 200 → re-run → picks up at trial 48.
        • A manual n_trials increase → re-run → adds trials on top of existing.
    """
    log.info("=" * 65)
    log.info("  03_hyperparameter_tuning.py  —  START")
    log.info("  Train: %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val:   %s → %s", VAL_START,   VAL_END)
    log.info("  TEST SET: LOCKED (not loaded at any point)")
    log.info("  N_TRIALS=%d  |  EPOCHS_PER_TRIAL=%d  |  SEQ_LEN=%d",
             N_TRIALS, EPOCHS_PER_TRIAL, SEQ_LEN)
    log.info("  DB: %s", STUDY_DB_PATH)
    log.info("  → Press Ctrl+C at any time to stop and export results.")
    log.info("=" * 65)

    # ── STEP 1: Load data once ────────────────────────────────────────────────
    log.info("[STEP 1] Loading and preparing data ...")
    (
        X_train, X_val, y_train, y_val,
        train_ds_fit, train_ds_eval, val_ds,
        y_train_aligned, y_val_aligned,
        feature_cols, scaler,
    ) = load_and_prepare_data(INPUT_CSV)

    # ── STEP 2: Create or resume Optuna study ─────────────────────────────────
    log.info("[STEP 2] Creating / resuming Optuna study '%s' ...", STUDY_NAME)

    # TPESampler: Tree-structured Parzen Estimator — Optuna's default Bayesian
    # sampler. Builds independent KDEs for good and bad trials to propose
    # promising regions of the search space. Explicitly specified here for
    # reproducibility (SEED) and to make the choice transparent for the thesis.
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        study_name    = STUDY_NAME,
        direction     = "maximize",     # maximise Val AUC
        sampler       = sampler,
        storage       = STUDY_DB_PATH,  # SQLite for crash-safe persistence
        load_if_exists= True,           # resume instead of raising an error
    )

    # How many trials have already been run (from a previous session)?
    completed_so_far = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    remaining        = max(0, N_TRIALS - completed_so_far)

    log.info(
        "Study loaded. Completed so far: %d / %d. Remaining: %d.",
        completed_so_far, N_TRIALS, remaining,
    )

    if remaining == 0:
        log.info("N_TRIALS already reached. Proceeding to export.")
        export_results(study)
        return

    # ── STEP 3: Build objective and progress callback ──────────────────────────
    log.info("[STEP 3] Building objective function ...")
    objective = make_objective(
        X_train         = X_train,
        X_val           = X_val,
        train_ds_fit    = train_ds_fit,
        train_ds_eval   = train_ds_eval,
        val_ds          = val_ds,
        y_train_aligned = y_train_aligned,
        y_val_aligned   = y_val_aligned,
    )

    study_start = time.perf_counter()
    progress_cb = make_progress_callback(study_start)

    # ── STEP 4: Run optimisation ───────────────────────────────────────────────
    log.info("[STEP 4] Running study.optimize() — %d trials remaining ...", remaining)

    try:
        study.optimize(
            objective,
            n_trials   = remaining,
            # catch=(KeyboardInterrupt,) marks an interrupted trial as FAILED
            # and allows the study loop to exit cleanly, proceeding to the
            # export step below.
            # NOTE: This catches KeyboardInterrupt raised INSIDE the objective
            # function (i.e., mid-trial). A Ctrl+C received BETWEEN trials
            # is handled by the outer except block below.
            catch      = (KeyboardInterrupt,),
            callbacks  = [progress_cb],
            # n_jobs=1: mandatory for TF/Keras to avoid GPU context conflicts
            # and non-deterministic weight sharing across parallel processes.
            n_jobs     = 1,
            show_progress_bar = False,   # use our custom callback instead
        )
        log.info("study.optimize() completed normally.")

    except KeyboardInterrupt:
        # Belt-and-suspenders catch for Ctrl+C received between trials
        # (in Optuna's own scheduling loop, outside the objective).
        log.warning(
            "KeyboardInterrupt received between trials. "
            "Stopping study gracefully and proceeding to export."
        )

    total_elapsed = time.perf_counter() - study_start
    log.info(
        "Optimisation wall-clock time: %.1f s  (%.1f min).",
        total_elapsed, total_elapsed / 60.0,
    )

    # ── STEP 5: Export results (always runs, even after interrupt) ─────────────
    log.info("[STEP 5] Exporting results ...")
    export_results(study)

    log.info("03_hyperparameter_tuning.py  —  COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
