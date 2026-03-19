#!/usr/bin/env python3
# =============================================================================
# 03_hyperparameter_tuning_v2.py
# =============================================================================
# PURPOSE:
#   Version 2 overnight Optuna hyperparameter search for the BTC LSTM v2
#   pipeline. Three structural changes from 03_hyperparameter_tuning.py:
#
#   [V2-1] ALIGNED DATE SPLITS
#          Train: 2015-01-01 → 2023-12-31  (9 years — matches 02_lstm_model_v2)
#          Val:   2024-01-01 → 2024-12-31  (1 full year)
#          Requires processed_dataset_v2.csv (from 01_data_generator_v2.py).
#
#   [V2-2] DETERMINISTIC BOTTLENECK (units_2 = units_1 // 2)
#          In v1, units_2 was an independent Optuna suggest parameter,
#          requiring a post-sample correction when units_2 > units_1.
#          In v2, units_2 is NOT sampled — it is always units_1 // 2.
#          Rationale:
#            (a) Eliminates the correction hack and the wasted trial budget
#                from architecturally invalid suggestions.
#            (b) Enforces a strict 2:1 compression ratio (a principled
#                bottleneck for supervised sequence classification).
#            (c) Reduces the search space from 6 dimensions to 5, giving
#                TPE more samples per meaningful parameter dimension.
#          Effective architectures after this change:
#            units_1=128 → units_2=64
#            units_1=256 → units_2=128
#            units_1=512 → units_2=256
#
#   [V2-3] SEQ_LEN = 30 
#          Aligned with 02_lstm_model_v2.py (see that script for rationale).
#
# UNCHANGED FROM v1:
#   • Objective: MAXIMIZE Val ROC AUC.
#   • Data loaded ONCE in main(); injected via closure into objective.
#   • NO test set loaded at any point (strict holdout preserved).
#   • Ironclad reproducibility reset at the start of every trial:
#       clear_session() + tf.random.set_seed(42) + np.random.seed(42)
#   • SQLite crash-safe storage; re-running resumes the study.
#   • Two-level KeyboardInterrupt handling.
#   • All Variant A + Variant B metrics tracked via trial.set_user_attr().
#   • CSV sorted by Val AUC descending; top_5_params.json exported.
#
# OUTPUTS:
#   • optuna_study_v2.db      — SQLite database (all trials, crash-safe)
#   • tuning_results_v2.csv   — full trials dataframe (sorted by Val AUC ↓)
#   • best_params_v2.json     — best trial's hyperparameters
#   • top_5_params_v2.json    — top-5 trials' hyperparameters
#   • run_log_tuning_v2.txt   — full session log
#
# OVERNIGHT USAGE:
#   python 03_hyperparameter_tuning_v2.py
#   → Press Ctrl+C at any point to stop cleanly and trigger export.
#   → Re-run to resume from the last completed trial.
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import sys
import time
import os

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

# Suppress TensorFlow verbose startup messages so the trial log is readable.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =============================================================================
# ███  GLOBAL CONFIGURATION  ███
# =============================================================================

# ── [V2-1] Chronological date splits ─────────────────────────────────────────
# NO test set — strict holdout preserved for the single final run in
# 02_lstm_model_v2.py. Splits match that script exactly.
TRAIN_START: str = "2015-01-01"
TRAIN_END:   str = "2023-12-31"
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"

# ── Feature pruning ───────────────────────────────────────────────────────────
# Must be kept IDENTICAL to PRUNED_FEATURES in 02_lstm_model_v2.py.
# These features showed negative permutation importance (permuting them
# improves val AUC → toxic). Excised before scaler.fit().
PRUNED_FEATURES: frozenset = frozenset({"gold_log_return", "nvda_log_return"})

# ── Data paths ────────────────────────────────────────────────────────────────
INPUT_CSV:         str = "processed_dataset_v2.csv"
STUDY_DB_PATH:     str = "sqlite:///optuna_study_v2.db"
RESULTS_CSV_PATH:  str = "tuning_results_v2.csv"
BEST_PARAMS_PATH:  str = "best_params_v2.json"
TOP_5_PARAMS_PATH: str = "top_5_params_v2.json"
LOG_FILE_PATH:     str = "run_log_tuning_v2.txt"

# ── Study configuration ───────────────────────────────────────────────────────
# CRITICAL: A unique STUDY_NAME prevents Optuna from resuming an incompatible
# old study that used different splits, SEQ_LEN, or feature sets. Mixing
# configurations in one study produces uninterpretable AUC comparisons.
# The v2 DB is separate (optuna_study_v2.db) so the v1 study is unaffected.
STUDY_NAME: str = "btc_lstm_v3"   # v3 = v2 pipeline config
N_TRIALS:   int = 215             # total budget; Ctrl+C stops gracefully

# ── Fixed constants (not tuned) ───────────────────────────────────────────────
# [V2-3] SEQ_LEN = 30 aligned with 02_lstm_model_v2.py.
SEQ_LEN:          int = 30
BATCH_SIZE:       int = 64
EPOCHS_PER_TRIAL: int = 50    # hard ceiling; EarlyStopping(patience=5) fires earlier

# ── Variant B thresholds (same as production script) ─────────────────────────
UPPER_THRESHOLD: float = 0.55
LOWER_THRESHOLD: float = 0.45

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# STEP 1 — DATA PIPELINE
# Loaded ONCE in main() and passed to the objective via closure.
# =============================================================================

def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Load the v2 processed dataset, slice Train and Val by exact date
    boundaries, apply StandardScaler (fit on Train only), and build
    windowed tf.data.Datasets ready for model.fit().

    Strict holdout guarantee
    ─────────────────────────
    The test slice is never extracted — only TRAIN_START–TRAIN_END and
    VAL_START–VAL_END are accessed. The test set is reserved exclusively
    for the single final evaluation in 02_lstm_model_v2.py.

    StandardScaler protocol (no leakage)
    ─────────────────────────────────────
    scaler.fit(X_train)    → μ/σ from training observations only.
    scaler.transform(X_val) → applies training μ/σ; no future information.

    Returns
    -------
    X_train, X_val          : np.ndarray — scaled feature matrices
    y_train, y_val          : np.ndarray — binary targets (unaligned, length N)
    train_ds_fit            : tf.data.Dataset — shuffled, for model.fit()
    train_ds_eval           : tf.data.Dataset — ordered, for metric computation
    val_ds                  : tf.data.Dataset — ordered
    y_train_aligned         : np.ndarray — y_train[SEQ_LEN-1:]
    y_val_aligned           : np.ndarray — y_val[SEQ_LEN-1:]
    feature_cols            : list[str]
    scaler                  : fitted StandardScaler
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

    for label, split in [("Train", train_df), ("Val", val_df)]:
        log.info(
            "  %-5s → %5d rows  [%s → %s]",
            label, len(split),
            split.index.min().date(), split.index.max().date(),
        )

    feature_cols = [
        c for c in train_df.columns
        if c != "target" and c not in PRUNED_FEATURES
    ]
    if PRUNED_FEATURES:
        log.info("Pruned toxic features: %s", sorted(PRUNED_FEATURES))
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    y_train     = train_df["target"].values.astype(np.float32)
    y_val       = val_df["target"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    log.info(
        "Scaler fitted on Train. Feature means: %s",
        dict(zip(feature_cols, np.round(scaler.mean_, 5))),
    )

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


def _make_dataset(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows with correct label alignment.

    Window i = features[i : i+SEQ_LEN]
    Target i = y[i + SEQ_LEN - 1]  → predicts direction of day i + SEQ_LEN
    """
    N         = len(X)
    n_windows = N - SEQ_LEN + 1

    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[SEQ_LEN - 1:]
    targets_aligned[n_windows:] = 0.0

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
# =============================================================================

def build_trial_model(
    units_1:           int,
    units_2:           int,
    dropout:           float,
    recurrent_dropout: float,
    l2_factor:         float,
    lr:                float,
    n_features:        int,
) -> tf.keras.Model:
    """
    Build a two-layer stacked LSTM with the given hyperparameters.

    [V2-2] units_2 is always units_1 // 2 (passed in deterministically from
    the objective; NOT independently sampled). The function signature is
    unchanged from v1 to keep the model builder generic and reusable.

    Note on cuDNN
    ─────────────
    Any recurrent_dropout > 0 disables TF's cuDNN LSTM kernel, falling back
    to the slower unoptimised implementation. EarlyStopping patience=5 kills
    low-quality trials early to compensate.
    """
    model = Sequential([
        LSTM(
            units=units_1,
            return_sequences=True,
            kernel_regularizer=L2Reg(l2_factor),
            recurrent_dropout=recurrent_dropout,
            input_shape=(SEQ_LEN, n_features),
            name="lstm_1",
        ),
        Dropout(dropout, name="drop_1"),
        LSTM(
            units=units_2,
            return_sequences=False,
            kernel_regularizer=L2Reg(l2_factor),
            recurrent_dropout=recurrent_dropout,
            name="lstm_2",
        ),
        Dropout(dropout, name="drop_2"),
        Dense(1, activation="sigmoid", name="output"),
    ], name=f"lstm_v2_{units_1}_{units_2}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# STEP 3 — VARIANT METRIC HELPERS
# =============================================================================

def _variant_a_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Variant A (0.5 threshold): AUC, Gini, Accuracy, Sensitivity, Specificity."""
    from sklearn.metrics import confusion_matrix as cm_fn

    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    acc    = accuracy_score(y_true.astype(int), y_pred)
    tn, fp, fn, tp = cm_fn(y_true.astype(int), y_pred).ravel()

    return dict(
        auc         = float(auc),
        gini        = float(2.0 * auc - 1.0),
        accuracy    = float(acc),
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    )


def _variant_b_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    upper:  float = UPPER_THRESHOLD,
    lower:  float = LOWER_THRESHOLD,
) -> dict:
    """Variant B (3-state filter): Coverage + Conditional Win Rate on traded days."""
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

    return dict(coverage=coverage, conditional_win_rate=cwr, n_traded=n_traded)


# =============================================================================
# STEP 4 — OPTUNA OBJECTIVE FUNCTION
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

    Uses a closure to inject pre-computed data into the objective without
    global variables. Optuna calls objective(trial) for each trial.

    [V2-2] units_2 search space change
    ─────────────────────────────────────
    v1: units_2 = suggest_categorical([64, 128, 256]) with post-sample clipping.
    v2: units_2 = units_1 // 2 (deterministic, never sampled).

    Fixed architecture (not sampled):
        units_1 = 512,  units_2 = 256  (2:1 bottleneck, hardcoded)
        Full search budget is spent on regularisation, not capacity search.

    Search space (4 dimensions):
        dropout            : {0.10, 0.15, …, 0.45}  (step=0.05)
        recurrent_dropout  : {0.10, 0.15, …, 0.40}  (step=0.05)
        l2_factor          : {1e-4, 1e-3}
        lr                 : {1e-3, 2e-3}
    """
    n_features = X_train.shape[1]

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: suggest → build → train → evaluate → log → return.

        Returns Val AUC (the maximisation objective).
        Returns -1.0 on any exception to mark the trial as FAILED.
        """
        # ── Ironclad reproducibility reset ────────────────────────────────────
        # Called EVERY trial, before any suggest_* or model building.
        # Rationale (identical to v1):
        #   1. clear_session()     — destroys all Keras objects from the previous
        #                            trial (layers, optimisers, weight variables).
        #                            Resets Keras's uid counters so layer names
        #                            are deterministic across trials.
        #   2. tf.random.set_seed  — resets TF's global random seed for weight
        #                            initialisers, dropout masks, shuffling.
        #   3. np.random.seed      — resets NumPy's global random state.
        # These three calls are necessary and sufficient for trial-level
        # reproducibility. Omitting any one allows state to leak between trials.
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        # ── Fixed architecture (not sampled) ───────────────────────────────────
        # Capacity locked to 512→256 so the full trial budget is spent on
        # regularisation tuning rather than architecture search.
        # Both values stored as user_attrs for consistent export.
        units_1 = 512
        units_2 = 256
        trial.set_user_attr("units_1", units_1)
        trial.set_user_attr("units_2", units_2)

        # ── Suggest hyperparameters (4 dimensions) ─────────────────────────────
        dropout           = trial.suggest_float(      "dropout",           0.10, 0.45, step=0.05)
        recurrent_dropout = trial.suggest_float(      "recurrent_dropout", 0.10, 0.40, step=0.05)
        l2_factor         = trial.suggest_categorical("l2_factor",         [1e-4, 1e-3])
        lr                = trial.suggest_categorical("lr",                 [1e-3, 2e-3])

        log.debug(
            "  Trial %d — arch: [%d→%d]  drop=%.2f  recdrop=%.2f  l2=%g  lr=%g",
            trial.number, units_1, units_2, dropout, recurrent_dropout, l2_factor, lr,
        )

        # ── Build, train, evaluate ─────────────────────────────────────────────
        try:
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
                    patience=5,               # tight: fail bad trials fast
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,               # fires before EarlyStopping
                    min_lr=1e-7,
                    verbose=0,
                ),
            ]

            history = model.fit(
                train_ds_fit,
                epochs=EPOCHS_PER_TRIAL,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=0,
            )

            actual_epochs = len(history.history["val_loss"])
            best_val_loss = float(min(history.history["val_loss"]))

            prob_train = model.predict(train_ds_eval, verbose=0).flatten()
            prob_val   = model.predict(val_ds,        verbose=0).flatten()

            m_train_a = _variant_a_metrics(y_train_aligned, prob_train)
            m_val_a   = _variant_a_metrics(y_val_aligned,   prob_val)
            m_train_b = _variant_b_metrics(y_train_aligned, prob_train)
            m_val_b   = _variant_b_metrics(y_val_aligned,   prob_val)

            # ── Generalization gap metrics ─────────────────────────────────────
            # auc_diff and acc_diff quantify the train→val performance drop.
            # Large positive values signal memorisation of raw price patterns.
            auc_diff = m_train_a["auc"]      - m_val_a["auc"]
            acc_diff = m_train_a["accuracy"] - m_val_a["accuracy"]

            # ── Log all metrics as persistent trial user attributes ────────────
            trial.set_user_attr("actual_epochs",   actual_epochs)
            trial.set_user_attr("best_val_loss",   round(best_val_loss, 5))
            trial.set_user_attr("auc_diff",        round(auc_diff,      4))
            trial.set_user_attr("acc_diff",        round(acc_diff,      4))

            trial.set_user_attr("train_a_auc",         round(m_train_a["auc"],         4))
            trial.set_user_attr("train_a_gini",        round(m_train_a["gini"],        4))
            trial.set_user_attr("train_a_accuracy",    round(m_train_a["accuracy"],    4))
            trial.set_user_attr("train_a_sensitivity", round(m_train_a["sensitivity"], 4))
            trial.set_user_attr("train_a_specificity", round(m_train_a["specificity"], 4))

            trial.set_user_attr("val_a_auc",           round(m_val_a["auc"],           4))
            trial.set_user_attr("val_a_gini",          round(m_val_a["gini"],          4))
            trial.set_user_attr("val_a_accuracy",      round(m_val_a["accuracy"],      4))
            trial.set_user_attr("val_a_sensitivity",   round(m_val_a["sensitivity"],   4))
            trial.set_user_attr("val_a_specificity",   round(m_val_a["specificity"],   4))

            trial.set_user_attr("train_b_coverage",    round(m_train_b["coverage"],    4))
            trial.set_user_attr("train_b_cond_wr", (
                round(m_train_b["conditional_win_rate"], 4)
                if not np.isnan(m_train_b["conditional_win_rate"]) else -1.0
            ))
            trial.set_user_attr("train_b_n_traded",    int(m_train_b["n_traded"]))

            trial.set_user_attr("val_b_coverage",      round(m_val_b["coverage"],      4))
            trial.set_user_attr("val_b_cond_wr", (
                round(m_val_b["conditional_win_rate"], 4)
                if not np.isnan(m_val_b["conditional_win_rate"]) else -1.0
            ))
            trial.set_user_attr("val_b_n_traded",      int(m_val_b["n_traded"]))

            # Val AUC is the maximisation objective:
            # • Threshold-independent.
            # • Robust to the ~53/47 class imbalance.
            # • Directly measures how well the model ranks UP vs DOWN days.
            return float(m_val_a["auc"])

        except Exception as exc:
            log.warning(
                "Trial %d raised %s: %s — marking as FAILED.",
                trial.number, type(exc).__name__, str(exc),
            )
            return -1.0

    return objective


# =============================================================================
# STEP 5 — OPTUNA PROGRESS CALLBACK
# =============================================================================

def make_progress_callback(study_start_time: float):
    """
    Build an Optuna callback that logs a concise per-trial summary.

    Architecture is fixed at 512→256, so only regularisation params are shown.
    Gap metrics use the renamed auc_diff / acc_diff user_attrs.
    """
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            log.warning(
                "Trial %3d  FAILED/PRUNED  —  "
                "arch: [512→256]  drop=%.2f  recdrop=%.2f  l2=%g  lr=%g",
                trial.number,
                trial.params.get("dropout",           float("nan")),
                trial.params.get("recurrent_dropout", float("nan")),
                trial.params.get("l2_factor",         float("nan")),
                trial.params.get("lr",                float("nan")),
            )
            return

        elapsed   = time.perf_counter() - study_start_time
        ua        = trial.user_attrs
        val_auc   = trial.value
        train_auc = ua.get("train_a_auc",   float("nan"))
        auc_gap   = ua.get("auc_diff",      float("nan"))
        acc_gap   = ua.get("acc_diff",      float("nan"))
        val_b_cov = ua.get("val_b_coverage", float("nan"))
        val_b_cwr = ua.get("val_b_cond_wr",  float("nan"))
        epochs    = ua.get("actual_epochs",  "?")

        is_best = (trial.number == study.best_trial.number)
        star    = "  ★ NEW BEST" if is_best else ""

        log.info(
            "Trial %3d  "
            "ValAUC=%.4f  TrainAUC=%.4f  AUC_Δ=%.4f  ACC_Δ=%.4f  "
            "VarB_Cov=%.2f  VarB_CWR=%.4f  "
            "arch=[512→256]  drop=%.2f  recdrop=%.2f  l2=%g  lr=%g  "
            "epochs=%s  elapsed=%.0fs%s",
            trial.number,
            val_auc, train_auc, auc_gap, acc_gap,
            val_b_cov, val_b_cwr,
            trial.params.get("dropout", 0.0),
            trial.params.get("recurrent_dropout", 0.0),
            trial.params.get("l2_factor", 0.0),
            trial.params.get("lr", 0.0),
            epochs,
            elapsed,
            star,
        )

    return callback


# =============================================================================
# STEP 6 — RESULTS EXPORT
# =============================================================================

def export_results(study: optuna.Study) -> None:
    """
    Export the full trials dataframe to CSV and best / top-5 parameters to JSON.

    Called unconditionally from main() — whether the study ran to N_TRIALS
    completion, was interrupted by Ctrl+C, or stopped after a resumed run.

    [V2-2] _trial_to_dict reads units_2 from user_attrs (not params["units_2"])
    because units_2 is no longer a sampled parameter in this study.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log.info(
        "Study ended — %d total trials  (%d COMPLETE, %d FAIL/PRUNED).",
        len(study.trials), len(completed), len(study.trials) - len(completed),
    )

    if not completed:
        log.warning("No completed trials to export.")
        return

    # ── Full trials CSV (sorted by Val AUC descending) ────────────────────────
    df_trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    df_trials.sort_values("value", ascending=False, inplace=True)
    df_trials.to_csv(RESULTS_CSV_PATH, index=False)
    log.info(
        "Full results exported (sorted Val AUC ↓) → %s  (%d rows)",
        RESULTS_CSV_PATH, len(df_trials),
    )

    # ── XLSX export (requires openpyxl) ───────────────────────────────────────
    xlsx_path = RESULTS_CSV_PATH.replace(".csv", ".xlsx")
    try:
        df_trials.to_excel(xlsx_path, index=False)
        log.info(
            "Full results also exported → %s  (%d rows)",
            xlsx_path, len(df_trials),
        )
    except ImportError:
        log.warning(
            "XLSX export skipped — openpyxl is not installed.  "
            "Run:  pip install openpyxl"
        )

    # ── Helper: build serialisable dict for one trial ─────────────────────────
    def _trial_to_dict(rank: int, t: optuna.trial.FrozenTrial) -> dict:
        return {
            "rank":                rank,
            "trial_number":        t.number,
            "val_auc":             round(t.value, 4),
            # Architecture is fixed; both read from user_attrs
            "units_1":             t.user_attrs.get("units_1", 512),
            "units_2":             t.user_attrs.get("units_2", 256),
            "dropout":             t.params["dropout"],
            "recurrent_dropout":   t.params["recurrent_dropout"],
            "l2_factor":           t.params["l2_factor"],
            "lr":                  t.params["lr"],
            "train_a_auc":         t.user_attrs.get("train_a_auc"),
            "val_a_auc":           t.user_attrs.get("val_a_auc"),
            "auc_diff":            t.user_attrs.get("auc_diff"),
            "acc_diff":            t.user_attrs.get("acc_diff"),
            "val_b_coverage":      t.user_attrs.get("val_b_coverage"),
            "val_b_cond_win_rate": t.user_attrs.get("val_b_cond_wr"),
            "actual_epochs":       t.user_attrs.get("actual_epochs"),
        }

    # ── Best params JSON ──────────────────────────────────────────────────────
    best        = study.best_trial
    best_params = _trial_to_dict(1, best)

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved → %s", BEST_PARAMS_PATH)

    # ── Top-5 params JSON ─────────────────────────────────────────────────────
    # Why top-5? Selection bias over 300 trials means the single best trial
    # may have benefited from a lucky gradient path. Top-5:
    #   (a) Cluster analysis: do winners share a config region?
    #   (b) Fallback candidates if #1 doesn't reproduce in model_v2.py.
    #   (c) Ensemble candidates: averaging top-N outputs is a marginal AUC booster.
    top5_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    top5_list   = [_trial_to_dict(rank, t) for rank, t in enumerate(top5_trials, start=1)]

    with open(TOP_5_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(top5_list, f, indent=2)
    log.info(
        "Top-5 params saved → %s  (Val AUC %.4f → %.4f)",
        TOP_5_PARAMS_PATH, top5_list[0]["val_auc"], top5_list[-1]["val_auc"],
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  OPTUNA v2 STUDY COMPLETE — BEST TRIAL SUMMARY")
    print("═" * 65)
    print(f"  Trial number      :  {best.number}")
    print(f"  Val AUC (obj)     :  {best.value:.4f}")
    print(f"  Train AUC         :  {best.user_attrs.get('train_a_auc', 'N/A')}")
    print(f"  AUC Gap (Δ)       :  {best.user_attrs.get('auc_diff', 'N/A')}")
    print(f"  Acc Gap (Δ)       :  {best.user_attrs.get('acc_diff', 'N/A')}")
    print(f"  Val B Coverage    :  {best.user_attrs.get('val_b_coverage', 'N/A')}")
    print(f"  Val B Cond WR     :  {best.user_attrs.get('val_b_cond_wr', 'N/A')}")
    print(f"  Actual epochs     :  {best.user_attrs.get('actual_epochs', 'N/A')}")
    print("─" * 65)
    print("  Best hyperparameters:")
    print(f"    units_1           :  512  (fixed)")
    print(f"    units_2           :  256  (fixed, = units_1 // 2)")
    print(f"    dropout           :  {best.params['dropout']}")
    print(f"    recurrent_dropout :  {best.params['recurrent_dropout']}")
    print(f"    l2_factor         :  {best.params['l2_factor']}")
    print(f"    lr                :  {best.params['lr']}")
    print("─" * 65)
    xlsx_path = RESULTS_CSV_PATH.replace(".csv", ".xlsx")
    print(f"  Full results CSV  →  {RESULTS_CSV_PATH}  (sorted Val AUC ↓)")
    print(f"  Full results XLSX →  {xlsx_path}")
    print(f"  Best params JSON  →  {BEST_PARAMS_PATH}")
    print(f"  Top-5 params JSON →  {TOP_5_PARAMS_PATH}")
    print(f"  Optuna DB         →  {STUDY_DB_PATH}")
    print("═" * 65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Orchestrate the full v2 overnight hyperparameter tuning run.

    Execution flow
    ──────────────
    1. Load and pre-process data ONCE (no repeated I/O in the objective).
    2. Create (or resume from DB) the Optuna study.
    3. Register a progress callback.
    4. Run study.optimize() with two-level KeyboardInterrupt handling.
    5. Export results to CSV and JSON unconditionally.

    Resume behaviour
    ────────────────
    If optuna_study_v2.db already contains trials (STUDY_NAME="btc_lstm_v3"),
    Optuna resumes automatically. A power cut at trial 47 → re-run → picks up
    at trial 48. Increasing N_TRIALS → re-run → adds trials on top.
    """
    log.info("=" * 65)
    log.info("  03_hyperparameter_tuning_v2.py  —  START")
    log.info("  Train: %s → %s", TRAIN_START, TRAIN_END)
    log.info("  Val:   %s → %s", VAL_START,   VAL_END)
    log.info("  TEST SET: LOCKED (not loaded at any point)")
    log.info("  N_TRIALS=%d  |  EPOCHS_PER_TRIAL=%d  |  SEQ_LEN=%d",
             N_TRIALS, EPOCHS_PER_TRIAL, SEQ_LEN)
    log.info("  [V2-2] units_2 = units_1 // 2  (deterministic, not sampled)")
    log.info("  DB: %s", STUDY_DB_PATH)
    log.info("  → Press Ctrl+C at any time to stop and export results.")
    log.info("=" * 65)

    # ── STEP 1: Load data once ────────────────────────────────────────────────
    log.info("[STEP 1] Loading and preparing v2 data ...")
    (
        X_train, X_val, y_train, y_val,
        train_ds_fit, train_ds_eval, val_ds,
        y_train_aligned, y_val_aligned,
        feature_cols, scaler,
    ) = load_and_prepare_data(INPUT_CSV)

    # ── STEP 2: Create or resume Optuna study ─────────────────────────────────
    log.info("[STEP 2] Creating / resuming Optuna study '%s' ...", STUDY_NAME)

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        study_name    = STUDY_NAME,
        direction     = "maximize",
        sampler       = sampler,
        storage       = STUDY_DB_PATH,
        load_if_exists= True,
    )

    completed_so_far = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    remaining = max(0, N_TRIALS - completed_so_far)

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
            # catch=(KeyboardInterrupt,): marks an interrupted trial as FAILED
            # and allows the loop to exit cleanly, proceeding to the export step.
            # A Ctrl+C received BETWEEN trials is caught by the outer except block.
            catch      = (KeyboardInterrupt,),
            callbacks  = [progress_cb],
            n_jobs     = 1,              # mandatory: TF/Keras is not thread-safe
            show_progress_bar = False,   # use our custom callback instead
        )
        log.info("study.optimize() completed normally.")

    except KeyboardInterrupt:
        log.warning(
            "KeyboardInterrupt received between trials. "
            "Stopping gracefully and proceeding to export."
        )

    total_elapsed = time.perf_counter() - study_start
    log.info(
        "Optimisation wall-clock time: %.1f s  (%.1f min).",
        total_elapsed, total_elapsed / 60.0,
    )

    # ── STEP 5: Export results (always runs, even after interrupt) ─────────────
    log.info("[STEP 5] Exporting results ...")
    export_results(study)

    log.info("03_hyperparameter_tuning_v2.py  —  COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
