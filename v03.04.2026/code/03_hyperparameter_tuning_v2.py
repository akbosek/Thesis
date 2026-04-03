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
# OUTPUTS (single DB, per-study files):
#   • optuna_dual_study.db                      — single SQLite DB (both studies)
#   • tuning_results_lstm_unweighted.csv/.xlsx  — unweighted study results
#   • best_params_lstm_unweighted.json          — unweighted best trial
#   • top_5_params_lstm_unweighted.json         — unweighted top-5 trials
#   • tuning_results_lstm_weighted.csv/.xlsx    — weighted study results
#   • best_params_lstm_weighted.json            — weighted best trial
#   • top_5_params_lstm_weighted.json           — weighted top-5 trials
#   • run_log_tuning_v2.txt                     — full session log
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
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
INPUT_CSV:     str = "processed_dataset_v2.csv"
LOG_FILE_PATH: str = "run_log_tuning_v2.txt"

# ── Study configuration ───────────────────────────────────────────────────────
# Per-study DB paths, result CSVs, and JSON files are derived at runtime from
# study_name inside run_tuning_session / export_results. N_TRIALS_PER_RUN is
# the budget for EACH of the two studies (unweighted + weighted).
N_TRIALS_PER_RUN: int = 50

# ── Fixed constants (not tuned) ───────────────────────────────────────────────
# [V2-3] SEQ_LEN = 30 aligned with 02_lstm_model_v2.py.
SEQ_LEN:          int = 30
BATCH_SIZE:       int = 64
EPOCHS_PER_TRIAL: int = 50    # hard ceiling; EarlyStopping(patience=5) fires earlier

# ── Variant B thresholds (same as production script) ─────────────────────────
UPPER_THRESHOLD: float = 0.52
LOWER_THRESHOLD: float = 0.48

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
    X_train, X_val  : np.ndarray — scaled feature matrices
    y_train, y_val  : np.ndarray — binary targets (unaligned, length N)
    train_df, val_df: pd.DataFrame — raw (unscaled) split DataFrames; needed
                      by create_objective to compute sample weights conditionally.
    feature_cols    : list[str]
    scaler          : fitted StandardScaler

    NOTE: Sample weights are NOT computed here. They are computed inside
    create_objective conditional on use_weights, so both studies share the
    same scaled X arrays without re-loading or re-fitting the scaler.
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

    log.info(
        "Windows (preview) — Train: %d | Val: %d",
        len(X_train) - SEQ_LEN + 1, len(X_val) - SEQ_LEN + 1,
    )

    return (
        X_train, X_val, y_train, y_val,
        train_df, val_df,
        feature_cols, scaler,
    )


def _make_dataset(
    X:              np.ndarray,
    y:              np.ndarray,
    shuffle:        bool = False,
    sample_weights: np.ndarray | None = None,
    batch_size:     int = BATCH_SIZE,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows with correct label alignment.

    Window i = features[i : i+SEQ_LEN]
    Target i = y[i + SEQ_LEN - 1]  → predicts direction of day i + SEQ_LEN

    When sample_weights is provided, yields (x, y, w) triples consumed by
    model.fit() as per-sample loss weights. Pass None for eval-only datasets
    (train_ds_eval, used only for model.predict()).
    The batch_size parameter allows per-trial dynamic batching.
    """
    N         = len(X)
    n_windows = N - SEQ_LEN + 1
    y_aligned = y[SEQ_LEN - 1:].astype(np.float32)

    if sample_weights is not None:
        X_windows = np.stack(
            [X[i : i + SEQ_LEN] for i in range(n_windows)]
        ).astype(np.float32)
        w  = sample_weights[:n_windows].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned, w))
        if shuffle:
            ds = ds.shuffle(buffer_size=n_windows, seed=SEED)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, y_aligned

    # ── Default path (no sample weights) ─────────────────────────────────────
    targets_aligned = np.empty(N, dtype=np.float32)
    targets_aligned[:n_windows] = y[SEQ_LEN - 1:]
    targets_aligned[n_windows:] = 0.0

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X.astype(np.float32),
        targets=targets_aligned,
        sequence_length=SEQ_LEN,
        sequence_stride=1,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )
    return dataset, y_aligned


def _compute_sample_weights(df, seq_len: int) -> np.ndarray:
    """
    Compute per-window sample weights from abs(btc_log_return) of the target day.
    weight[i] corresponds to window i, which predicts day i+seq_len−1.
    Normalised so mean(w) = 1.0 (total gradient magnitude unchanged vs unweighted).
    Returns shape (N − seq_len + 1,), dtype float32.
    """
    returns = df["btc_log_return"].values[seq_len - 1:]
    w       = np.abs(returns)
    w_sum   = float(np.sum(w))
    if w_sum > 0:
        w = w / w_sum * len(w)      # normalise so mean(w) = 1
    return w.astype(np.float32)


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
# STEP 3 — OPTUNA OBJECTIVE FUNCTION
# =============================================================================

def create_objective(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    train_df:    pd.DataFrame,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    val_df:      pd.DataFrame,
    use_weights: bool,
    seq_len:     int,
    n_features:  int,
):
    """
    Factory that builds and returns the Optuna objective function.

    Uses a closure to inject pre-computed data into the objective without
    global variables. Optuna calls objective(trial) for each trial.

    Sample weights are computed here (once per study) rather than inside
    objective(trial), so the weight vectors are shared across all trials.

    Fixed architecture (not sampled):
        units_1 = 512,  units_2 = 256  (2:1 bottleneck, hardcoded)
        Full search budget is spent on regularisation, not capacity search.

    Search space (4 dimensions):
        dropout            : {0.25}
        recurrent_dropout  : {0.25}
        l2_factor          : {0.0002, 0.0005, 0.001}
        lr                 : {0.001, 0.002}
        batch_size         : 64 (fixed — not sampled)
    """
    if use_weights:
        train_weights = _compute_sample_weights(train_df, seq_len)
        val_weights   = _compute_sample_weights(val_df,   seq_len)
    else:
        train_weights = val_weights = None

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: suggest → build datasets → train → evaluate → log → return.

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

        # ── Suggest hyperparameters (5 dimensions) ─────────────────────────────
        dropout           = trial.suggest_categorical("dropout",           [0.25])
        recurrent_dropout = trial.suggest_categorical("recurrent_dropout", [0.25])
        l2_factor         = trial.suggest_categorical("l2_factor",         [0.0002, 0.0005, 0.001])
        lr                = trial.suggest_categorical("lr",                 [0.001, 0.002])
        batch_size        = 64  # fixed; not sampled

        log.debug(
            "  Trial %d — arch: [%d→%d]  drop=%.2f  recdrop=%.2f  l2=%g  lr=%g",
            trial.number, units_1, units_2, dropout, recurrent_dropout, l2_factor, lr,
        )

        # ── Build datasets for this trial's batch_size ─────────────────────────
        # train_ds_fit  : (x, y, w) triples — sample-weighted, shuffled
        # val_ds        : (x, y, w) triples — sample-weighted, unshuffled
        # train_ds_eval : built inside try block, after model.fit()
        train_ds_fit, _       = _make_dataset(
            X_train, y_train, shuffle=True,
            sample_weights=train_weights, batch_size=batch_size,
        )
        val_ds, y_val_aligned = _make_dataset(
            X_val, y_val, shuffle=False,
            sample_weights=val_weights, batch_size=batch_size,
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
                    patience=7,
                    restore_best_weights=True,
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

            actual_epochs = len(history.history["loss"])

            prob_val = model.predict(val_ds, verbose=0).ravel()
            val_auc  = roc_auc_score(y_val_aligned, prob_val)

            train_ds_eval, y_train_aligned = _make_dataset(
                X_train, y_train, shuffle=False, batch_size=batch_size,
            )
            prob_train = model.predict(train_ds_eval, verbose=0).ravel()
            train_auc  = roc_auc_score(y_train_aligned, prob_train)

            auc_diff = train_auc - val_auc

            trial.set_user_attr("auc_diff",      float(auc_diff))
            trial.set_user_attr("actual_epochs", actual_epochs)
            trial.set_user_attr("train_a_auc",   float(train_auc))
            trial.set_user_attr("val_a_auc",     float(val_auc))

            return float(val_auc)

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
            "arch=[512→256]  drop=%.2f  recdrop=%.2f  l2=%g  lr=%g  bs=%s  "
            "epochs=%s  elapsed=%.0fs%s",
            trial.number,
            val_auc, train_auc, auc_gap, acc_gap,
            val_b_cov, val_b_cwr,
            trial.params.get("dropout", 0.0),
            trial.params.get("recurrent_dropout", 0.0),
            trial.params.get("l2_factor", 0.0),
            trial.params.get("lr", 0.0),
            trial.params.get("batch_size", "?"),
            epochs,
            elapsed,
            star,
        )

    return callback


# =============================================================================
# STEP 6 — RESULTS EXPORT
# =============================================================================

def export_results(study: optuna.Study, db_path: str) -> None:
    """
    Export the full trials dataframe to CSV and best / top-5 parameters to JSON.

    Output file paths are derived from study.study_name so that the two
    studies (lstm_unweighted / lstm_weighted) write to separate files.

    Called unconditionally from run_tuning_session() — whether the study ran
    to completion, was interrupted by Ctrl+C, or stopped after a resumed run.
    """
    # ── Derive per-study output paths ─────────────────────────────────────────
    results_csv = f"tuning_results_{study.study_name}.csv"
    xlsx_path   = f"tuning_results_{study.study_name}.xlsx"
    best_json   = f"best_params_{study.study_name}.json"
    top5_json   = f"top_5_params_{study.study_name}.json"

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log.info(
        "Study '%s' ended — %d total trials  (%d COMPLETE, %d FAIL/PRUNED).",
        study.study_name,
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
    df_trials.to_csv(results_csv, index=False)
    log.info(
        "Full results exported (sorted Val AUC ↓) → %s  (%d rows)",
        results_csv, len(df_trials),
    )

    # ── XLSX export (requires openpyxl) ───────────────────────────────────────
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
            "batch_size":          t.params.get("batch_size", 64),  # fixed; fallback for legacy trials
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

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved → %s", best_json)

    # ── Top-5 params JSON ─────────────────────────────────────────────────────
    # Why top-5? Selection bias over many trials means the single best trial
    # may have benefited from a lucky gradient path. Top-5:
    #   (a) Cluster analysis: do winners share a config region?
    #   (b) Fallback candidates if #1 doesn't reproduce in model_v2.py.
    #   (c) Ensemble candidates: averaging top-N outputs is a marginal AUC booster.
    top5_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    top5_list   = [_trial_to_dict(rank, t) for rank, t in enumerate(top5_trials, start=1)]

    with open(top5_json, "w", encoding="utf-8") as f:
        json.dump(top5_list, f, indent=2)
    log.info(
        "Top-5 params saved → %s  (Val AUC %.4f → %.4f)",
        top5_json, top5_list[0]["val_auc"], top5_list[-1]["val_auc"],
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  OPTUNA STUDY COMPLETE — {study.study_name.upper()}")
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
    print(f"    batch_size        :  64  (fixed)")
    print("─" * 65)
    print(f"  Full results CSV  →  {results_csv}  (sorted Val AUC ↓)")
    print(f"  Full results XLSX →  {xlsx_path}")
    print(f"  Best params JSON  →  {best_json}")
    print(f"  Top-5 params JSON →  {top5_json}")
    print(f"  Optuna DB         →  {db_path}")
    print("═" * 65 + "\n")


# =============================================================================
# STEP 7 — STUDY RUNNER
# =============================================================================

def run_tuning_session(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    train_df:    pd.DataFrame,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    val_df:      pd.DataFrame,
    use_weights: bool,
    study_name:  str,
    db_path:     str,
    n_trials:    int,
    seq_len:     int,
    n_features:  int,
) -> None:
    """
    Run one complete Optuna tuning session (create/resume study, optimise,
    export results). Designed to be called twice from main() — once for the
    unweighted study and once for the weighted study.

    The try/except/finally guarantees that export_results() is called even
    on KeyboardInterrupt or unexpected exceptions, so partial results are
    never silently lost.
    """
    log.info("=" * 65)
    log.info("  OPTUNA SESSION: %s  |  use_weights=%s", study_name.upper(), use_weights)
    log.info("  n_trials=%d  |  DB: %s", n_trials, db_path)
    log.info("=" * 65)

    objective_fn = create_objective(
        X_train, y_train, train_df,
        X_val,   y_val,   val_df,
        use_weights, seq_len, n_features,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(
        study_name     = study_name,
        direction      = "maximize",
        sampler        = sampler,
        storage        = db_path,
        load_if_exists = True,
    )

    completed_so_far = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = max(0, n_trials - completed_so_far)
    log.info(
        "Completed so far: %d / %d.  Remaining: %d.",
        completed_so_far, n_trials, remaining,
    )

    if remaining == 0:
        log.info("n_trials already reached — skipping optimisation, exporting.")
        export_results(study, db_path)
        return

    study_start = time.perf_counter()
    progress_cb = make_progress_callback(study_start)

    try:
        study.optimize(
            objective_fn,
            n_trials          = remaining,
            callbacks         = [progress_cb],
            n_jobs            = 1,              # TF/Keras not thread-safe
            show_progress_bar = True,
        )
        log.info("study.optimize() completed normally.")

    except KeyboardInterrupt:
        print(
            "\n[WARNING] Optimisation interrupted by user. "
            "Exporting results gathered so far..."
        )
        log.warning("KeyboardInterrupt caught — exporting partial results.")

    finally:
        elapsed = time.perf_counter() - study_start
        log.info(
            "Wall-clock time: %.1f s  (%.1f min).", elapsed, elapsed / 60.0,
        )
        export_results(study, db_path)

    log.info("=== COMPLETED SESSION: %s ===", study_name.upper())


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Dual-Optuna pipeline: load data once, then run two sequential tuning
    studies (unweighted and weighted) using the same scaled arrays.

    Each study writes to its own SQLite DB and output files derived from
    study_name. Both studies are resumable independently via load_if_exists.
    Press Ctrl+C at any time — export_results() fires unconditionally.
    """
    log.info("=" * 65)
    log.info("  03_hyperparameter_tuning_v2.py  —  DUAL-OPTUNA START")
    log.info("  Train: %s → %s  |  Val: %s → %s",
             TRAIN_START, TRAIN_END, VAL_START, VAL_END)
    log.info("  TEST SET: LOCKED (not loaded at any point)")
    log.info("  N_TRIALS_PER_RUN=%d  |  EPOCHS_PER_TRIAL=%d  |  SEQ_LEN=%d",
             N_TRIALS_PER_RUN, EPOCHS_PER_TRIAL, SEQ_LEN)
    log.info("  → Press Ctrl+C at any time to stop and export results.")
    log.info("=" * 65)

    # ── STEP 1: Load data ONCE — shared by both studies ───────────────────────
    log.info("[STEP 1] Loading and preparing v2 data ...")
    (
        X_train, X_val, y_train, y_val,
        train_df, val_df,
        feature_cols, scaler,
    ) = load_and_prepare_data(INPUT_CSV)
    n_features = len(feature_cols)
    log.info("Features after pruning: %d", n_features)

    # ── STEP 2: Unweighted study ──────────────────────────────────────────────
    run_tuning_session(
        X_train, y_train, train_df,
        X_val,   y_val,   val_df,
        use_weights = False,
        study_name  = "lstm_unweighted",
        db_path     = "sqlite:///optuna_dual_study.db",
        n_trials    = N_TRIALS_PER_RUN,
        seq_len     = SEQ_LEN,
        n_features  = n_features,
    )

    # ── STEP 3: Weighted study ────────────────────────────────────────────────
    run_tuning_session(
        X_train, y_train, train_df,
        X_val,   y_val,   val_df,
        use_weights = True,
        study_name  = "lstm_weighted",
        db_path     = "sqlite:///optuna_dual_study.db",
        n_trials    = N_TRIALS_PER_RUN,
        seq_len     = SEQ_LEN,
        n_features  = n_features,
    )

    log.info("03_hyperparameter_tuning_v2.py  —  DUAL-OPTUNA COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
