#!/usr/bin/env python3
# =============================================================================
# 03_optuna_optimizer.py  (v2 — Dual-Study Refactor)
# =============================================================================
#
# TITLE  : Dual-Study Bayesian Hyperparameter Optimisation for BTC Direction LSTM
# AUTHOR : Bachelor's Thesis — Quantitative Finance / Machine Learning
#
# OVERVIEW
# --------
# Runs TWO independent Optuna TPE studies sequentially:
#   Study 1 — Unweighted : standard BCE loss, uniform sample weight
#   Study 2 — Weighted   : target-volatility sample weights reshape the loss landscape
#
# Each trial generates a 3-panel learning-curve diagnostic saved to:
#   results/optuna/unweighted/trial_NNN_learning_curve.png
#   results/optuna/weighted/trial_NNN_learning_curve.png
#
# EXPANDED SEARCH SPACE  (continuous distributions; TPE navigates freely)
# -----------------------------------------------------------------------
#   units_1           : Categorical {64, 128, 256, 512}
#   units_2           : Categorical {32, 64, 128, 256}
#   dropout_rate      : Uniform    [0.10, 0.70]
#   recurrent_dropout : Uniform    [0.10, 0.70]
#   l2_reg            : LogUniform [1e-6, 1e-2]
#   learning_rate     : LogUniform [1e-5, 1e-2]
#   optimizer_name    : Categorical {adam, adamw}
#
# OBJECTIVE  (unchanged from v1)
# --------------------------------
#   Score = Val_AUC − λ · max(0, Train_AUC − Val_AUC)   with λ = 0.5
#
# OUTPUTS
# -------
#   results/optuna/unweighted/trial_NNN_learning_curve.png
#   results/optuna/weighted/trial_NNN_learning_curve.png
#   best_params_unweighted.json
#   best_params_weighted.json
#   optuna_unweighted.db  /  optuna_weighted.db   — SQLite persistence
#   optuna_unweighted.xlsx / optuna_weighted.xlsx  — full trial export
#   optuna_run_log.txt
#
# USAGE
# -----
#   python 03_optuna_optimizer.py               # 80 trials per study (160 total)
#   python 03_optuna_optimizer.py --trials 50   # 50 trials per study
# =============================================================================

import argparse
import json
import logging
import os 
import pathlib
import sys
from typing import Optional

# ── CPU determinism flags (must precede tensorflow import) ────────────────────
os.environ["PYTHONHASHSEED"]       = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — MUST precede pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.regularizers import l2 as L2Reg # type: ignore

# ── Force single-threaded TF execution for full CPU determinism ───────────────
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism() 

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    from optuna.integration import TFKerasPruningCallback
except ImportError:
    from optuna_integration import TFKerasPruningCallback

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    _AdamW = tf.keras.optimizers.AdamW
except AttributeError:
    try:
        _AdamW = tf.keras.optimizers.experimental.AdamW
    except AttributeError:
        _AdamW = tf.keras.optimizers.Adam  # TF < 2.11 fallback


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_START: str = "2015-01-01"
TRAIN_END:   str = "2023-12-31"
VAL_START:   str = "2024-01-01"
VAL_END:     str = "2024-12-31"

SEQ_LEN:     int   = 30
BATCH_SIZE:  int   = 64
EPOCHS:      int   = 100
GAP_PENALTY: float = 0.5

# Volatility-weight parameters (Study 2 only)
VOL_WINDOW:  int   = 20   # rolling std window in trading days
VOL_CLIP_LO: float = 0.10  # floor — prevents near-zero weighting in calm regimes
VOL_CLIP_HI: float = 5.0   # ceiling — caps crash/squeeze outliers

PRUNED_FEATURES: frozenset = frozenset({"vix_level"})

INPUT_CSV: str = "processed_dataset_v3.csv"

# Output layout
_RESULTS_ROOT:       pathlib.Path = pathlib.Path("results new optuna") / "optuna"
UNWEIGHTED_PLOT_DIR: pathlib.Path = _RESULTS_ROOT / "unweighted"
WEIGHTED_PLOT_DIR:   pathlib.Path = _RESULTS_ROOT / "weighted"

UNWEIGHTED_DB:   str = "sqlite:///optuna_unweighted.db"
WEIGHTED_DB:     str = "sqlite:///optuna_weighted.db"
UNWEIGHTED_XLSX: str = "optuna_unweighted.xlsx"
WEIGHTED_XLSX:   str = "optuna_weighted.xlsx"
LOG_FILE:        str = "optuna_run_log.txt"

SEED: int = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA PIPELINE
# =============================================================================

def load_and_split(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed dataset and return (train_df, val_df).

    Each split includes SEQ_LEN rows before its calendar boundary so that the
    first sliding-window prediction lands exactly on TRAIN_START / VAL_START
    (warm-up buffer fix — prevents silently skipping the first month).
    """
    log.info("Loading dataset from '%s' ...", csv_path)
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )

    def _slice_with_buffer(start_date: str, end_date: str) -> pd.DataFrame:
        mask       = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        start_idx  = df[mask].index.min()
        end_idx    = df[mask].index.max()
        safe_start = max(0, start_idx - SEQ_LEN + 1)
        sliced     = df.iloc[safe_start : end_idx + 1].copy().reset_index(drop=True)
        return sliced.set_index("Date")

    train_df = _slice_with_buffer(TRAIN_START, TRAIN_END)
    val_df   = _slice_with_buffer(VAL_START,   VAL_END)

    log.info(
        "  Train → %5d rows  [%s → %s]  (incl. %d-day warm-up buffer)",
        len(train_df), train_df.index.min().date(),
        train_df.index.max().date(), SEQ_LEN,
    )
    log.info(
        "  Val   → %5d rows  [%s → %s]  (incl. %d-day warm-up buffer)",
        len(val_df), val_df.index.min().date(),
        val_df.index.max().date(), SEQ_LEN,
    )
    return train_df, val_df


def scale_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> tuple:
    """
    Fit StandardScaler on train only, transform both splits.

    Returns (X_train, X_val, y_train, y_val, feature_cols).
    Scaler is fitted exclusively on training data to prevent look-ahead bias.
    """
    feature_cols = [
        c for c in train_df.columns
        if c != "target" and c not in PRUNED_FEATURES
    ]
    log.info("Pruned features : %s", sorted(PRUNED_FEATURES))
    log.info("Model features  (%d): %s", len(feature_cols), feature_cols)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_val_raw   = val_df[feature_cols].values.astype(np.float32)
    y_train     = train_df["target"].values.astype(np.float32)
    y_val       = val_df["target"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    return X_train, X_val, y_train, y_val, feature_cols


def compute_vol_weights(
    train_df: pd.DataFrame,
    window:   int   = VOL_WINDOW,
    clip_lo:  float = VOL_CLIP_LO,
    clip_hi:  float = VOL_CLIP_HI,
) -> np.ndarray:
    """
    Compute target-volatility sample weights from the training DataFrame.

    Searches for a BTC log-return column and computes a rolling standard
    deviation series.  Weights are normalised to mean=1.0 so that total
    gradient magnitude stays comparable across studies.

    Falls back to uniform weights (all 1.0) if no return column is detected.
    """
    return_col: Optional[str] = None
    for col in train_df.columns:
        lo = col.lower()
        if ("btc" in lo or "bitcoin" in lo) and ("return" in lo or "ret" in lo):
            return_col = col
            break
    if return_col is None:
        for col in train_df.columns:
            lo = col.lower()
            if "log_return" in lo or "logreturn" in lo:
                return_col = col
                break

    if return_col is None:
        log.warning(
            "No BTC log-return column found — using uniform sample weights.  "
            "Available columns: %s", list(train_df.columns)
        )
        return np.ones(len(train_df), dtype=np.float32)

    returns = train_df[return_col].astype(float)
    vol     = returns.rolling(window=window, min_periods=1).std().fillna(0.0).values
    vol     = np.where(vol < 1e-10, 1e-10, vol)

    weights = (vol / vol.mean()).astype(np.float32)
    weights = np.clip(weights, clip_lo, clip_hi)

    log.info(
        "Vol weights from '%s' (window=%d)  |  "
        "mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        return_col, window,
        weights.mean(), weights.std(), weights.min(), weights.max(),
    )
    return weights


def make_tf_dataset(
    X:              np.ndarray,
    y:              np.ndarray,
    seq_len:        int,
    batch_size:     int,
    sample_weights: Optional[np.ndarray] = None,
    shuffle:        bool = False,
) -> tuple[tf.data.Dataset, np.ndarray]:
    """
    Build a sliding-window tf.data.Dataset, optionally with sample weights.

    Windows are pre-computed via np.stack to guarantee that zipping a weight
    tensor produces exact sample-wise alignment (avoids the index-offset
    ambiguity of timeseries_dataset_from_array when a separate weight
    array is added post-hoc).

    When sample_weights is provided the dataset yields 3-tuples
    (X_batch, y_batch, w_batch), which Keras interprets as weighted BCE.
    When None it yields standard (X_batch, y_batch) 2-tuples.

    Returns (dataset, y_aligned) where y_aligned = y[seq_len − 1 :].

    Window ↔ Target alignment (no look-ahead bias)
    -----------------------------------------------
    Window i  = X[i : i + seq_len]
    Target i  = y[i + seq_len − 1]   (direction on the last day of the window)
    Weight i  = w[i + seq_len − 1]   (volatility on the last day of the window)
    """
    n_windows = len(X) - seq_len + 1
    y_aligned = y[seq_len - 1:].astype(np.float32)

    X_windows = np.stack(
        [X[i : i + seq_len] for i in range(n_windows)], axis=0
    ).astype(np.float32)

    if sample_weights is not None:
        w_aligned = sample_weights[seq_len - 1:].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned, w_aligned))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X_windows, y_aligned))

    if shuffle:
        ds = ds.shuffle(buffer_size=n_windows, seed=SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, y_aligned


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def _resolve_optimizer(
    name: str, lr: float, weight_decay: float = 1e-4,
) -> tf.keras.optimizers.Optimizer:
    if name == "adamw":
        try:
            return _AdamW(learning_rate=lr, weight_decay=weight_decay)
        except TypeError:
            return _AdamW(weight_decay, learning_rate=lr)
    return {"adam": tf.keras.optimizers.Adam}[name](learning_rate=lr)


def build_trial_model(
    trial:      optuna.Trial,
    seq_len:    int,
    n_features: int,
) -> tf.keras.Model:
    """
    Sample hyperparameters from continuous / log-uniform distributions.

    v2 changes vs v1 categorical grid
    -----------------------------------
    · units_1 / units_2 : independent categorical sets; unlocks narrow-deep
      architectures (e.g. 64×32) that the old "256_128" / "512_256" strings
      could not express.
    · dropout_rate / recurrent_dropout : separate Uniform[0.1, 0.7] params so
      TPE can express asymmetric regularisation (heavy recurrent drop, light
      input drop, or vice-versa).
    · l2_reg   : LogUniform[1e-6, 1e-2]  — 4-decade range, sub-grid optima visible.
    · learning_rate : LogUniform[1e-5, 1e-2] — extends down to 1e-5 to handle
      the steeper loss landscape of the weighted study.
    """
    units1   = trial.suggest_categorical("units_1",           [64, 128, 256])
    units2   = trial.suggest_categorical("units_2",           [32, 64, 128])
    dropout  = trial.suggest_float(      "dropout_rate",       0.25, 0.70)
    rec_drop = trial.suggest_float(      "recurrent_dropout",  0.25, 0.70)
    l2_reg   = trial.suggest_float(      "l2_reg",             1e-6, 1e-2, log=True)
    lr       = trial.suggest_float(      "learning_rate",      1e-6, 1e-2, log=True)
    opt_name = trial.suggest_categorical("optimizer_name",    ["adam", "adamw"])

    model = Sequential(
        [
            LSTM(
                units=units1,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=rec_drop,
                input_shape=(seq_len, n_features),
                name="lstm_1",
            ),
            Dropout(dropout, name="drop_1"),
            LSTM(
                units=units2,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=rec_drop,
                name="lstm_2",
            ),
            Dropout(dropout, name="drop_2"),
            Dense(1, activation="sigmoid", name="output"),
        ],
        name=f"trial_{trial.number}_{units1}x{units2}",
    )
    model.compile(
        optimizer=_resolve_optimizer(opt_name, lr, weight_decay=l2_reg),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# =============================================================================
# LEARNING-CURVE DIAGNOSTICS
# =============================================================================

def plot_learning_curves(
    history:     tf.keras.callbacks.History,
    trial:       optuna.Trial,
    val_auc:     float,
    train_auc:   float,
    plot_dir:    pathlib.Path,
    study_label: str,
) -> None:
    """
    Render a 3-panel learning-curve figure and write to disk.

    Panels
    ------
    1. Binary Cross-Entropy Loss  — train vs val per epoch
    2. Classification Accuracy    — train vs val per epoch
    3. ROC AUC (Keras streaming)  — train vs val per epoch, with final
       sklearn values annotated as horizontal reference lines

    File: results/optuna/<label>/trial_NNN_learning_curve.png
    """
    h   = history.history
    eps = range(1, len(h["loss"]) + 1)

    # Keras may append integer suffixes to metric names in some TF versions
    _key = lambda stem: next(
        (k for k in h if k == stem or k.startswith(stem + "_")), stem
    )
    auc_key     = _key("auc")
    val_auc_key = _key("val_auc")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"[{study_label.upper()}]  Trial {trial.number:03d}"
        f"  |  Val AUC = {val_auc:.4f}"
        f"  |  Train AUC = {train_auc:.4f}"
        f"  |  Gap = {train_auc - val_auc:+.4f}",
        fontsize=11, fontweight="bold",
    )

    TRAIN_CLR = "#1976D2"
    VAL_CLR   = "#D32F2F"

    # ── Panel 1: BCE Loss ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, h["loss"],     color=TRAIN_CLR, label="Train")
    ax.plot(eps, h["val_loss"], color=VAL_CLR,   label="Val", linestyle="--")
    ax.set_title("Binary Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 2: Classification Accuracy ─────────────────────────────────────
    ax = axes[1]
    ax.plot(eps, h["accuracy"],     color=TRAIN_CLR, label="Train")
    ax.plot(eps, h["val_accuracy"], color=VAL_CLR,   label="Val", linestyle="--")
    ax.set_title("Classification Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 3: ROC AUC ──────────────────────────────────────────────────────
    ax = axes[2]
    if auc_key in h and val_auc_key in h:
        ax.plot(eps, h[auc_key],     color=TRAIN_CLR, label="Train (Keras)", alpha=0.8)
        ax.plot(eps, h[val_auc_key], color=VAL_CLR,   label="Val (Keras)",
                linestyle="--", alpha=0.8)
    ax.axhline(val_auc,   color=VAL_CLR,   linestyle=":", linewidth=2.0,
               label=f"Val final  = {val_auc:.4f}")
    ax.axhline(train_auc, color=TRAIN_CLR, linestyle=":", linewidth=2.0,
               label=f"Train final = {train_auc:.4f}")
    ax.set_title("ROC AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Hyperparameter annotation in figure footer
    param_str = "  |  ".join(
        f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in trial.params.items()
    )
    fig.text(0.5, -0.03, param_str, ha="center", fontsize=7,
             color="#555555", transform=fig.transFigure)

    plt.tight_layout()
    out_path = plot_dir / f"trial_{trial.number:03d}_learning_curve.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(
    trial:                optuna.Trial,
    X_train:              np.ndarray,
    y_train:              np.ndarray,
    X_val:                np.ndarray,
    y_val:                np.ndarray,
    n_features:           int,
    sample_weights_train: Optional[np.ndarray],
    plot_dir:             pathlib.Path,
    study_label:          str,
) -> float:
    """
    Generalisation-gap penalised AUC objective for Optuna maximisation.

        Score = Val_AUC − λ · max(0, Train_AUC − Val_AUC)

    When sample_weights_train is not None the training dataset yields 3-tuples
    (X, y, w) and Keras applies the weights to the BCE loss, altering the loss
    landscape compared to the unweighted study.

    AUC metrics for the objective are always computed with sklearn (unweighted,
    on the full sequence) so that scores remain directly comparable across both
    studies regardless of whether sample weighting was active.

    Learning-curve plots are saved for every COMPLETED (non-pruned) trial.
    Pruned trials raise optuna.TrialPruned inside model.fit() via the
    TFKerasPruningCallback, which causes the code below model.fit() to be
    skipped — no partial plot is written for pruned trials.
    """
    tf.keras.backend.clear_session()

    model = build_trial_model(trial, SEQ_LEN, n_features)

    train_ds,      y_train_al = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE,
        sample_weights=sample_weights_train, shuffle=True,
    )
    # Separate unshuffled, unweighted replica for Train AUC prediction — avoids
    # the shuffle-order misalignment that would collapse Train AUC toward 0.50
    train_eval_ds, _          = make_tf_dataset(
        X_train, y_train, SEQ_LEN, BATCH_SIZE,
        sample_weights=None, shuffle=False,
    )
    val_ds,        y_val_al   = make_tf_dataset(
        X_val, y_val, SEQ_LEN, BATCH_SIZE,
        sample_weights=None, shuffle=False,
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            TFKerasPruningCallback(trial, "val_auc"),
            EarlyStopping(
                monitor="val_auc",
                patience=25,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=12,
                min_lr=1e-7,
                verbose=0,
            ),
        ],
        verbose=0,
    )

    prob_train = model.predict(train_eval_ds, verbose=0).ravel()
    prob_val   = model.predict(val_ds,        verbose=0).ravel()

    train_auc = float(roc_auc_score(y_train_al, prob_train))
    val_auc   = float(roc_auc_score(y_val_al,   prob_val))
    gap       = train_auc - val_auc
    score     = val_auc - GAP_PENALTY * max(0.0, gap)

    try:
        tn, fp, fn, tp = confusion_matrix(y_val_al.astype(int),
                                          (prob_val >= 0.5).astype(int)).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        sensitivity, specificity = 0.0, 0.0

    trial.set_user_attr("train_auc",   train_auc)
    trial.set_user_attr("val_auc",     val_auc)
    trial.set_user_attr("gap",         gap)
    trial.set_user_attr("sensitivity", float(sensitivity))
    trial.set_user_attr("specificity", float(specificity))

    plot_learning_curves(history, trial, val_auc, train_auc, plot_dir, study_label)

    log.info(
        "[%s] Trial %3d | score=%+.4f | val=%.4f | train=%.4f | gap=%+.4f"
        " | u1=%d u2=%d dr=%.2f rd=%.2f lr=%.1e l2=%.1e opt=%s",
        study_label,
        trial.number, score, val_auc, train_auc, gap,
        trial.params["units_1"], trial.params["units_2"],
        trial.params["dropout_rate"], trial.params["recurrent_dropout"],
        trial.params["learning_rate"], trial.params["l2_reg"],
        trial.params["optimizer_name"],
    )
    return score


# =============================================================================
# STUDY RUNNER
# =============================================================================

def run_study(
    label:                str,
    db_uri:               str,
    xlsx_path:            str,
    json_path:            str,
    plot_dir:             pathlib.Path,
    n_trials:             int,
    X_train:              np.ndarray,
    y_train:              np.ndarray,
    X_val:                np.ndarray,
    y_val:                np.ndarray,
    n_features:           int,
    sample_weights_train: Optional[np.ndarray],
) -> optuna.Study:
    """
    Create (or resume) one Optuna study, run n_trials, export results, and
    save the best hyperparameters to a JSON file.
    """
    study_name = f"btc_lstm_{label}"
    log.info("=" * 65)
    log.info("  Starting Study : %s", study_name)
    log.info(
        "  use_sample_weights=%s  |  db=%s",
        sample_weights_train is not None, db_uri,
    )
    log.info("=" * 65)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        storage=db_uri,
        load_if_exists=True,
    )
    prior = len(study.trials)
    log.info("Study '%s' — %d prior trial(s) found", study_name, prior)

    try:
        study.optimize(
            lambda trial: objective(
                trial,
                X_train, y_train, X_val, y_val,
                n_features,
                sample_weights_train,
                plot_dir,
                label,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        log.info("[%s] Interrupted — reporting best trial so far.", label)

    # Guard: nothing to report if every trial was pruned / failed
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.warning("[%s] No completed trials — skipping summary and export.", label)
        return study

    best = study.best_trial

    print(f"\n{'═' * 64}")
    print(f"  [{label.upper()}]  BEST TRIAL SUMMARY")
    print(f"{'─' * 64}")
    print(f"  Trial number        :  {best.number}")
    print(f"  Objective score     :  {best.value:.4f}")
    print(f"  Val  AUC (sklearn)  :  {best.user_attrs.get('val_auc',    float('nan')):.4f}")
    print(f"  Train AUC (sklearn) :  {best.user_attrs.get('train_auc', float('nan')):.4f}")
    print(
        f"  Generalisation gap  :  "
        f"{best.user_attrs.get('gap', float('nan')):+.4f}  (λ={GAP_PENALTY})"
    )
    print(f"  Sensitivity @ 0.50  :  {best.user_attrs.get('sensitivity', float('nan')):.4f}")
    print(f"  Specificity @ 0.50  :  {best.user_attrs.get('specificity', float('nan')):.4f}")
    print(f"{'─' * 64}")
    print("  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<28s}:  {v}")
    print(f"{'─' * 64}")
    print(f"  Completed trials    :  {len(completed)} / {len(study.trials)}")
    print(f"  Plots saved to      :  {plot_dir}")
    print(f"{'═' * 64}\n")

    # ── Export all trials to Excel ────────────────────────────────────────────
    try:
        study.trials_dataframe().to_excel(xlsx_path, index=False)
        log.info(
            "[%s] Trials exported → %s  (%d rows)", label, xlsx_path, len(study.trials)
        )
    except Exception as exc:
        log.warning("[%s] Excel export failed: %s", label, exc)

    # ── Persist best hyperparameters as JSON ──────────────────────────────────
    best_record = {
        "study_label":     label,
        "use_weights":     sample_weights_train is not None,
        "trial_number":    best.number,
        "objective_score": round(best.value, 6),
        "val_auc":         round(best.user_attrs.get("val_auc",    float("nan")), 6),
        "train_auc":       round(best.user_attrs.get("train_auc", float("nan")), 6),
        "gap":             round(best.user_attrs.get("gap",        float("nan")), 6),
        "sensitivity":     round(best.user_attrs.get("sensitivity", float("nan")), 6),
        "specificity":     round(best.user_attrs.get("specificity", float("nan")), 6),
        "hyperparameters": {
            k: (round(v, 8) if isinstance(v, float) else v)
            for k, v in best.params.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(best_record, fh, indent=2)
    log.info("[%s] Best params saved → %s", label, json_path)

    return study


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def _fmtv(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def print_comparison_table(
    best_uw: optuna.Trial,
    best_w:  optuna.Trial,
) -> None:
    """Print a side-by-side comparison of both studies' optimal hyperparameters."""
    all_keys = sorted(set(best_uw.params) | set(best_w.params))

    C = max(28, max(len(k) for k in all_keys) + 2)
    V = 16
    DIV = "═" * (C + V * 2 + 8)

    print("\n" + DIV)
    print("  DUAL-STUDY COMPARISON  —  Optimal Hyperparameters & Metrics")
    print("─" * len(DIV))
    print(f"  {'Parameter':<{C}}  {'UNWEIGHTED':>{V}}  {'WEIGHTED':>{V}}")
    print("─" * len(DIV))

    for k in all_keys:
        uw = _fmtv(best_uw.params.get(k, "—"))
        w  = _fmtv(best_w.params.get(k,  "—"))
        marker = " ◄" if uw != w else ""
        print(f"  {k:<{C}}  {uw:>{V}}  {w:>{V}}{marker}")

    metrics = [
        ("Objective score",    "value",       "{:.4f}"),
        ("Val  AUC (sklearn)", "val_auc",     "{:.4f}"),
        ("Train AUC (sklearn)","train_auc",   "{:.4f}"),
        ("Generalisation gap", "gap",         "{:+.4f}"),
        ("Sensitivity @ 0.50", "sensitivity", "{:.4f}"),
        ("Specificity @ 0.50", "specificity", "{:.4f}"),
    ]
    print("─" * len(DIV))
    print(f"\n  {'Performance Metric':<{C}}  {'UNWEIGHTED':>{V}}  {'WEIGHTED':>{V}}")
    print("─" * len(DIV))
    for label, attr, fmt in metrics:
        if attr == "value":
            uw_m = fmt.format(best_uw.value)
            w_m  = fmt.format(best_w.value)
        else:
            uw_m = fmt.format(best_uw.user_attrs.get(attr, float("nan")))
            w_m  = fmt.format(best_w.user_attrs.get(attr,  float("nan")))
        print(f"  {label:<{C}}  {uw_m:>{V}}  {w_m:>{V}}")

    print(DIV + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main(n_trials: int = 100) -> None:
    log.info("=" * 65)
    log.info("  03_optuna_optimizer.py  v2  —  Dual-Study Bayesian HPO")
    log.info("  Sampler : TPE  |  Pruner : Median  |  Direction : maximise")
    log.info(
        "  %d trials × 2 studies (%d total)  |  "
        "SEQ_LEN=%d  BATCH=%d  EPOCHS=%d",
        n_trials, n_trials * 2, SEQ_LEN, BATCH_SIZE, EPOCHS,
    )
    log.info(
        "  Objective : Val_AUC − %.1f × max(0, Train_AUC − Val_AUC)",
        GAP_PENALTY,
    )
    log.info("=" * 65)

    # ── Create output directories ─────────────────────────────────────────────
    UNWEIGHTED_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTED_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Plot dirs  :  %s  |  %s",
        UNWEIGHTED_PLOT_DIR, WEIGHTED_PLOT_DIR,
    )

    # ── Load and scale once — arrays are shared across both studies ───────────
    train_df, val_df = load_and_split(INPUT_CSV)
    X_train, X_val, y_train, y_val, _ = scale_features(train_df, val_df)
    n_features = X_train.shape[1]
    log.info(
        "Input shape : (%d, %d, %d)  [windows × seq_len × features]",
        len(X_train) - SEQ_LEN + 1, SEQ_LEN, n_features,
    )

    # ── Precompute volatility weights (Study 2 only; computed once) ───────────
    vol_weights = compute_vol_weights(train_df)

    # ── Study 1 : Unweighted ──────────────────────────────────────────────────
    study_uw = run_study(
        label                = "unweighted",
        db_uri               = UNWEIGHTED_DB,
        xlsx_path            = UNWEIGHTED_XLSX,
        json_path            = "best_params_unweighted.json",
        plot_dir             = UNWEIGHTED_PLOT_DIR,
        n_trials             = n_trials,
        X_train              = X_train,
        y_train              = y_train,
        X_val                = X_val,
        y_val                = y_val,
        n_features           = n_features,
        sample_weights_train = None,
    )

    # ── Study 2 : Weighted ────────────────────────────────────────────────────
    study_w = run_study(
        label                = "weighted",
        db_uri               = WEIGHTED_DB,
        xlsx_path            = WEIGHTED_XLSX,
        json_path            = "best_params_weighted.json",
        plot_dir             = WEIGHTED_PLOT_DIR,
        n_trials             = n_trials,
        X_train              = X_train,
        y_train              = y_train,
        X_val                = X_val,
        y_val                = y_val,
        n_features           = n_features,
        sample_weights_train = vol_weights,
    )

    # ── Cross-study comparison ────────────────────────────────────────────────
    uw_complete = any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study_uw.trials
    )
    w_complete  = any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study_w.trials
    )
    if uw_complete and w_complete:
        print_comparison_table(study_uw.best_trial, study_w.best_trial)
    else:
        log.warning(
            "Skipping comparison table — one or both studies have no completed trials."
        )

    log.info(
        "Both studies complete  |  UW best=%.4f  |  W best=%.4f",
        study_uw.best_value if uw_complete else float("nan"),
        study_w.best_value  if w_complete  else float("nan"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-study Bayesian HPO (TPE) for BTC direction LSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optuna trials PER STUDY (total = 2 × N).",
    )
    args = parser.parse_args()
    main(n_trials=args.trials)
