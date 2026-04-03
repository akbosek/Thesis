#!/usr/bin/env python3
# =============================================================================
# 05_walk_forward.py
# =============================================================================
# PURPOSE:
#   Expanding-window walk-forward validation to test robustness against
#   concept drift across 4 years of out-of-sample (OOS) BTC data.
#
#   Each fold expands the training window by one calendar year, then tests
#   on the immediately following year.  The model is retrained from scratch
#   for every fold — no weight sharing across folds.
#
#   Folds:
#     Fold 1: Train [2015-01-01 → 2021-12-31]  →  Test [2022-01-01 → 2022-12-31]
#     Fold 2: Train [2015-01-01 → 2022-12-31]  →  Test [2023-01-01 → 2023-12-31]
#     Fold 3: Train [2015-01-01 → 2023-12-31]  →  Test [2024-01-01 → 2024-12-31]
#     Fold 4: Train [2015-01-01 → 2024-12-31]  →  Test [2025-01-01 → 2025-12-31]
#
#   Internal validation = last 10% of each train fold — used ONLY for
#   EarlyStopping; never reported.  StandardScaler is fitted on the training
#   portion only (last 10% excluded), preventing any future leakage.
#
#   Golden hyperparameters (from Optuna study btc_lstm_v3):
#     LSTM 512 → Dropout 0.25 → LSTM 256 → Dropout 0.25
#     L2 = 0.0005  |  RecurrentDropout = 0.25  |  LR = 0.001  |  Batch = 64
#
# OUTPUTS:
#   walk_forward_equity_curve.png   — Combined 4-year OOS equity curve
#                                     (Variant A / B / C vs BTC B&H)
#   run_log_walk_forward.txt        — Full session log
#
# USAGE:
#   python 05_walk_forward.py
#   Requires processed_dataset_v2.csv in the same directory.
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import sys
import os

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow as tf
from tensorflow.keras import Sequential          # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.regularizers import l2 as L2Reg     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix as cm_fn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =============================================================================
# ███  CONFIGURATION  ███
# =============================================================================

# ── Walk-forward fold boundaries ─────────────────────────────────────────────
GLOBAL_START: str = "2015-01-01"   # first day of training data across all folds

FOLDS: list[dict] = [
    {"fold": 1, "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"fold": 2, "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"fold": 3, "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"fold": 4, "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
]

# ── Internal validation split (no-leak EarlyStopping proxy) ──────────────────
# Last VAL_FRAC of each training fold is withheld as internal val.
# scaler.fit() uses only the training portion (first 1-VAL_FRAC rows).
VAL_FRAC: float = 0.10

# ── Golden hyperparameters ────────────────────────────────────────────────────
SEQ_LEN:    int   = 30
BATCH_SIZE: int   = 64
EPOCHS:     int   = 100   # hard ceiling; EarlyStopping(patience=10) fires earlier

UNITS_1:   int   = 512
UNITS_2:   int   = 256
DROPOUT:   float = 0.25
REC_DROP:  float = 0.25
L2:        float = 0.0005
LR:        float = 0.001

# ── Variant thresholds ────────────────────────────────────────────────────────
UPPER_THRESHOLD: float = 0.52
LOWER_THRESHOLD: float = 0.48

# ── Variant C weight bounds ───────────────────────────────────────────────────
VC_MIN_WEIGHT: float = 0.25
VC_MAX_WEIGHT: float = 1.00

# ── Feature pruning — must match 02 and 03 exactly ───────────────────────────
PRUNED_FEATURES: frozenset = frozenset({"gold_log_return", "nvda_log_return"})

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV:    str = "processed_dataset_v2.csv"
EQUITY_PLOT:  str = "walk_forward_equity_curve.png"
LOG_FILE:     str = "run_log_walk_forward.txt"

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
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA HELPERS
# =============================================================================

def _compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Per-window sample weights from abs(btc_log_return) of the target day.
    weight[i] = abs(btc_log_return[i + SEQ_LEN − 1]), normalised to mean = 1.0.
    Returns shape (N − SEQ_LEN + 1,), dtype float32.
    """
    raw      = np.abs(df["btc_log_return"].values[SEQ_LEN - 1:]).astype(np.float32)
    mean_val = float(raw.mean())
    if mean_val < 1e-8:
        return np.ones(len(raw), dtype=np.float32)
    return raw / mean_val


def _make_dataset(
    X:              np.ndarray,
    y:              np.ndarray,
    shuffle:        bool = False,
    sample_weights: np.ndarray | None = None,
) -> tuple:
    """
    Build a tf.data.Dataset of sliding windows (SEQ_LEN steps, stride 1).

    Window i covers X[i : i+SEQ_LEN]; label = y[i + SEQ_LEN - 1].
    When sample_weights is provided the dataset yields (x, y, w) triples
    so Keras applies per-sample loss scaling automatically.
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
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds, y_aligned

    # ── Default path (no weights) ─────────────────────────────────────────────
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
    return dataset, y_aligned


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model(n_features: int) -> tf.keras.Model:
    """Build a 2-layer stacked LSTM with the golden hyperparameters."""
    model = Sequential([
        LSTM(
            units=UNITS_1,
            return_sequences=True,
            kernel_regularizer=L2Reg(L2),
            recurrent_dropout=REC_DROP,
            input_shape=(SEQ_LEN, n_features),
            name="lstm_1",
        ),
        Dropout(DROPOUT, name="drop_1"),
        LSTM(
            units=UNITS_2,
            return_sequences=False,
            kernel_regularizer=L2Reg(L2),
            recurrent_dropout=REC_DROP,
            name="lstm_2",
        ),
        Dropout(DROPOUT, name="drop_2"),
        Dense(1, activation="sigmoid", name="output"),
    ], name="wf_lstm_golden")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _variant_a_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Variant A (0.5 threshold): AUC, Gini, Accuracy, Sensitivity, Specificity."""
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


def _variant_b_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Variant B (3-state filter): Coverage + Conditional Win Rate on traded days."""
    trade_mask = (y_prob > UPPER_THRESHOLD) | (y_prob < LOWER_THRESHOLD)
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


def _compute_variant_c_weights(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Variant C signals (+1/0/−1) and continuous position weights.

    Long  P > UPPER : signal=+1, weight = VC_MIN + (VC_MAX-VC_MIN) × (P-UPPER) / (1-UPPER)
    Short P < LOWER : signal=−1, weight = VC_MIN + (VC_MAX-VC_MIN) × (LOWER-P) / LOWER
    Hold  otherwise : signal= 0, weight = 0.0
    """
    signals = np.zeros(len(probs), dtype=np.float32)
    weights = np.zeros(len(probs), dtype=np.float32)

    long_mask  = probs > UPPER_THRESHOLD
    short_mask = probs < LOWER_THRESHOLD
    weight_range = VC_MAX_WEIGHT - VC_MIN_WEIGHT

    signals[long_mask]  = 1.0
    signals[short_mask] = -1.0

    weights[long_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (probs[long_mask] - UPPER_THRESHOLD)
        / (1.0 - UPPER_THRESHOLD)
    )
    weights[short_mask] = (
        VC_MIN_WEIGHT
        + weight_range * (LOWER_THRESHOLD - probs[short_mask])
        / LOWER_THRESHOLD
    )
    return signals, weights


def _compute_quant_metrics(daily_returns: np.ndarray, trading_days: int = 252) -> dict:
    """
    Annualised Sharpe (rf=0), Annualised Volatility, Maximum Drawdown
    from an array of daily log-returns.
    """
    if len(daily_returns) == 0:
        return dict(sharpe=float("nan"), vol=float("nan"), max_dd=float("nan"))

    mean_ret = float(np.mean(daily_returns))
    std_ret  = float(np.std(daily_returns, ddof=1))
    sharpe   = (mean_ret / std_ret * np.sqrt(trading_days)) if std_ret > 0 else 0.0
    vol      = std_ret * np.sqrt(trading_days)

    cum    = np.exp(np.cumsum(daily_returns))
    peak   = np.maximum.accumulate(cum)
    dd     = (cum - peak) / peak
    max_dd = float(dd.min())

    return dict(sharpe=float(sharpe), vol=float(vol), max_dd=float(max_dd))


# =============================================================================
# SINGLE-FOLD PIPELINE
# =============================================================================

def run_fold(
    fold_cfg: dict,
    df:       pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Execute one expanding-window fold end-to-end.

    Pipeline per fold:
      1. Slice train / test rows by date boundaries.
      2. Split train into training portion (first 90%) and internal val (last 10%).
      3. Fit StandardScaler on the training portion only.
      4. Build windowed datasets with sample weighting.
      5. Train with EarlyStopping(patience=10) monitoring internal val loss.
      6. Predict on OOS test.
      7. Compute Variant A / B / C metrics on OOS test.

    Returns a dict with:
        fold        : fold number
        dates       : DatetimeIndex of OOS test dates (SEQ_LEN-aligned)
        btc_returns : np.ndarray — BTC daily log-returns (aligned)
        prob_oos    : np.ndarray — model probabilities on OOS test
        y_oos       : np.ndarray — true binary labels (aligned)
        metrics_a   : Variant A metrics dict
        metrics_b   : Variant B metrics dict
        ret_a / ret_b / ret_c : per-day strategy log-returns
    """
    fold_num = fold_cfg["fold"]
    log.info("─" * 65)
    log.info("  FOLD %d  |  Train → %s  |  Test: %s → %s",
             fold_num, fold_cfg["train_end"],
             fold_cfg["test_start"], fold_cfg["test_end"])
    log.info("─" * 65)

    # ── Slice ─────────────────────────────────────────────────────────────────
    train_mask = (df.index >= GLOBAL_START) & (df.index <= fold_cfg["train_end"])
    test_mask  = (df.index >= fold_cfg["test_start"]) & (df.index <= fold_cfg["test_end"])

    train_df_full = df[train_mask]
    test_df       = df[test_mask]

    n_train_full = len(train_df_full)
    n_val_rows   = max(SEQ_LEN + 1, int(np.ceil(n_train_full * VAL_FRAC)))
    n_core_rows  = n_train_full - n_val_rows

    train_df_core = train_df_full.iloc[:n_core_rows]
    val_df_int    = train_df_full.iloc[n_core_rows:]

    log.info(
        "  Split: core_train=%d rows  |  internal_val=%d rows  |  test=%d rows",
        n_core_rows, len(val_df_int), len(test_df),
    )

    # ── Scaler: fit on core_train only ────────────────────────────────────────
    X_core = train_df_core[feature_cols].values.astype(np.float32)
    y_core = train_df_core["target"].values.astype(np.float32)

    X_val  = val_df_int[feature_cols].values.astype(np.float32)
    y_val  = val_df_int["target"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_core  = scaler.fit_transform(X_core)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Sample weights ────────────────────────────────────────────────────────
    core_weights = _compute_sample_weights(train_df_core)
    log.info(
        "  Sample weights — core: n=%d mean=%.4f  min=%.4f  max=%.4f",
        len(core_weights), core_weights.mean(), core_weights.min(), core_weights.max(),
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds, _               = _make_dataset(X_core, y_core, shuffle=True,
                                              sample_weights=core_weights)
    val_ds,   y_val_aligned   = _make_dataset(X_val,  y_val,  shuffle=False)
    test_ds,  y_test_aligned  = _make_dataset(X_test, y_test, shuffle=False)

    # ── Reproducibility reset ─────────────────────────────────────────────────
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ── Build model ───────────────────────────────────────────────────────────
    n_features = X_core.shape[1]
    model = build_model(n_features)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0,
        ),
    ]

    log.info("  Training fold %d (max %d epochs, EarlyStopping patience=10) ...",
             fold_num, EPOCHS)
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )
    actual_epochs = len(history.history["val_loss"])
    best_val_loss = float(min(history.history["val_loss"]))
    log.info("  Fold %d training done — actual epochs: %d  |  best val_loss: %.5f",
             fold_num, actual_epochs, best_val_loss)

    # ── Predict on OOS test ───────────────────────────────────────────────────
    prob_oos = model.predict(test_ds, verbose=0).flatten()

    # Alignment: OOS predictions map to test rows [SEQ_LEN-1 : SEQ_LEN-1+n_windows]
    n_oos       = len(prob_oos)                               # = len(test_df) - SEQ_LEN + 1
    btc_returns = test_df["btc_log_return"].values[SEQ_LEN - 1 : SEQ_LEN - 1 + n_oos]
    dates_oos   = test_df.index[SEQ_LEN - 1 : SEQ_LEN - 1 + n_oos]

    # ── Variant A ─────────────────────────────────────────────────────────────
    m_a     = _variant_a_metrics(y_test_aligned, prob_oos)
    signal_a = np.where(prob_oos >= 0.5, 1.0, -1.0)
    ret_a    = signal_a * btc_returns

    # ── Variant B ─────────────────────────────────────────────────────────────
    m_b      = _variant_b_metrics(y_test_aligned, prob_oos)
    signal_b = np.select(
        [prob_oos > UPPER_THRESHOLD, prob_oos < LOWER_THRESHOLD],
        [1.0, -1.0],
        default=0.0,
    )
    ret_b = signal_b * btc_returns

    # ── Variant C ─────────────────────────────────────────────────────────────
    sig_c, wt_c = _compute_variant_c_weights(prob_oos)
    ret_c       = sig_c * wt_c * btc_returns

    # ── Per-fold report ───────────────────────────────────────────────────────
    q_a = _compute_quant_metrics(ret_a)
    q_b = _compute_quant_metrics(ret_b)
    q_c = _compute_quant_metrics(ret_c)
    q_btc = _compute_quant_metrics(btc_returns)

    print(f"\n{'═' * 65}")
    print(f"  Fold {fold_num}  |  OOS: {fold_cfg['test_start']} → {fold_cfg['test_end']}")
    print(f"  Epochs trained: {actual_epochs}")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<22}  {'Var A':>8}  {'Var B':>8}  {'Var C':>8}  {'BTC B&H':>8}")
    print(f"  {'AUC (Var A)':<22}  {m_a['auc']:>8.4f}")
    print(f"  {'Accuracy':<22}  {m_a['accuracy']:>8.4f}")
    print(f"  {'Sensitivity':<22}  {m_a['sensitivity']:>8.4f}")
    print(f"  {'Specificity':<22}  {m_a['specificity']:>8.4f}")
    print(f"  {'B Coverage':<22}  {'':>8}  {m_b['coverage']:>8.4f}")
    print(f"  {'B Cond WR':<22}  {'':>8}  {m_b['conditional_win_rate']:>8.4f}" if not np.isnan(m_b["conditional_win_rate"]) else f"  {'B Cond WR':<22}  {'':>8}  {'N/A':>8}")
    print(f"  {'Total LogRet':<22}  {np.sum(ret_a):>+8.4f}  {np.sum(ret_b):>+8.4f}  {np.sum(ret_c):>+8.4f}  {np.sum(btc_returns):>+8.4f}")
    print(f"  {'Sharpe (ann)':<22}  {q_a['sharpe']:>+8.3f}  {q_b['sharpe']:>+8.3f}  {q_c['sharpe']:>+8.3f}  {q_btc['sharpe']:>+8.3f}")
    print(f"  {'Vol (ann)':<22}  {q_a['vol']:>8.3f}  {q_b['vol']:>8.3f}  {q_c['vol']:>8.3f}  {q_btc['vol']:>8.3f}")
    print(f"  {'Max Drawdown':<22}  {q_a['max_dd']:>+8.3f}  {q_b['max_dd']:>+8.3f}  {q_c['max_dd']:>+8.3f}  {q_btc['max_dd']:>+8.3f}")
    print(f"{'═' * 65}\n")

    return dict(
        fold        = fold_num,
        test_start  = fold_cfg["test_start"],
        test_end    = fold_cfg["test_end"],
        dates       = dates_oos,
        btc_returns = btc_returns,
        prob_oos    = prob_oos,
        y_oos       = y_test_aligned,
        metrics_a   = m_a,
        metrics_b   = m_b,
        ret_a       = ret_a,
        ret_b       = ret_b,
        ret_c       = ret_c,
    )


# =============================================================================
# WALK-FORWARD EQUITY CURVE PLOT
# =============================================================================

FOLD_COLORS = ["#e05c5c", "#e0a03c", "#5c9ee0", "#5cc47c"]   # one per fold


def plot_walk_forward_equity(fold_results: list[dict]) -> None:
    """
    Plot the combined 4-year OOS equity curve.

    Each of the four strategies (Variant A, B, C, BTC B&H) is shown as
    a single continuous line across all fold test periods.  Vertical
    dashed lines mark fold boundaries.  Individual fold backgrounds are
    lightly shaded to show which returns came from which fold.

    Saves EQUITY_PLOT (dpi=150).
    """
    # ── Concatenate across folds ──────────────────────────────────────────────
    all_dates   = np.concatenate([r["dates"]       for r in fold_results])
    all_btc     = np.concatenate([r["btc_returns"] for r in fold_results])
    all_ret_a   = np.concatenate([r["ret_a"]       for r in fold_results])
    all_ret_b   = np.concatenate([r["ret_b"]       for r in fold_results])
    all_ret_c   = np.concatenate([r["ret_c"]       for r in fold_results])

    # Convert log-returns → cumulative %-change (start = 0 %)
    def to_pct(rets: np.ndarray) -> np.ndarray:
        return (np.exp(np.cumsum(rets)) - 1.0) * 100.0

    pct_a   = to_pct(all_ret_a)
    pct_b   = to_pct(all_ret_b)
    pct_c   = to_pct(all_ret_c)
    pct_btc = to_pct(all_btc)

    q_a   = _compute_quant_metrics(all_ret_a)
    q_b   = _compute_quant_metrics(all_ret_b)
    q_c   = _compute_quant_metrics(all_ret_c)
    q_btc = _compute_quant_metrics(all_btc)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 8))

    # Fold background shading
    boundaries = [pd.Timestamp(r["test_start"]) for r in fold_results]
    boundaries.append(pd.Timestamp(fold_results[-1]["test_end"]))
    for i, r in enumerate(fold_results):
        t0 = pd.Timestamp(r["test_start"])
        t1 = pd.Timestamp(r["test_end"])
        ax.axvspan(t0, t1, alpha=0.05, color=FOLD_COLORS[i], zorder=0)
        ax.axvline(t0, color=FOLD_COLORS[i], lw=1.0, ls="--", alpha=0.45, zorder=1)
        mid = t0 + (t1 - t0) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] else 1,
                f"Fold {r['fold']}", ha="center", va="top",
                fontsize=8, color=FOLD_COLORS[i], alpha=0.7)

    # Strategy lines
    dates_dt = pd.DatetimeIndex(all_dates)

    ax.plot(dates_dt, pct_btc, lw=2.0, color="dimgray",   ls="-.",  zorder=2,
            label=f"BTC Buy & Hold  (Sharpe: {q_btc['sharpe']:+.2f})  →  {pct_btc[-1]:+.1f}%")
    ax.plot(dates_dt, pct_a,   lw=1.8, color="steelblue", ls="-",   zorder=3,
            label=f"Variant A  (Sharpe: {q_a['sharpe']:+.2f})  →  {pct_a[-1]:+.1f}%")
    ax.plot(dates_dt, pct_b,   lw=1.8, color="darkorange",ls="--",  zorder=3,
            label=f"Variant B  (Sharpe: {q_b['sharpe']:+.2f})  →  {pct_b[-1]:+.1f}%")
    ax.plot(dates_dt, pct_c,   lw=2.5, color="seagreen",  ls="-",   zorder=4,
            label=f"Variant C  (Sharpe: {q_c['sharpe']:+.2f})  →  {pct_c[-1]:+.1f}%")

    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)

    # ── Labels / formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_title(
        "Walk-Forward Validation — 4-Year OOS Equity Curve  (2022–2025)\n"
        f"Expanding Train Window  |  Golden Params: 512→256  drop=0.25  l2={L2}  lr={LR}  bs={BATCH_SIZE}\n"
        "Fold boundaries are dashed vertical lines",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── Annotation table ──────────────────────────────────────────────────────
    table_lines = [
        f"{'Strategy':<12}  {'Tot%':>7}  {'Sharpe':>7}  {'Vol':>6}  {'MDD':>7}",
        "─" * 47,
        f"{'Var A':<12}  {pct_a[-1]:>+7.1f}  {q_a['sharpe']:>+7.3f}  {q_a['vol']:>6.3f}  {q_a['max_dd']:>+7.3f}",
        f"{'Var B':<12}  {pct_b[-1]:>+7.1f}  {q_b['sharpe']:>+7.3f}  {q_b['vol']:>6.3f}  {q_b['max_dd']:>+7.3f}",
        f"{'Var C':<12}  {pct_c[-1]:>+7.1f}  {q_c['sharpe']:>+7.3f}  {q_c['vol']:>6.3f}  {q_c['max_dd']:>+7.3f}",
        f"{'BTC B&H':<12}  {pct_btc[-1]:>+7.1f}  {q_btc['sharpe']:>+7.3f}  {q_btc['vol']:>6.3f}  {q_btc['max_dd']:>+7.3f}",
    ]
    ax.text(
        0.99, 0.03, "\n".join(table_lines),
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
        fontfamily="monospace",
    )

    plt.tight_layout()
    fig.savefig(EQUITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Walk-forward equity curve saved → %s", EQUITY_PLOT)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 65)
    log.info("  05_walk_forward.py  —  START")
    log.info("  Golden params: units=[%d→%d]  drop=%.2f  rec_drop=%.2f",
             UNITS_1, UNITS_2, DROPOUT, REC_DROP)
    log.info("  L2=%.5f  LR=%.4f  Batch=%d  SEQ_LEN=%d",
             L2, LR, BATCH_SIZE, SEQ_LEN)
    log.info("  Internal val split: last %.0f%% of each train fold", VAL_FRAC * 100)
    log.info("=" * 65)

    # ── Load full dataset once ────────────────────────────────────────────────
    log.info("Loading '%s' ...", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    log.info(
        "Full dataset: %d rows  [%s → %s]",
        len(df), df.index.min().date(), df.index.max().date(),
    )

    feature_cols = [
        c for c in df.columns
        if c != "target" and c not in PRUNED_FEATURES
    ]
    if PRUNED_FEATURES:
        log.info("Pruned toxic features: %s", sorted(PRUNED_FEATURES))
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    # ── Run all folds ─────────────────────────────────────────────────────────
    fold_results = []
    for fold_cfg in FOLDS:
        result = run_fold(fold_cfg, df, feature_cols)
        fold_results.append(result)

    # ── Aggregate across all folds ────────────────────────────────────────────
    all_ret_a   = np.concatenate([r["ret_a"]       for r in fold_results])
    all_ret_b   = np.concatenate([r["ret_b"]       for r in fold_results])
    all_ret_c   = np.concatenate([r["ret_c"]       for r in fold_results])
    all_btc     = np.concatenate([r["btc_returns"] for r in fold_results])
    all_y       = np.concatenate([r["y_oos"]       for r in fold_results])
    all_prob    = np.concatenate([r["prob_oos"]    for r in fold_results])

    q_a   = _compute_quant_metrics(all_ret_a)
    q_b   = _compute_quant_metrics(all_ret_b)
    q_c   = _compute_quant_metrics(all_ret_c)
    q_btc = _compute_quant_metrics(all_btc)
    auc_4yr = float(roc_auc_score(all_y, all_prob))

    print("\n" + "═" * 65)
    print("  WALK-FORWARD SUMMARY — 4-Year Concatenated OOS (2022–2025)")
    print("═" * 65)
    print(f"  Total OOS days (windows) : {len(all_ret_a)}")
    print(f"  Concatenated OOS AUC     : {auc_4yr:.4f}")
    print(f"{'─' * 65}")
    print(f"  {'Strategy':<14}  {'TotLogRet':>10}  {'Sharpe':>8}  {'Vol':>6}  {'MaxDD':>8}")
    for lbl, rets, q in [
        ("Variant A",  all_ret_a,  q_a),
        ("Variant B",  all_ret_b,  q_b),
        ("Variant C",  all_ret_c,  q_c),
        ("BTC B&H",    all_btc,    q_btc),
    ]:
        print(f"  {lbl:<14}  {np.sum(rets):>+10.4f}  {q['sharpe']:>+8.3f}  "
              f"{q['vol']:>6.3f}  {q['max_dd']:>+8.3f}")
    print("═" * 65 + "\n")

    # ── Plot combined equity curve ────────────────────────────────────────────
    log.info("Generating walk-forward equity curve ...")
    plot_walk_forward_equity(fold_results)

    log.info("=" * 65)
    log.info("  05_walk_forward.py  —  COMPLETE")
    log.info("  Plot → %s", EQUITY_PLOT)
    log.info("=" * 65)


# =============================================================================
if __name__ == "__main__":
    main()
