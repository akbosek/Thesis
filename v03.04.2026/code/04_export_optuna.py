#!/usr/bin/env python3
# =============================================================================
# 04_export_optuna.py
# =============================================================================
# PURPOSE:
#   Standalone post-processing script for the btc_lstm_v3 Optuna study.
#   Reads the SQLite DB produced by 03_hyperparameter_tuning_v2.py and:
#     1. Exports all COMPLETE trials to a formatted XLSX file.
#     2. Produces a train_auc vs val_auc scatter plot colour-coded by the
#        generalization gap (auc_diff = train − val), with a y = x diagonal
#        so overfitting trials are immediately visible below the line.
#
# OUTPUTS:
#   tuning_results_final.xlsx   — full sorted trial table
#   train_vs_val_auc.png        — scatter: train AUC × val AUC × gap colour
#
# USAGE:
#   python 04_export_optuna.py
#   Run from the same directory as optuna_study_v2.db.
#
# REQUIREMENTS:
#   optuna, pandas, numpy, matplotlib
#   openpyxl (for XLSX):  pip install openpyxl
# =============================================================================

import logging
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION — must match 03_hyperparameter_tuning_v2.py exactly
# =============================================================================
STUDY_NAME: str = "btc_lstm_v3"
DB_PATH:    str = "sqlite:///optuna_study_v2.db"
XLSX_OUT:   str = "tuning_results_final.xlsx"
PLOT_OUT:   str = "train_vs_val_auc.png"

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — LOAD STUDY
# =============================================================================

def load_study() -> optuna.Study:
    """Load the Optuna study from the SQLite DB with clear failure messages."""
    log.info("Loading study '%s' from %s ...", STUDY_NAME, DB_PATH)
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    except KeyError:
        log.error(
            "Study '%s' not found in %s. "
            "Run 03_hyperparameter_tuning_v2.py first.",
            STUDY_NAME, DB_PATH,
        )
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to load study: %s", exc)
        sys.exit(1)

    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    log.info(
        "Study loaded — %d total trials, %d COMPLETE.",
        len(study.trials), n_complete,
    )
    return study


# =============================================================================
# STEP 2 — BUILD DATAFRAME
# =============================================================================

def build_dataframe(study: optuna.Study) -> pd.DataFrame:
    """
    Extract the trials dataframe, filter to COMPLETE trials, and sort by
    val_a_auc descending (falls back to the raw Optuna objective value).

    Column naming after trials_dataframe():
        user_attrs_train_a_auc   — train AUC (Variant A)
        user_attrs_val_a_auc     — val AUC (same as objective value)
        user_attrs_auc_diff      — train − val (generalization gap)
        user_attrs_acc_diff      — train − val accuracy gap
    """
    df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )

    complete_mask = df["state"] == "COMPLETE"
    n_complete    = int(complete_mask.sum())
    n_total       = len(df)

    log.info(
        "Trials: %d total | %d COMPLETE | %d FAIL/PRUNED",
        n_total, n_complete, n_total - n_complete,
    )

    if n_complete == 0:
        log.error(
            "No COMPLETE trials found in study '%s'. Nothing to export.",
            STUDY_NAME,
        )
        sys.exit(1)

    df = df[complete_mask].copy()
    df.rename(columns={"value": "val_auc_objective"}, inplace=True)

    # Sort by the user_attrs val AUC when available; fall back to objective.
    sort_col = "user_attrs_val_a_auc"
    if sort_col in df.columns:
        df.sort_values(sort_col, ascending=False, inplace=True)
    else:
        log.warning(
            "Column '%s' not found — sorting by 'val_auc_objective' instead. "
            "This happens when the study was run with an older version of the script.",
            sort_col,
        )
        df.sort_values("val_auc_objective", ascending=False, inplace=True)

    df.reset_index(drop=True, inplace=True)
    log.info("DataFrame ready: %d rows × %d columns.", len(df), len(df.columns))
    return df


# =============================================================================
# STEP 3 — EXPORT XLSX
# =============================================================================

def export_xlsx(df: pd.DataFrame) -> None:
    """Write the full sorted trials DataFrame to XLSX (requires openpyxl)."""
    try:
        df.to_excel(XLSX_OUT, index=False)
        log.info("XLSX exported → %s  (%d rows)", XLSX_OUT, len(df))
    except ImportError:
        log.warning(
            "XLSX export skipped — openpyxl is not installed.  "
            "Run:  pip install openpyxl"
        )
    except Exception as exc:
        log.error("XLSX export failed: %s", exc)


# =============================================================================
# STEP 4 — SCATTER PLOT: TRAIN AUC × VAL AUC × GENERALIZATION GAP
# =============================================================================

def plot_scatter(df: pd.DataFrame) -> None:
    """
    Scatter plot: Train AUC (x) vs Val AUC (y), colour-coded by auc_diff.

    Visual encoding:
        Colour  — auc_diff = train − val  (cmap RdYlGn_r)
                  Red:   large positive → overfitting (train >> val)
                  Green: near zero / negative → well-generalised
        Diagonal y = x → perfect generalization reference
                  Points BELOW line: Train AUC > Val AUC (overfitting)
                  Points ABOVE line: Val AUC > Train AUC (unusual, good)
        Gold star — best trial by Val AUC
    """
    col_train = "user_attrs_train_a_auc"
    col_val   = "user_attrs_val_a_auc"
    col_diff  = "user_attrs_auc_diff"

    missing = [c for c in (col_train, col_val, col_diff) if c not in df.columns]
    if missing:
        log.warning(
            "Columns missing from DataFrame: %s — scatter plot skipped. "
            "Ensure 03_hyperparameter_tuning_v2.py sets the auc_diff user_attr.",
            missing,
        )
        return

    train_auc = df[col_train].values.astype(float)
    val_auc   = df[col_val].values.astype(float)
    auc_diff  = df[col_diff].values.astype(float)

    # Best trial by val AUC (highest is best)
    best_idx      = int(np.nanargmax(val_auc))
    best_trial_no = int(df["number"].iloc[best_idx]) if "number" in df.columns else best_idx
    best_train    = float(train_auc[best_idx])
    best_val      = float(val_auc[best_idx])

    # Axis limits: clamp to [0.48, 1.0] to keep diagonal meaningful
    all_vals = np.concatenate([train_auc, val_auc])
    lim_min  = max(0.48, float(np.nanmin(all_vals)) - 0.01)
    lim_max  = min(1.00, float(np.nanmax(all_vals)) + 0.01)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        train_auc, val_auc,
        c=auc_diff,
        cmap="RdYlGn_r",   # red = overfitting, green = good generalization
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=45,
        zorder=3,
    )

    # y = x diagonal (perfect generalization)
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        color="steelblue", linestyle="--", linewidth=1.5,
        alpha=0.65, label="y = x  (no generalization gap)",
        zorder=2,
    )

    # Best trial: gold star + arrow annotation
    ax.scatter(
        best_train, best_val,
        marker="*", s=260, color="gold", edgecolors="black",
        linewidths=0.8, zorder=5,
        label=f"Best trial #{best_trial_no}  (Val AUC = {best_val:.4f})",
    )
    # Offset annotation so it doesn't overlap the star; clamp to stay in frame
    txt_x = best_train + (0.004 if best_train < lim_max - 0.02 else -0.02)
    txt_y = best_val   - (0.007 if best_val   > lim_min + 0.02 else -0.01)
    ax.annotate(
        f"Trial #{best_trial_no}\nVal={best_val:.4f}\nTrain={best_train:.4f}",
        xy=(best_train, best_val),
        xytext=(txt_x, txt_y),
        fontsize=8.0,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(
        "AUC Diff  (train − val)\n"
        "< 0  →  val > train  (good)\n"
        "> 0  →  overfitting",
        fontsize=8.5,
    )

    ax.set_xlabel("Train AUC  (Variant A, 0.5 threshold)", fontsize=12)
    ax.set_ylabel("Val AUC  (Optuna objective)", fontsize=12)
    ax.set_title(
        f"Train vs Val AUC — Optuna Study  '{STUDY_NAME}'\n"
        f"n = {len(df)} completed trials  ·  "
        f"Best Val AUC = {best_val:.4f}  (Trial #{best_trial_no})\n"
        f"Trials below the diagonal are overfitting  (Train AUC > Val AUC)",
        fontsize=11, pad=12,
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Scatter plot saved → %s", PLOT_OUT)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("  04_export_optuna.py  —  START")
    log.info("  Study : %s", STUDY_NAME)
    log.info("  DB    : %s", DB_PATH)
    log.info("=" * 60)

    study = load_study()
    df    = build_dataframe(study)

    log.info("[1/2] Exporting XLSX ...")
    export_xlsx(df)

    log.info("[2/2] Generating scatter plot ...")
    plot_scatter(df)

    log.info("=" * 60)
    log.info("  04_export_optuna.py  —  COMPLETE")
    log.info("  XLSX → %s", XLSX_OUT)
    log.info("  Plot → %s", PLOT_OUT)
    log.info("=" * 60)


# =============================================================================
if __name__ == "__main__":
    main()
