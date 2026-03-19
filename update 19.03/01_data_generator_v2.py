#!/usr/bin/env python3
# =============================================================================
# 01_data_generator_v2.py
# =============================================================================
# PURPOSE:
#   Version 2 of the data generator.  Identical feature engineering pipeline
#   to v1 with two structural upgrades:
#
#   [V2-1] USE_RAW_PRICES toggle
#          When False (default):  log-returns for all 6 assets are the model
#          features — stationary, scale-invariant, neural-network-friendly.
#          When True:  raw adjusted close prices replace log-returns for the
#          5 non-BTC price assets; btc_log_return is KEPT as an auxiliary
#          reference column (the model script needs it for equity-curve
#          computation regardless of feature mode).
#          NOTE: Derived features (RSI, rolling vol, volume ratio, target)
#          are ALWAYS computed from btc_log_return internally, regardless of
#          the toggle — they are based on price *changes*, not price levels.
#          Target is ALWAYS binary next-day BTC direction (never changes).
#
#   [V2-2] Extended fetch window (START_DATE = "2014-11-01")
#          The v1 generator fetched from 2015-01-01. After the 21-row rolling-
#          vol burn-in, the first valid row was ≈ 2015-01-23 — losing the first
#          3 weeks of 2015. Moving the fetch start to 2014-11-01 ensures the
#          first valid CSV row falls before 2015-01-01, so the model scripts
#          can cleanly split training data from exactly 2015-01-01 without
#          discarding any early-2015 observations.
#
# PIPELINE STEPS:
#   1. Fetch    – Download adjusted daily Close (+ BTC Volume) for 6 assets.
#   2. Sync     – Left-join on BTC calendar; forward-fill missing values.
#   3. Engineer – Weekend flag, log-returns, VIX level, BTC Volume Ratio,
#                 RSI(14), Rolling Volatility(21), binary target.
#                 [V2-1] Conditionally swap log-returns for raw closes.
#   4. Visualize– Pearson correlation heatmap (all features + target).
#   5. Export   – Save cleaned DataFrame to processed_dataset_v2.csv.
#
# OUTPUTS:
#   • processed_dataset_v2.csv    – NaN-free, feature-engineered dataset
#   • correlation_heatmap_v2.png  – seaborn Pearson correlation heatmap
#
# FEATURE SET — USE_RAW_PRICES=False  (default, 11 input features + 1 target):
#   Cross-asset returns : btc, spy, gold, nvda, dxy, vix  (_log_return)
#   Macro level         : vix_level         (raw VIX close, stationary)
#   Microstructure      : btc_volume_ratio  (log(V / 20d rolling mean))
#   Momentum            : btc_rsi_14        (Wilder RSI, normalised [0,1])
#   Regime / Vol        : btc_roll_vol_21   (21d rolling std of BTC returns)
#   Calendar            : is_weekend        (binary)
#   Target              : next-day BTC direction (binary)
#
# FEATURE SET — USE_RAW_PRICES=True  (11 input features + btc ref + 1 target):
#   Cross-asset PRICES  : btc_close, spy_close, gold_close, nvda_close,
#                         dxy_close  (adjusted raw close prices)
#   Macro level         : vix_level  (raw VIX close = same as vix_close)
#   Microstructure      : btc_volume_ratio  (unchanged)
#   Momentum            : btc_rsi_14        (unchanged — derived from log-ret)
#   Regime / Vol        : btc_roll_vol_21   (unchanged — derived from log-ret)
#   Calendar            : is_weekend        (unchanged)
#   Aux reference       : btc_log_return    (NOT a model feature; kept for
#                                            equity-curve computation in model)
#   Target              : next-day BTC direction (binary, unchanged)
#
# REQUIREMENTS:
#   pip install yfinance pandas numpy matplotlib seaborn
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless/server use
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# ███  GLOBAL CONFIGURATION  ███
# Edit ONLY this block to change the time window, feature mode, or paths.
# =============================================================================

# [V2-1] Master toggle for feature representation mode.
#   False (default) → log-returns (stationary, recommended for LSTM training).
#   True            → raw adjusted close prices (experimental; the model script
#                     must also be set up to handle non-stationary price inputs,
#                     e.g., by including sufficient regularisation and the
#                     btc_log_return reference column for equity-curve code).
USE_RAW_PRICES: bool = True

# [V2-2] Extended fetch window to eliminate burn-in truncation at 2015 boundary.
#   v1 used "2015-01-01" → first valid row ≈ 2015-01-23 (21-row burn-in loss).
#   v2 uses "2014-11-01" → first valid row falls before 2015-01-01; no 2015
#   data is lost when the model script applies TRAIN_START = "2015-01-01".
START_DATE: str = "2014-11-01"
END_DATE:   str = "2026-01-01"  # exclusive — yfinance convention

TICKER_MAP: dict = {
    "BTC-USD":  "btc",
    "SPY":      "spy",
    "GC=F":     "gold",
    "NVDA":     "nvda",
    "DX-Y.NYB": "dxy",
    "^VIX":     "vix",
}

VOLUME_ASSET: str = "btc"
ANCHOR_ASSET: str = "btc"

OUTPUT_CSV:     str = "processed_dataset_v2.csv"
OUTPUT_HEATMAP: str = "correlation_heatmap_v2.png"

RSI_PERIOD:    int = 14
VOL_WINDOW:    int = 21
VOL_RATIO_WIN: int = 20


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
# STEP 1 — DATA FETCHING
# =============================================================================

def fetch_close_prices(
    ticker:         str,
    short_name:     str,
    start:          str,
    end:            str,
    include_volume: bool = False,
) -> pd.DataFrame:
    """
    Download dividend- and split-adjusted daily OHLCV data for one asset,
    returning the Close price (and optionally the Volume column).

    Parameters
    ----------
    ticker         : yfinance ticker symbol (e.g. 'BTC-USD').
    short_name     : Internal prefix for output columns (e.g. 'btc').
    start          : Start date 'YYYY-MM-DD' (inclusive).
    end            : End date   'YYYY-MM-DD' (exclusive — yfinance convention).
    include_volume : If True, also returns '{short_name}_volume'.

    Returns
    -------
    pd.DataFrame with timezone-naive DatetimeIndex.
    """
    log.info(
        "Downloading %-5s (%s)%s ...",
        short_name.upper(), ticker,
        " [+Volume]" if include_volume else "",
    )

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}'. "
            "Check the symbol spelling and network connection."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols_to_keep = ["Close"]
    rename_map   = {"Close": f"{short_name}_close"}

    if include_volume:
        cols_to_keep.append("Volume")
        rename_map["Volume"] = f"{short_name}_volume"

    df = raw[cols_to_keep].rename(columns=rename_map)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    log.info(
        "  %-5s → %5d rows  [%s … %s]",
        short_name.upper(), len(df),
        df.index.min().date(), df.index.max().date(),
    )
    return df


# =============================================================================
# STEP 2 — SYNCHRONIZATION
# =============================================================================

def merge_and_synchronize(frames: dict, anchor: str) -> pd.DataFrame:
    """
    Left-join all per-asset DataFrames onto the anchor asset's date index,
    then forward-fill NaN values introduced by non-trading days.

    Bitcoin trades 24/7 so its calendar is used as the base. NYSE-listed
    assets (SPY, NVDA) and COMEX futures (Gold) are closed on weekends;
    VIX is an exchange-hours index. Forward-fill propagates the most recent
    available closing price — the information a participant actually holds on
    a non-trading day — without introducing look-ahead bias.
    """
    df = frames[anchor].copy()

    for name, frame in frames.items():
        if name == anchor:
            continue
        df = df.join(frame, how="left")

    n_missing = int(df.isnull().sum().sum())
    log.info("Forward-filling %d NaN cells (weekends / market holidays).", n_missing)
    df.ffill(inplace=True)

    n_before = len(df)
    df.dropna(inplace=True)
    log.info(
        "Dropped %d leading NaN rows (pre-history). Remaining: %d rows.",
        n_before - len(df), len(df),
    )
    return df


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================

def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary 'is_weekend': 1 for Saturday/Sunday, 0 otherwise."""
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(np.int8)
    log.info("Feature added: is_weekend")
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns for every asset in TICKER_MAP.

    Formula: r_t = ln( P_t / P_{t-1} )

    IMPORTANT: Always computed regardless of USE_RAW_PRICES, because derived
    features (RSI, rolling vol) and the target always require btc_log_return.
    In raw-price mode, these log-return columns are subsequently dropped
    (except btc_log_return, which is retained as an auxiliary reference).
    """
    for name in TICKER_MAP.values():
        close_col  = f"{name}_close"
        return_col = f"{name}_log_return"
        df[return_col] = np.log(df[close_col] / df[close_col].shift(1))
        log.info("  Computed: %s", return_col)
    return df


def add_vix_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve the raw VIX closing price as 'vix_level'.

    VIX is stationary (mean-reverting) so its LEVEL carries regime information
    orthogonal to its log-return. This function must run after add_log_returns
    (which reads 'vix_close') and before the close-column drop.

    In USE_RAW_PRICES=True mode this also doubles as the 'VIX close price'
    feature, so no additional vix_close column is needed.
    """
    df["vix_level"] = df["vix_close"].copy()
    log.info("Feature added: vix_level  (raw VIX close; range ~10–90)")
    return df


def add_btc_volume_ratio(df: pd.DataFrame, window: int = VOL_RATIO_WIN) -> pd.DataFrame:
    """
    Add 'btc_volume_ratio': log(today's BTC volume / rolling-mean volume).

    Encoding log(V / roll_mean) centers at 0, is symmetric, and compresses
    outlier spikes. The rolling mean uses min_periods=window to avoid a
    noisy baseline during the burn-in period.
    """
    vol = df["btc_volume"].replace(0.0, np.nan).ffill()
    roll_mean = vol.rolling(window=window, min_periods=window).mean()
    df["btc_volume_ratio"] = np.log(vol / roll_mean)
    log.info(
        "Feature added: btc_volume_ratio  (log(BTC_Vol / %dd rolling mean))", window
    )
    return df


def _compute_wilder_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Compute Wilder's Smoothed RSI, normalised to [0, 1].

    Uses EWM with alpha = 1/period, adjust=False — the exact recursive
    formula. min_periods=period prevents an unstable value during burn-in.
    """
    gains  = series.clip(lower=0.0)
    losses = series.clip(upper=0.0).abs()

    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs      = avg_gain / avg_loss.replace(0.0, 1e-10)
    rsi_raw = 100.0 - (100.0 / (1.0 + rs))
    return (rsi_raw / 100.0).clip(0.0, 1.0)


def add_btc_rsi_14(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'btc_rsi_14': Wilder's 14-day RSI on BTC log returns, in [0, 1]."""
    df["btc_rsi_14"] = _compute_wilder_rsi(df["btc_log_return"], period=RSI_PERIOD)
    log.info(
        "Feature added: btc_rsi_14  (Wilder RSI(%d), burn-in ≈ %d rows)",
        RSI_PERIOD, RSI_PERIOD + 1,
    )
    return df


def add_btc_rolling_vol_21(df: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Add 'btc_roll_vol_21': 21-day rolling std of BTC log returns."""
    df["btc_roll_vol_21"] = (
        df["btc_log_return"]
        .rolling(window=window, min_periods=window)
        .std()
    )
    log.info(
        "Feature added: btc_roll_vol_21  (window=%d, burn-in = %d rows)",
        window, window + 1,
    )
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary classification target: next-day BTC return direction.

    target[t] = 1.0   if  btc_log_return[t+1] > 0
    target[t] = 0.0   if  btc_log_return[t+1] ≤ 0
    target[t] = NaN   for the LAST row (no day t+1 available)

    No look-ahead bias: features[t] → predicts direction of day t+1.
    """
    next_day_ret = df["btc_log_return"].shift(-1)

    df["target"] = np.nan
    valid_mask = next_day_ret.notna()
    df.loc[valid_mask, "target"] = (next_day_ret[valid_mask] > 0).astype(float)

    up_days   = int((df["target"] == 1).sum())
    down_days = int((df["target"] == 0).sum())
    log.info(
        "Target built — Up: %d  |  Down: %d  |  Balance: %.1f%% up",
        up_days, down_days, 100 * up_days / (up_days + down_days),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature-engineering sub-steps and return the final,
    clean dataset.

    Execution order (strictly enforced)
    ------------------------------------
    1. add_weekend_flag       – DatetimeIndex only; no column dependencies.
    2. add_log_returns        – All tickers including VIX; MUST precede vix_level.
    3. add_vix_level          – Renames vix_close → vix_level; MUST follow step 2.
    4. add_btc_volume_ratio   – Consumes btc_volume.
    5. add_btc_rsi_14         – Consumes btc_log_return; MUST follow step 2.
    6. add_btc_rolling_vol_21 – Consumes btc_log_return; MUST follow step 2.
    7. add_target             – Consumes btc_log_return; MUST follow step 2.
    8. Feature selection      – [V2-1] Conditionally swap log-returns for closes.
    9. dropna()               – Single pass removes all burn-in + last-row NaN.

    [V2-1] Feature selection logic (step 8)
    ----------------------------------------
    USE_RAW_PRICES=False (default):
        Drop raw close columns ({name}_close); keep all log-return columns.
        This is the standard stationary feature set identical to v1.

    USE_RAW_PRICES=True:
        Keep raw close columns for btc, spy, gold, nvda, dxy as features.
        vix_close already renamed to vix_level → no additional vix column.
        Drop log-return columns EXCEPT btc_log_return, which is retained as
        an auxiliary reference column for equity-curve computation in the
        model script. It is NOT a model input feature — the model script
        must exclude it from the feature list (add to PRUNED_FEATURES or
        filter explicitly).
        Drop btc_volume (already consumed for btc_volume_ratio).

    Output columns — USE_RAW_PRICES=False  (11 features + 1 target):
        is_weekend | btc_log_return | spy_log_return | gold_log_return |
        nvda_log_return | dxy_log_return | vix_log_return | vix_level |
        btc_volume_ratio | btc_rsi_14 | btc_roll_vol_21 | target

    Output columns — USE_RAW_PRICES=True  (10 features + 1 aux ref + 1 target):
        is_weekend | btc_close | spy_close | gold_close | nvda_close |
        dxy_close | vix_level | btc_volume_ratio | btc_rsi_14 |
        btc_roll_vol_21 | btc_log_return [ref] | target
    """
    df = df.copy()

    # ── Steps 1–7: compute all intermediate columns ────────────────────────────
    df = add_weekend_flag(df)
    df = add_log_returns(df)          # MUST precede vix_level rename
    df = add_vix_level(df)            # renames vix_close → vix_level
    df = add_btc_volume_ratio(df)     # consumes btc_volume
    df = add_btc_rsi_14(df)           # consumes btc_log_return
    df = add_btc_rolling_vol_21(df)   # consumes btc_log_return
    df = add_target(df)               # consumes btc_log_return

    # ── Step 8: [V2-1] Feature selection based on USE_RAW_PRICES ──────────────
    if USE_RAW_PRICES:
        log.info(
            "[V2-1] USE_RAW_PRICES=True — retaining raw close prices as features. "
            "btc_log_return kept as auxiliary reference column (NOT a model input)."
        )
        # Drop all log-return columns EXCEPT btc_log_return (auxiliary reference).
        # btc_log_return is excluded from the model's feature list by the model
        # script (add to PRUNED_FEATURES or handle explicitly).
        log_return_cols_to_drop = [
            f"{name}_log_return"
            for name in TICKER_MAP.values()
            if name != VOLUME_ASSET   # "btc" — keep btc_log_return as reference
        ]
        # Also drop vix_close if it still exists (add_vix_level renamed it but
        # errors='ignore' handles the case cleanly).
        df.drop(columns=log_return_cols_to_drop, inplace=True, errors="ignore")

        # Drop btc_volume (already encoded in btc_volume_ratio).
        df.drop(columns=[f"{VOLUME_ASSET}_volume"], inplace=True, errors="ignore")

        log.info(
            "Raw-price mode columns: %s",
            [c for c in df.columns if c != "target"],
        )

    else:
        log.info(
            "[V2-1] USE_RAW_PRICES=False — retaining log-returns as features "
            "(default stationary representation)."
        )
        # Drop all raw close columns (log-returns fully encode relative changes).
        # 'vix_close' was already renamed to 'vix_level', so it is not in the
        # close_cols list — errors='ignore' handles it gracefully.
        close_cols  = [f"{name}_close" for name in TICKER_MAP.values()]
        volume_cols = [f"{VOLUME_ASSET}_volume"]
        df.drop(columns=close_cols + volume_cols, inplace=True, errors="ignore")

    # ── Step 9: Remove all NaN rows in a single pass ──────────────────────────
    # Sources of NaN:
    #   Rows 0–21: burn-in (btc_roll_vol_21 is the binding constraint).
    #   Last row:  target is NaN (no day t+1 available).
    n_before = len(df)
    df.dropna(inplace=True)
    df["target"] = df["target"].astype(np.int8)

    log.info(
        "Dropped %d NaN rows (burn-in + last row). Final shape: %s.",
        n_before - len(df), df.shape,
    )
    return df


# =============================================================================
# STEP 4 — VISUALIZATION
# =============================================================================

_HEATMAP_LABEL_MAP: dict = {
    # Log-return mode labels
    "btc_log_return":  "BTC Return",
    "spy_log_return":  "SPY\n(S&P 500)",
    "gold_log_return": "Gold",
    "nvda_log_return": "NVDA",
    "dxy_log_return":  "DXY\n(USD Index)",
    "vix_log_return":  "VIX Return",
    # Raw-price mode labels
    "btc_close":       "BTC Price",
    "spy_close":       "SPY Price",
    "gold_close":      "Gold Price",
    "nvda_close":      "NVDA Price",
    "dxy_close":       "DXY Price",
    # Shared labels (both modes)
    "vix_level":       "VIX Level",
    "btc_volume_ratio": "BTC Vol.\nRatio",
    "btc_rsi_14":      "BTC\nRSI(14)",
    "btc_roll_vol_21": "BTC RollVol\n(21d)",
    "is_weekend":      "Is Weekend",
    "target":          "Target\n(BTC Dir.)",
}


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Compute Pearson correlations among all engineered features and the target,
    then save a colour-annotated seaborn heatmap.
    """
    # Exclude the btc_log_return auxiliary reference column from the heatmap
    # in raw-price mode (it is not a model feature; including it would make the
    # heatmap misleadingly show high btc_return ↔ btc_close correlation).
    cols_for_heatmap = [
        c for c in df.columns
        if not (USE_RAW_PRICES and c == "btc_log_return")
    ]
    corr = df[cols_for_heatmap].corr()
    corr = corr.rename(index=_HEATMAP_LABEL_MAP, columns=_HEATMAP_LABEL_MAP)

    n_cols   = len(corr.columns)
    fig_size = max(11, n_cols * 1.2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        annot_kws={"size": 9, "weight": "bold"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    mode_label = "Raw Prices" if USE_RAW_PRICES else "Log-Returns"
    ax.set_title(
        f"Pearson Correlation Matrix — All Features & BTC Direction Target\n"
        f"Feature mode: {mode_label}  |  "
        f"Sample period: {START_DATE}  →  {END_DATE}  |  "
        f"{n_cols - 1} features + target",
        fontsize=11,
        pad=16,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Correlation heatmap saved → %s", output_path)


# =============================================================================
# MAIN — Orchestrate all pipeline steps
# =============================================================================

def main() -> None:
    """Run the full v2 data-generation pipeline from fetch to export."""
    log.info("=" * 65)
    log.info("  01_data_generator_v2.py  —  PIPELINE START")
    log.info("  Period        : %s  →  %s", START_DATE, END_DATE)
    log.info("  Assets        : %s", list(TICKER_MAP.keys()))
    log.info("  USE_RAW_PRICES: %s", USE_RAW_PRICES)
    log.info("=" * 65)

    # ── STEP 1: Fetch ─────────────────────────────────────────────────────────
    log.info("[STEP 1/5] Fetching raw price data from Yahoo Finance ...")
    frames: dict = {}
    for ticker, name in TICKER_MAP.items():
        include_vol = (name == VOLUME_ASSET)
        frames[name] = fetch_close_prices(
            ticker, name, START_DATE, END_DATE,
            include_volume=include_vol,
        )

    # ── STEP 2: Synchronize ───────────────────────────────────────────────────
    log.info("[STEP 2/5] Merging and synchronizing on BTC calendar ...")
    df_sync = merge_and_synchronize(frames, anchor=ANCHOR_ASSET)

    # ── STEP 3: Feature Engineering ───────────────────────────────────────────
    log.info("[STEP 3/5] Engineering features and building target ...")
    df_final = engineer_features(df_sync)
    log.info("Final columns (%d): %s", len(df_final.columns), df_final.columns.tolist())

    # ── STEP 4: Visualize ─────────────────────────────────────────────────────
    log.info("[STEP 4/5] Generating correlation heatmap ...")
    plot_correlation_heatmap(df_final, OUTPUT_HEATMAP)

    # ── STEP 5: Export ────────────────────────────────────────────────────────
    log.info("[STEP 5/5] Saving processed dataset to CSV ...")
    df_final.to_csv(OUTPUT_CSV)
    n_features = df_final.shape[1] - 1   # exclude target column
    log.info(
        "Saved → '%s'  (%d rows × %d features  +  1 target column)",
        OUTPUT_CSV, len(df_final), n_features,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DATASET SUMMARY  (v2)")
    print("─" * 65)
    print(df_final.describe().round(5).to_string())
    print(
        f"\n  Date range  : {df_final.index.min().date()}  →  "
        f"{df_final.index.max().date()}"
    )
    print(f"  Total rows  : {len(df_final)}")
    print(f"  Features    : {n_features}")
    print(f"  Feature mode: {'Raw Prices' if USE_RAW_PRICES else 'Log-Returns (default)'}")
    print(f"\n  Target distribution:")
    vc = df_final["target"].value_counts()
    print(f"    Up   (1) : {vc.get(1, 0)}  ({100*vc.get(1,0)/len(df_final):.1f}%)")
    print(f"    Down (0) : {vc.get(0, 0)}  ({100*vc.get(0,0)/len(df_final):.1f}%)")
    print(f"\n  Derived feature ranges:")
    for col in ["vix_level", "btc_volume_ratio", "btc_rsi_14", "btc_roll_vol_21"]:
        if col in df_final.columns:
            print(
                f"    {col:<22} min={df_final[col].min():.4f}  "
                f"max={df_final[col].max():.4f}  "
                f"mean={df_final[col].mean():.4f}"
            )
    if USE_RAW_PRICES and "btc_log_return" in df_final.columns:
        print(
            "\n  [V2-1] btc_log_return retained as auxiliary reference "
            "(equity-curve use; NOT a model input feature)."
        )
    print("─" * 65 + "\n")

    log.info("01_data_generator_v2.py  —  PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
