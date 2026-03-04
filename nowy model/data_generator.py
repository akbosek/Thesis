#!/usr/bin/env python3
# =============================================================================
# 01_data_generator.py
# =============================================================================
# PURPOSE:
#   Fetch multi-asset daily OHLCV data via yfinance, synchronize all
#   assets onto the Bitcoin 24/7 calendar, engineer predictive features,
#   visualize cross-asset correlations, and persist the cleaned dataset.
#
# PIPELINE STEPS:
#   1. Fetch    – Download adjusted daily Close prices for 5 assets.
#   2. Sync     – Left-join on BTC calendar; forward-fill missing values.
#   3. Engineer – Create is_weekend flag, log-returns, binary target.
#   4. Visualize– Pearson correlation heatmap (log returns + target).
#   5. Export   – Save cleaned DataFrame to processed_dataset.csv.
#
# OUTPUTS:
#   • processed_dataset.csv    – NaN-free, feature-engineered dataset
#   • correlation_heatmap.png  – seaborn Pearson correlation heatmap
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
# Edit ONLY this block to change the time window or output paths.
# =============================================================================

START_DATE: str = "2015-01-01"  # Fetch start date (inclusive)
END_DATE: str = "2026-01-01"    # Fetch end date   (exclusive — yfinance convention)

# Mapping: yfinance ticker symbol  →  short internal column prefix
# Rationale for each asset:
#   BTC-USD   – target asset; 24/7 trading drives the shared calendar
#   SPY       – S&P 500 ETF; US equity market sentiment proxy
#   GC=F      – Gold futures; safe-haven / risk-off indicator
#   NVDA      – NVIDIA; high-beta tech stock, historically correlated with BTC
#   DX-Y.NYB  – US Dollar Index; macro backdrop & inverse crypto signal
TICKER_MAP: dict = {
    "BTC-USD":  "btc",
    "SPY":      "spy",
    "GC=F":     "gold",
    "NVDA":     "nvda",
    "DX-Y.NYB": "dxy",
}

ANCHOR_ASSET: str   = "btc"                    # Asset whose calendar is the date index
OUTPUT_CSV: str     = "processed_dataset.csv"
OUTPUT_HEATMAP: str = "correlation_heatmap.png"


# =============================================================================
# LOGGING SETUP
# Using a structured log format makes pipeline progress easy to audit.
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
    ticker: str,
    short_name: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download dividend- and split-adjusted daily Close prices for one asset.

    Parameters
    ----------
    ticker     : yfinance ticker symbol (e.g. 'BTC-USD').
    short_name : Internal prefix used to name the output column (e.g. 'btc').
    start      : Start date string 'YYYY-MM-DD' (inclusive).
    end        : End date string 'YYYY-MM-DD' (exclusive).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with a timezone-naive DatetimeIndex and
        column '{short_name}_close'.

    Raises
    ------
    ValueError  if yfinance returns an empty response.
    """
    log.info("Downloading %-5s (%s) ...", short_name.upper(), ticker)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # Adjusts Close for splits & cash dividends —
                            # ensures prices are comparable across time.
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}'. "
            "Check the symbol spelling and network connection."
        )

    # ── yfinance compatibility fix ────────────────────────────────────────────
    # From yfinance ≥ 0.2 downloading a SINGLE ticker may still return a
    # MultiIndex column structure in certain configurations. Flatten it so the
    # rest of the code works regardless of library version.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ── Retain only the adjusted daily Close ─────────────────────────────────
    df = raw[["Close"]].rename(columns={"Close": f"{short_name}_close"})

    # ── Normalise DatetimeIndex to timezone-naive UTC midnight ────────────────
    # Mixing timezone-aware and naive indices causes merge errors. Strip tz.
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

def merge_and_synchronize(
    frames: dict,
    anchor: str,
) -> pd.DataFrame:
    """
    Left-join all per-asset DataFrames onto the anchor asset's date index,
    then forward-fill NaN values introduced by non-trading days.

    Why forward-fill (and not back-fill or interpolation)?
    -------------------------------------------------------
    Bitcoin trades 24/7; NYSE-listed assets (SPY, NVDA) and COMEX futures
    (Gold) trade Monday–Friday and are closed on US public holidays.
    After a left-join on the BTC date index, Saturday and Sunday rows
    carry NaN for traditional assets.

    Forward-fill propagates the most recent available *closing* price,
    which is precisely the information a market participant has on a
    weekend. This is:
      - Economically correct:  no future price is used.
      - Industry standard:     consistent with how risk systems handle
                               non-trading days.

    Back-fill and interpolation would introduce look-ahead bias.

    Parameters
    ----------
    frames : dict  –  {short_name: pd.DataFrame}. Each DataFrame has one
                      column '{short_name}_close'.
    anchor : str   –  Key of the asset used as the index baseline.

    Returns
    -------
    pd.DataFrame  –  One '{name}_close' column per asset, BTC-aligned.
    """
    # Start with the anchor asset (BTC) as the base frame.
    df = frames[anchor].copy()

    for name, frame in frames.items():
        if name == anchor:
            continue
        # LEFT join: every BTC date is kept; NaN injected where asset was closed.
        df = df.join(frame, how="left")

    n_missing = int(df.isnull().sum().sum())
    log.info("Forward-filling %d NaN cells (weekends / market holidays).", n_missing)
    df.ffill(inplace=True)

    # Drop any rows that are STILL NaN after ffill — these are rows at the very
    # beginning of the dataset before all traditional-market series have started.
    # Back-filling these would import future prices, so we drop them instead.
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
    """
    Add a binary integer column 'is_weekend' to the DataFrame.

    Encoding:  1 → Saturday (dayofweek == 5) or Sunday (dayofweek == 6)
               0 → Monday through Friday

    Motivation
    ----------
    On weekends, the traditional-asset log-returns will be exactly 0.0
    (because ffill carries Friday's close forward). The 'is_weekend' flag
    lets the LSTM learn to discount those artificial zero-returns and
    detect any genuine weekend-specific Bitcoin dynamics.
    """
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(np.int8)
    log.info("Feature added: is_weekend")
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns for every asset and add them as new columns.

    Formula:  r_t = ln( P_t / P_{t-1} )

    Why log-returns over simple (arithmetic) returns?
    --------------------------------------------------
    1. Time-additivity  – Multi-period log-returns sum, enabling cumulative
       analysis: R_{0→T} = Σ r_t.
    2. Stationarity     – Log-returns are approximately stationary, which is
       a desirable property for sequence models like LSTMs.
    3. Symmetry         – A +50 % gain followed by a −50 % loss gives
       r ≈ +0.41 and r ≈ −0.70, both with equal magnitude. Simple returns
       are asymmetric (+0.50 and −0.50), which is misleading.
    4. Neural network stability – Log-returns live in a narrower range than
       prices, reducing gradient explosion risk.

    The first row of each return series will be NaN (no prior close price).
    This NaN is removed downstream via dropna().
    """
    for name in TICKER_MAP.values():
        close_col  = f"{name}_close"
        return_col = f"{name}_log_return"
        df[return_col] = np.log(df[close_col] / df[close_col].shift(1))
        log.info("  Computed: %s", return_col)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary classification target: next-day BTC return direction.

    Definition
    ----------
    target[t] = 1.0   if  btc_log_return[t+1] > 0  (BTC closes HIGHER tomorrow)
    target[t] = 0.0   if  btc_log_return[t+1] ≤ 0  (BTC flat or lower tomorrow)
    target[t] = NaN   for the LAST row (no day t+1 available)

    Implementation note
    -------------------
    pandas .shift(-1) shifts the NEXT row's value INTO the current row.
    We then explicitly mark the last row as NaN (instead of relying on
    NaN propagation from the comparison) to guarantee that dropna() removes
    it — preventing a spurious '0' label on the last observation.

    No look-ahead bias: the model receives features[t] → predicts target[t],
    which encodes the direction of day t+1. Features for day t+1 are never
    included in the input window that produces this prediction.
    """
    next_day_ret = df["btc_log_return"].shift(-1)   # bring next day's return here

    # Initialise entire column as NaN, then fill valid rows.
    df["target"] = np.nan
    valid_mask = next_day_ret.notna()
    df.loc[valid_mask, "target"] = (next_day_ret[valid_mask] > 0).astype(float)

    up_days   = int((df["target"] == 1).sum())
    down_days = int((df["target"] == 0).sum())
    log.info(
        "Target built — Up (1): %d days | Down (0): %d days | "
        "Class balance: %.1f%% up",
        up_days, down_days, 100 * up_days / (up_days + down_days),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature-engineering sub-steps and return the final dataset.

    Column order in the output CSV
    --------------------------------
    is_weekend | btc_log_return | spy_log_return | gold_log_return |
    nvda_log_return | dxy_log_return | target

    The raw '{name}_close' price columns are dropped before export because:
      (a) prices are non-stationary — unsuitable as direct LSTM inputs.
      (b) log-returns already encode all relative price information.
    """
    df = df.copy()

    df = add_weekend_flag(df)
    df = add_log_returns(df)
    df = add_target(df)

    # ── Remove raw close columns — log-returns contain all relative price info.
    close_cols = [f"{name}_close" for name in TICKER_MAP.values()]
    df.drop(columns=close_cols, inplace=True)

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    # Two rows are guaranteed to have NaN after the steps above:
    #   • Row 0  : first log-return row (no prior close to divide by)
    #   • Last row: target row (no subsequent BTC close)
    n_before = len(df)
    df.dropna(inplace=True)

    # Convert target to int8 now that all NaN rows are gone.
    df["target"] = df["target"].astype(np.int8)

    log.info(
        "Dropped %d NaN rows (first + last). Final shape: %s.",
        n_before - len(df), df.shape,
    )
    return df


# =============================================================================
# STEP 4 — VISUALIZATION
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Compute Pearson correlations among all log-return series and the target,
    then save a colour-annotated seaborn heatmap.

    Thesis interpretation
    ---------------------
    • Strong BTC ↔ NVDA correlation: both are risk-on assets.
    • Negative BTC ↔ DXY correlation: dollar strength often pressures crypto.
    • Near-zero feature ↔ target correlation is EXPECTED — the target is
      a binary next-day direction, which is inherently noisy. Non-linear
      models (like LSTM) can exploit patterns invisible to Pearson r.
    The heatmap documents feature relationships and is a required section
    in any serious quantitative research methodology chapter.

    Parameters
    ----------
    df          : Final engineered DataFrame (output of engineer_features).
    output_path : Destination file path for the PNG image.
    """
    # Select only the continuous log-return columns plus the binary target.
    heatmap_cols = [c for c in df.columns if "log_return" in c] + ["target"]
    corr = df[heatmap_cols].corr()

    # Map technical column names to readable labels for the published figure.
    label_map = {
        "btc_log_return":  "BTC Return",
        "spy_log_return":  "SPY (S&P 500)",
        "gold_log_return": "Gold",
        "nvda_log_return": "NVDA",
        "dxy_log_return":  "DXY (USD Index)",
        "target":          "Target\n(BTC Direction)",
    }
    corr.rename(index=label_map, columns=label_map, inplace=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",      # red = negative correlation, green = positive
        center=0,           # white at r = 0 for visual symmetry
        vmin=-1,
        vmax=1,
        linewidths=0.6,
        square=True,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(
        "Pearson Correlation Matrix — Daily Log Returns & BTC Direction Target\n"
        f"Full sample period: {START_DATE}  →  {END_DATE}",
        fontsize=12,
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Correlation heatmap saved → %s", output_path)


# =============================================================================
# MAIN — Orchestrate all pipeline steps
# =============================================================================

def main() -> None:
    """Run the full data-generation pipeline from fetch to export."""
    log.info("=" * 65)
    log.info("  01_data_generator.py  —  PIPELINE START")
    log.info("  Period: %s  →  %s", START_DATE, END_DATE)
    log.info("  Assets: %s", list(TICKER_MAP.keys()))
    log.info("=" * 65)

    # ── STEP 1: Fetch ─────────────────────────────────────────────────────────
    log.info("[STEP 1/5] Fetching raw price data from Yahoo Finance ...")
    frames: dict = {}
    for ticker, name in TICKER_MAP.items():
        frames[name] = fetch_close_prices(ticker, name, START_DATE, END_DATE)

    # ── STEP 2: Synchronize ───────────────────────────────────────────────────
    log.info("[STEP 2/5] Merging and synchronizing on BTC calendar ...")
    df_sync = merge_and_synchronize(frames, anchor=ANCHOR_ASSET)

    # ── STEP 3: Feature Engineering ───────────────────────────────────────────
    log.info("[STEP 3/5] Engineering features and building target ...")
    df_final = engineer_features(df_sync)
    log.info("Final columns: %s", df_final.columns.tolist())

    # ── STEP 4: Visualize ─────────────────────────────────────────────────────
    log.info("[STEP 4/5] Generating correlation heatmap ...")
    plot_correlation_heatmap(df_final, OUTPUT_HEATMAP)

    # ── STEP 5: Export ────────────────────────────────────────────────────────
    log.info("[STEP 5/5] Saving processed dataset to CSV ...")
    df_final.to_csv(OUTPUT_CSV)
    log.info(
        "Saved → '%s'  (%d rows × %d features  +  1 target column)",
        OUTPUT_CSV, len(df_final), df_final.shape[1] - 1,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DATASET SUMMARY")
    print("─" * 65)
    print(df_final.describe().round(5).to_string())
    print(f"\n  Date range : {df_final.index.min().date()}  →  "
          f"{df_final.index.max().date()}")
    print(f"  Total rows : {len(df_final)}")
    print(f"\n  Target distribution:")
    vc = df_final["target"].value_counts()
    print(f"    Up   (1) : {vc.get(1, 0)}  ({100*vc.get(1,0)/len(df_final):.1f}%)")
    print(f"    Down (0) : {vc.get(0, 0)}  ({100*vc.get(0,0)/len(df_final):.1f}%)")
    print("─" * 65 + "\n")

    log.info("01_data_generator.py  —  PIPELINE COMPLETE")


# =============================================================================
if __name__ == "__main__":
    main()
