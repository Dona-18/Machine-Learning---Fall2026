"""Calibration diagnostics and station relationship checks."""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


class Logger:
    """Tee stdout to both console and a log file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def get_project_root() -> Path:
    """Find the project root containing data/processed and results."""
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd
    if (cwd / "Machine-Learning---Fall2026" / "data" / "processed").exists():
        return cwd / "Machine-Learning---Fall2026"
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / "calibration_check_output.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        run_checks(processed_dir, results_dir)
    finally:
        sys.stdout = logger.terminal
        logger.close()


def run_checks(processed_dir: Path, results_dir: Path) -> None:
    print("=" * 80)
    print("STEP 1 FIX: Station Calibration Check & Diagnostics")
    print("=" * 80)

    # Load cleaned inputs
    bassac_path = processed_dir / "bassac_daily.csv"
    kohkhel_path = processed_dir / "kohkhel_daily.csv"

    df_b = pd.read_csv(bassac_path)
    df_k = pd.read_csv(kohkhel_path)

    df_b["dt"] = pd.to_datetime(df_b["date"])
    df_k["dt"] = pd.to_datetime(df_k["date"])

    # Shared days (2020-01-01 to 2023-12-31 where WL and WL_raw are both not NaN)
    df_merged = pd.merge(df_b, df_k, on="dt", how="inner")
    shared = df_merged[
        (df_merged["dt"] >= "2020-01-01")
        & (df_merged["dt"] <= "2023-12-31")
        & df_merged["WL"].notna()
        & df_merged["WL_raw"].notna()
    ].copy()

    # -------------------------------------------------------------------------
    # [A] Raw gap on shared days
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[A] Raw gap on shared days (WL_raw vs WL)")
    print("=" * 80)

    n_shared = len(shared)
    r_shared = shared["WL"].corr(shared["WL_raw"])
    raw_gap = shared["WL_raw"] - shared["WL"]
    mean_gap = raw_gap.mean()
    min_gap = raw_gap.min()
    max_gap = raw_gap.max()
    mae_raw = raw_gap.abs().mean()
    count_raw_le_wl = int((shared["WL_raw"] <= shared["WL"]).sum())

    print(f"n (shared days)        : {n_shared} (Guide value: 645; difference is 2 dropped partial days)")
    print(f"Pearson r              : {r_shared:.6f} (Guide value: 0.9957)")
    print(f"Mean of (WL_raw - WL)  : {mean_gap:.4f} m (Guide gap range: 1.2 to 2.1 m)")
    print(f"Min of (WL_raw - WL)   : {min_gap:.4f} m (Guide gap min: ~1.2 m)")
    print(f"Max of (WL_raw - WL)   : {max_gap:.4f} m (Guide gap max: ~2.1 m)")
    print(f"MAE of |WL_raw - WL|   : {mae_raw:.4f} m (Guide value: 1.60 m)")
    print(f"Days where WL_raw <= WL: {count_raw_le_wl}")

    print("\nMean gap in bins of WL_raw:")
    bins = [-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf]
    bin_labels = ["<3", "3-4", "4-5", "5-6", ">=6"]
    shared["bin"] = pd.cut(shared["WL_raw"], bins=bins, labels=bin_labels, right=False)
    for b_lbl in bin_labels:
        sub = shared[shared["bin"] == b_lbl]
        b_gap = sub["WL_raw"] - sub["WL"]
        print(f"  Bin {b_lbl:<4}: n = {len(sub):>3}, mean gap = {b_gap.mean():.4f} m, min = {b_gap.min():.4f} m, max = {b_gap.max():.4f} m")

    # -------------------------------------------------------------------------
    # [B] Linear fit on all shared days
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[B] Linear fit on all shared days (WL = a * WL_raw + b)")
    print("=" * 80)

    x_shared = shared["WL_raw"].values
    y_shared = shared["WL"].values
    a_fit, b_fit = np.polyfit(x_shared, y_shared, 1)

    shared["pred"] = a_fit * shared["WL_raw"] + b_fit
    shared["residual"] = shared["WL"] - shared["pred"]

    mae_ols = float(shared["residual"].abs().mean())
    rmse_ols = float(np.sqrt(np.mean(shared["residual"] ** 2)))
    max_abs_res = float(shared["residual"].abs().max())

    print(f"Fitted slope a         : {a_fit:.6f} (Guide value: 0.7936)")
    print(f"Fitted intercept b     : {b_fit:.6f} (Guide value: -0.4725)")
    print(f"In-sample MAE          : {mae_ols:.4f} m (Guide value: 0.093 m)")
    print(f"In-sample RMSE         : {rmse_ols:.4f} m")
    print(f"Max |residual|         : {max_abs_res:.4f} m")

    # -------------------------------------------------------------------------
    # [C] Residual patterns
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[C] Residual patterns (residual = WL - (a * WL_raw + b))")
    print("=" * 80)

    print("\n1. Residuals by month (months 6 to 12):")
    for m_num in range(6, 13):
        sub_m = shared[shared["dt"].dt.month == m_num]
        m_mean = sub_m["residual"].mean()
        m_std = sub_m["residual"].std()
        print(f"  Month {m_num:>2}: n = {len(sub_m):>3}, mean residual = {m_mean:+.4f} m, std = {m_std:.4f} m")

    print("\n2. Residuals by WL_raw bins from [A]:")
    for b_lbl in bin_labels:
        sub_b = shared[shared["bin"] == b_lbl]
        b_mean = sub_b["residual"].mean()
        b_mae = sub_b["residual"].abs().mean()
        print(f"  Bin {b_lbl:<4}: n = {len(sub_b):>3}, mean residual = {b_mean:+.4f} m, MAE = {b_mae:.4f} m")

    print("\n3. Rising / falling patterns (d3 = WL_raw(t) - WL_raw(t-3)):")
    # Build complete daily calendar for Koh Khel to compute d3 without false adjacency
    full_k_calendar = df_k.set_index("dt")["WL_raw"].reindex(
        pd.date_range(df_k["dt"].min(), df_k["dt"].max(), freq="D")
    )
    d3_series = full_k_calendar - full_k_calendar.shift(3)
    shared["d3"] = shared["dt"].map(d3_series)

    # Classify
    def classify_d3(val):
        if pd.isna(val):
            return "missing_d3"
        elif val > 0.10:
            return "rising"
        elif val < -0.10:
            return "falling"
        else:
            return "flat"

    shared["rf_label"] = shared["d3"].apply(classify_d3)

    for lbl in ["rising", "falling", "flat", "missing_d3"]:
        sub_rf = shared[shared["rf_label"] == lbl]
        if len(sub_rf) > 0:
            rf_mean = sub_rf["residual"].mean()
            rf_mae = sub_rf["residual"].abs().mean()
            print(f"  Label {lbl:<10}: n = {len(sub_rf):>3}, mean residual = {rf_mean:+.4f} m, MAE = {rf_mae:.4f} m")

    corr_res_raw = float(shared["residual"].corr(shared["WL_raw"]))
    print(f"\nPearson correlation between residual and WL_raw: {corr_res_raw:.6e} (orthogonal by OLS design)")

    # -------------------------------------------------------------------------
    # [D] Out-of-sample test (leave one year out)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[D] Out-of-sample test (Leave-One-Year-Out Cross-Validation)")
    print("=" * 80)

    shared["year"] = shared["dt"].dt.year
    years = [2020, 2021, 2022, 2023]

    mae_lin_all_folds = []
    mae_const_all_folds = []

    for held_out in years:
        train = shared[shared["year"] != held_out]
        test = shared[shared["year"] == held_out]
        n_t = len(test)

        # Model 1: linear fit
        a_f, b_f = np.polyfit(train["WL_raw"].values, train["WL"].values, 1)
        pred_l = a_f * test["WL_raw"] + b_f
        res_l = test["WL"] - pred_l
        mae_l = float(res_l.abs().mean())
        rmse_l = float(np.sqrt(np.mean(res_l ** 2)))
        bias_l = float(res_l.mean())
        mae_lin_all_folds.append(mae_l)

        # Model 2: constant offset
        offset = float((train["WL_raw"] - train["WL"]).mean())
        pred_c = test["WL_raw"] - offset
        res_c = test["WL"] - pred_c
        mae_c = float(res_c.abs().mean())
        rmse_c = float(np.sqrt(np.mean(res_c ** 2)))
        bias_c = float(res_c.mean())
        mae_const_all_folds.append(mae_c)

        print(f"\nFold held-out: Year {held_out} (n_test = {n_t}):")
        print(f"  Model 1 (Linear)   : a = {a_f:.4f}, b = {b_f:.4f}, MAE = {mae_l:.4f} m, RMSE = {rmse_l:.4f} m, Bias = {bias_l:+.4f} m")
        print(f"  Model 2 (Constant) : offset = {offset:.4f} m, MAE = {mae_c:.4f} m, RMSE = {rmse_c:.4f} m, Bias = {bias_c:+.4f} m")

        # Subset WL_raw < 3.0 m
        test_sub = test[test["WL_raw"] < 3.0]
        n_sub = len(test_sub)
        if n_sub > 0:
            pred_l_sub = a_f * test_sub["WL_raw"] + b_f
            res_l_sub = test_sub["WL"] - pred_l_sub
            mae_l_sub = float(res_l_sub.abs().mean())
            rmse_l_sub = float(np.sqrt(np.mean(res_l_sub ** 2)))
            bias_l_sub = float(res_l_sub.mean())

            pred_c_sub = test_sub["WL_raw"] - offset
            res_c_sub = test_sub["WL"] - pred_c_sub
            mae_c_sub = float(res_c_sub.abs().mean())
            rmse_c_sub = float(np.sqrt(np.mean(res_c_sub ** 2)))
            bias_c_sub = float(res_c_sub.mean())

            print(f"  -- Subset WL_raw < 3.0 m (n = {n_sub}):")
            print(f"     Model 1 (Linear)   : MAE = {mae_l_sub:.4f} m, RMSE = {rmse_l_sub:.4f} m, Bias = {bias_l_sub:+.4f} m")
            print(f"     Model 2 (Constant) : MAE = {mae_c_sub:.4f} m, RMSE = {rmse_c_sub:.4f} m, Bias = {bias_c_sub:+.4f} m")
        else:
            print("  -- Subset WL_raw < 3.0 m: n = 0 (no days below 3.0 m in this year)")

    avg_mae_lin = float(np.mean(mae_lin_all_folds))
    avg_mae_const = float(np.mean(mae_const_all_folds))

    print("\nSummary of Out-of-Sample MAE across 4 folds:")
    print(f"  Average MAE Model 1 (Linear)   : {avg_mae_lin:.4f} m")
    print(f"  Average MAE Model 2 (Constant) : {avg_mae_const:.4f} m")
    print(f"  In-sample MAE from [B]         : {mae_ols:.4f} m")

    # -------------------------------------------------------------------------
    # [E] Extrapolation into the test period
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[E] Extrapolation into the test period (2024-07-01 onward)")
    print("=" * 80)

    k_2024 = df_k[df_k["dt"] >= "2024-07-01"].copy()
    k_2024["WL_cal"] = a_fit * k_2024["WL_raw"] + b_fit

    shared_raw_min = float(shared["WL_raw"].min())
    shared_raw_max = float(shared["WL_raw"].max())
    test_raw_min = float(k_2024["WL_raw"].min())
    test_raw_max = float(k_2024["WL_raw"].max())

    print(f"Shared days WL_raw range     : min = {shared_raw_min:.2f} m, max = {shared_raw_max:.2f} m")
    print(f"2024 onward WL_raw range     : min = {test_raw_min:.2f} m, max = {test_raw_max:.2f} m")

    n_below_shared = int((k_2024["WL_raw"] < shared_raw_min).sum())
    n_above_shared = int((k_2024["WL_raw"] > shared_raw_max).sum())
    print(f"Rows with WL_raw < shared min ({shared_raw_min:.2f} m): {n_below_shared}")
    print(f"Rows with WL_raw > shared max ({shared_raw_max:.2f} m): {n_above_shared}")

    row_cal_min = k_2024.loc[k_2024["WL_cal"].idxmin()]
    row_cal_max = k_2024.loc[k_2024["WL_cal"].idxmax()]
    print(f"WL_cal min (2024 onward)     : {row_cal_min['WL_cal']:.4f} m on {row_cal_min['dt'].strftime('%Y-%m-%d')}")
    print(f"WL_cal max (2024 onward)     : {row_cal_max['WL_cal']:.4f} m on {row_cal_max['dt'].strftime('%Y-%m-%d')}")

    b_valid = df_b[df_b["WL"].notna()]
    row_b_min = b_valid.loc[b_valid["WL"].idxmin()]
    row_b_max = b_valid.loc[b_valid["WL"].idxmax()]
    b_min_val = float(row_b_min["WL"])
    b_max_val = float(row_b_max["WL"])
    print(f"Bassac_01 WL min (all valid) : {b_min_val:.4f} m on {row_b_min['dt'].strftime('%Y-%m-%d')}")
    print(f"Bassac_01 WL max (all valid) : {b_max_val:.4f} m on {row_b_max['dt'].strftime('%Y-%m-%d')}")

    below_b_min = k_2024[k_2024["WL_cal"] < b_min_val]
    above_b_max = k_2024[k_2024["WL_cal"] > b_max_val]
    print(f"Rows with WL_cal < Bassac_01 min ({b_min_val:.4f} m): {len(below_b_min)}")
    print(f"Rows with WL_cal > Bassac_01 max ({b_max_val:.4f} m): {len(above_b_max)}")

    print("\nMonths of rows where WL_cal < Bassac_01 min:")
    m_counts_below = below_b_min["dt"].dt.month.value_counts().sort_index()
    for m_k, cnt in m_counts_below.items():
        print(f"  Month {m_k:>2}: {cnt} rows")

    print("\nMonths of rows where WL_cal > Bassac_01 max:")
    m_counts_above = above_b_max["dt"].dt.month.value_counts().sort_index()
    for m_k, cnt in m_counts_above.items():
        print(f"  Month {m_k:>2}: {cnt} rows")

    # -------------------------------------------------------------------------
    # [F] 07:00 reading vs daily mean (Bassac_01 only)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[F] 07:00 reading vs daily mean (Bassac_01 only)")
    print("=" * 80)

    sub_0700 = df_b[df_b["WL"].notna() & df_b["WL_0700"].notna()].copy()
    diff_0700 = sub_0700["WL_0700"] - sub_0700["WL"]

    n_0700 = len(sub_0700)
    mean_0700 = float(diff_0700.mean())
    std_0700 = float(diff_0700.std())
    mae_0700 = float(diff_0700.abs().mean())
    min_0700 = float(diff_0700.min())
    max_0700 = float(diff_0700.max())

    print(f"n (valid WL and WL_0700)     : {n_0700}")
    print(f"Mean of (WL_0700 - WL)       : {mean_0700:+.4f} m")
    print(f"Std of (WL_0700 - WL)        : {std_0700:.4f} m")
    print(f"MAE of |WL_0700 - WL|        : {mae_0700:.4f} m")
    print(f"Min of (WL_0700 - WL)        : {min_0700:+.4f} m")
    print(f"Max of (WL_0700 - WL)        : {max_0700:+.4f} m")

    sub_0700["diff"] = diff_0700
    print("\nDifferences by month (months 1 to 12):")
    for m_i in range(1, 13):
        sm = sub_0700[sub_0700["dt"].dt.month == m_i]
        d_sm = sm["diff"]
        print(f"  Month {m_i:>2}: n = {len(sm):>3}, mean = {d_sm.mean():+.4f} m, std = {d_sm.std():.4f} m, MAE = {d_sm.abs().mean():.4f} m, min = {d_sm.min():+.4f} m, max = {d_sm.max():+.4f} m")

    dec_may = sub_0700[sub_0700["dt"].dt.month.isin([12, 1, 2, 3, 4, 5])]["diff"]
    jun_nov = sub_0700[sub_0700["dt"].dt.month.isin([6, 7, 8, 9, 10, 11])]["diff"]

    print("\nSeasonal comparison:")
    print(f"  Dec–May (Dry Season)  : n = {len(dec_may)}, mean = {dec_may.mean():+.4f} m, std = {dec_may.std():.4f} m, MAE = {dec_may.abs().mean():.4f} m, min = {dec_may.min():+.4f} m, max = {dec_may.max():+.4f} m")
    print(f"  Jun–Nov (Flood Season): n = {len(jun_nov)}, mean = {jun_nov.mean():+.4f} m, std = {jun_nov.std():.4f} m, MAE = {jun_nov.abs().mean():.4f} m, min = {jun_nov.min():+.4f} m, max = {jun_nov.max():+.4f} m")

    # Shared days where WL_0700 is not NaN
    shared_0700 = shared[shared["WL_0700"].notna()].copy()
    a_7, b_7 = np.polyfit(shared_0700["WL_raw"].values, shared_0700["WL_0700"].values, 1)
    res_7 = shared_0700["WL_0700"] - (a_7 * shared_0700["WL_raw"] + b_7)
    mae_7 = float(res_7.abs().mean())
    rmse_7 = float(np.sqrt(np.mean(res_7 ** 2)))

    # Measure [B] formula against WL_0700 vs WL
    pred_b_7 = a_fit * shared_0700["WL_raw"] + b_fit
    mae_b_against_7 = float((shared_0700["WL_0700"] - pred_b_7).abs().mean())
    mae_b_against_wl = float((shared_0700["WL"] - pred_b_7).abs().mean())

    print("\nFit against WL_0700 on shared days:")
    print(f"  Direct fit (WL_0700 = a * WL_raw + b) : a = {a_7:.6f}, b = {b_7:.6f}, MAE = {mae_7:.4f} m, RMSE = {rmse_7:.4f} m")
    print(f"  Formula [B] MAE measured against WL_0700: {mae_b_against_7:.4f} m")
    print(f"  Formula [B] MAE measured against WL     : {mae_b_against_wl:.4f} m")

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Diagnostics Plots...")
    print("=" * 80)

    # Plot 1: cal_residual_vs_level.png
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        shared["WL_raw"],
        shared["residual"],
        c=shared["dt"].dt.month,
        cmap="viridis",
        alpha=0.75,
        s=35,
        edgecolors="none"
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    cbar = plt.colorbar(sc, ax=ax, ticks=sorted(shared["dt"].dt.month.unique()))
    cbar.set_label("Month of Year", fontsize=11)
    ax.set_title("Calibration Residual vs. Koh_Khel Water Level (Shared Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Koh_Khel WL_raw (m)", fontsize=11)
    ax.set_ylabel("Residual: WL - (a * WL_raw + b) (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1 = results_dir / "cal_residual_vs_level.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[Plot 1] Saved: {p1}")

    # Plot 2: cal_residual_by_month.png
    fig, ax = plt.subplots(figsize=(8, 5))
    month_data = [shared[shared["dt"].dt.month == m]["residual"].values for m in range(6, 13)]
    ax.boxplot(month_data, tick_labels=[str(m) for m in range(6, 13)])
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Calibration Residual Distribution by Month (Shared Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month of Year", fontsize=11)
    ax.set_ylabel("Residual: WL - (a * WL_raw + b) (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p2 = results_dir / "cal_residual_by_month.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[Plot 2] Saved: {p2}")

    # Plot 3: cal_residual_rising_falling.png
    fig, ax = plt.subplots(figsize=(7, 5))
    rf_categories = ["rising", "flat", "falling"]
    rf_data = [shared[shared["rf_label"] == cat]["residual"].values for cat in rf_categories]
    ax.boxplot(rf_data, tick_labels=["Rising (d3 > 0.10 m)", "Flat (|d3| <= 0.10 m)", "Falling (d3 < -0.10 m)"])
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Calibration Residual by 3-Day River Dynamic", fontsize=12, fontweight="bold")
    ax.set_xlabel("River State", fontsize=11)
    ax.set_ylabel("Residual: WL - (a * WL_raw + b) (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = results_dir / "cal_residual_rising_falling.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[Plot 3] Saved: {p3}")

    # Plot 4: cal_extrapolation.png
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(k_2024["dt"], k_2024["WL_cal"], color="#2ca02c", linewidth=1.2, label="WL_cal (2024-07 onward)")
    ax.axhline(b_min_val, color="blue", linestyle="--", linewidth=1.0, label=f"Bassac_01 Min ({b_min_val:.2f} m)")
    ax.axhline(b_max_val, color="blue", linestyle="-.", linewidth=1.0, label=f"Bassac_01 Max ({b_max_val:.2f} m)")
    shared_wl_min = float(shared["WL"].min())
    shared_wl_max = float(shared["WL"].max())
    ax.axhline(shared_wl_min, color="red", linestyle=":", linewidth=1.2, label=f"Shared-day WL Min ({shared_wl_min:.2f} m)")
    ax.axhline(shared_wl_max, color="red", linestyle="--", linewidth=1.0, label=f"Shared-day WL Max ({shared_wl_max:.2f} m)")
    ax.set_title("Calibrated Water Level (WL_cal) Extrapolation (2024–2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Water Level (m)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p4 = results_dir / "cal_extrapolation.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"[Plot 4] Saved: {p4}")

    # Plot 5: cal_0700_vs_mean.png
    fig, ax = plt.subplots(figsize=(9, 5))
    m_diff_data = [sub_0700[sub_0700["dt"].dt.month == m]["diff"].values for m in range(1, 13)]
    ax.boxplot(m_diff_data, tick_labels=[str(m) for m in range(1, 13)])
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Daily Tide Bias: (WL_0700 - Daily Mean WL) by Month (Bassac_01)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month of Year", fontsize=11)
    ax.set_ylabel("WL_0700 - Daily Mean WL (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p5 = results_dir / "cal_0700_vs_mean.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"[Plot 5] Saved: {p5}")

    print("\nCalibration check script execution completed.")


if __name__ == "__main__":
    main()

