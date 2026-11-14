"""Verification and visualization script for the 4 cleaned datasets."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


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

    print("=" * 80)
    print("STEP 5: Verifying & Checking Cleaned Datasets")
    print("=" * 80)

    # 1. Load the 4 cleaned CSV files
    bassac_path = processed_dir / "bassac_daily.csv"
    kohkhel_path = processed_dir / "kohkhel_daily.csv"
    rain_path = processed_dir / "rain_kandal_dekad.csv"
    glofas_path = processed_dir / "glofas_daily.csv"

    df_b = pd.read_csv(bassac_path)
    df_k = pd.read_csv(kohkhel_path)
    df_r = pd.read_csv(rain_path)
    df_g = pd.read_csv(glofas_path)

    # Convert date columns
    df_b["dt"] = pd.to_datetime(df_b["date"])
    df_k["dt"] = pd.to_datetime(df_k["date"])
    df_r["dt"] = pd.to_datetime(df_r["date"])
    df_g["dt"] = pd.to_datetime(df_g["date"])

    # ---------------------------------------------------------
    # PRINT CHECKS
    # ---------------------------------------------------------

    # Check 1: Dimensions & NaNs for each file
    print("\n[Check 1] Summary for all 4 cleaned files:")
    files_info = [
        ("bassac_daily.csv", df_b),
        ("kohkhel_daily.csv", df_k),
        ("rain_kandal_dekad.csv", df_r),
        ("glofas_daily.csv", df_g),
    ]
    for fname, df in files_info:
        min_d = df["dt"].min().strftime("%Y-%m-%d")
        max_d = df["dt"].max().strftime("%Y-%m-%d")
        print(f"\n  File: {fname}")
        print(f"    - Total rows: {len(df):,}")
        print(f"    - Date range: {min_d} to {max_d}")
        print("    - NaN count per column:")
        for col in df.columns:
            if col != "dt":
                nan_c = df[col].isna().sum()
                pct = (nan_c / len(df)) * 100
                print(f"        * {col}: {nan_c:,} ({pct:.2f}%)")

    # Check 2: Bassac_01 valid and missing days
    valid_bassac = int(df_b["WL"].notna().sum())
    missing_bassac = int(df_b["WL"].isna().sum())
    total_bassac = len(df_b)
    print("\n[Check 2] Bassac_01 Day Counts:")
    print(f"    - Valid days (WL not NaN): {valid_bassac} (Expected: 1,327)")
    print(f"    - Missing days (WL is NaN): {missing_bassac} (Expected: 111)")
    print(f"    - Total daily sequence: {total_bassac} days")
    print(f"    - Matches expectation: {valid_bassac == 1327 and missing_bassac == 111}")

    # Check 3: Koh_Khel rows from 2024-07-01 onward & missing calendar days
    df_k_2024 = df_k[df_k["dt"] >= "2024-07-01"].copy()
    rows_k_2024 = len(df_k_2024)
    cal_range_k = pd.date_range("2024-07-01", "2026-11-11", freq="D")
    total_cal_k = len(cal_range_k)
    missing_cal_k = total_cal_k - rows_k_2024

    print("\n[Check 3] Koh_Khel Target Period (2024-07-01 to 2026-11-11):")
    print(f"    - Rows from 2024-07-01 onward: {rows_k_2024} (Expected: 851)")
    print(f"    - Total calendar days in range: {total_cal_k}")
    print(f"    - Missing calendar days: {missing_cal_k} (Expected: 13)")
    print(f"    - Matches expectation: {rows_k_2024 == 851 and missing_cal_k == 13}")

    # Check 4: Shared days (2020-01-01 to 2023-12-31) where BOTH WL and WL_raw are not NaN
    df_merged = pd.merge(df_b, df_k, on="dt", how="inner", suffixes=("_b", "_k"))
    df_shared = df_merged[
        (df_merged["dt"] >= "2020-01-01")
        & (df_merged["dt"] <= "2023-12-31")
        & df_merged["WL"].notna()
        & df_merged["WL_raw"].notna()
    ].copy()

    shared_count = len(df_shared)
    print("\n[Check 4] Shared Days (2020–2023 overlap between Bassac_01 and Koh_Khel):")
    print(f"    - Exact shared days count: {shared_count} days (Guide raw count: 645)")
    print("    - Shared days count by month (demonstrating seasonal coverage):")
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    month_dist = df_shared["dt"].dt.month.value_counts().sort_index()
    for m_num in range(1, 13):
        cnt = month_dist.get(m_num, 0)
        print(f"        * {month_names[m_num]:<9} (Month {m_num:>2}): {cnt:>3} days")

    # Check 5: Pearson correlation and mean difference on shared days
    pearson_corr = float(df_shared["WL"].corr(df_shared["WL_raw"]))
    mean_diff = float((df_shared["WL_raw"] - df_shared["WL"]).mean())
    print("\n[Check 5] Statistics on Shared Days (No fitting applied):")
    print(f"    - Pearson correlation (r): {pearson_corr:.6f} (Expected: ~0.9957)")
    print(f"    - Mean difference (WL_raw - WL): {mean_diff:+.4f} m (Koh Khel is ~{mean_diff:.2f} m higher)")

    # Check 6: Rainfall last date and calendar gap to 2026-11-11
    last_rain_date = df_r["dt"].max()
    target_end_date = pd.Timestamp("2026-11-11")
    rain_gap_days = (target_end_date - last_rain_date).days
    print("\n[Check 6] Rainfall Timeline Check:")
    print(f"    - Last date in rain_kandal_dekad.csv: {last_rain_date.strftime('%Y-%m-%d')}")
    print(f"    - Days between last rainfall date and 2026-11-11: {rain_gap_days} calendar days")

    # Check 7: Overlap gap between last valid Bassac_01 date and first Koh_Khel date from 2024-07-01
    last_valid_b = df_b[df_b["WL"].notna()]["dt"].max()
    first_k_2024 = df_k_2024["dt"].min()
    gap_between_stations = (first_k_2024 - last_valid_b).days - 1
    print("\n[Check 7] Station Handover Gap (June 2024 transition):")
    print(f"    - Last valid Bassac_01 date: {last_valid_b.strftime('%Y-%m-%d')}")
    print(f"    - First Koh_Khel date (2024 onward): {first_k_2024.strftime('%Y-%m-%d')}")
    print(f"    - In-between missing calendar days: {gap_between_stations} days (May 26 to June 30, 2024)")

    # ---------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Verification Plots...")
    print("=" * 80)

    # 1. check_overlap.png: Bassac_01 WL and Koh_Khel WL_raw (2020-06 to 2023-12)
    # Ensure gaps in data are not connected by reindexing to a full daily range
    overlap_range = pd.date_range("2020-06-01", "2023-12-31", freq="D")
    s_b_overlap = df_b.set_index("dt")["WL"].reindex(overlap_range)
    s_k_overlap = df_k.set_index("dt")["WL_raw"].reindex(overlap_range)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(overlap_range, s_b_overlap, color="#1f77b4", label="Bassac_01 (Koh Thum) WL", linewidth=1.2)
    ax.plot(overlap_range, s_k_overlap, color="#ff7f0e", label="Koh_Khel WL_raw", linewidth=1.2)
    ax.set_title("Water Level Comparison: Bassac_01 vs. Koh_Khel (2020–2023 Shared Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Water Level (m)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plot1 = results_dir / "check_overlap.png"
    fig.savefig(plot1, dpi=150)
    plt.close(fig)
    print(f"[Plot 1] Saved: {plot1}")

    # 2. check_overlap_scatter.png: scatter of WL_raw (x) vs WL (y) on shared days only, colored by month
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        df_shared["WL_raw"],
        df_shared["WL"],
        c=df_shared["dt"].dt.month,
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
        s=30
    )
    cbar = plt.colorbar(sc, ax=ax, ticks=sorted(df_shared["dt"].dt.month.unique()))
    cbar.set_label("Month of Year", fontsize=11)
    ax.set_title(f"Koh Khel vs. Bassac_01 Water Level on Shared Days (n={len(df_shared)}, r={pearson_corr:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Koh_Khel WL_raw (m)", fontsize=11)
    ax.set_ylabel("Bassac_01 WL (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot2 = results_dir / "check_overlap_scatter.png"
    fig.savefig(plot2, dpi=150)
    plt.close(fig)
    print(f"[Plot 2] Saved: {plot2}")

    # 3. check_bassac_full.png: Bassac_01 daily WL over its full range
    full_b_range = pd.date_range(df_b["dt"].min(), df_b["dt"].max(), freq="D")
    s_b_full = df_b.set_index("dt")["WL"].reindex(full_b_range)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(full_b_range, s_b_full, color="#1f77b4", linewidth=1.1, label="Bassac_01 daily mean WL")
    ax.set_title("Bassac_01 Water Level Full Time Series (Jun 2020 – May 2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Water Level (m)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plot3 = results_dir / "check_bassac_full.png"
    fig.savefig(plot3, dpi=150)
    plt.close(fig)
    print(f"[Plot 3] Saved: {plot3}")

    # 4. check_kohkhel_full.png: Koh_Khel WL_raw over its full range
    full_k_range = pd.date_range(df_k["dt"].min(), df_k["dt"].max(), freq="D")
    s_k_full = df_k.set_index("dt")["WL_raw"].reindex(full_k_range)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(full_k_range, s_k_full, color="#ff7f0e", linewidth=1.1, label="Koh_Khel daily WL_raw")
    ax.set_title("Koh_Khel Water Level Full Time Series (Jun 2020 – Nov 2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Water Level (m)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plot4 = results_dir / "check_kohkhel_full.png"
    fig.savefig(plot4, dpi=150)
    plt.close(fig)
    print(f"[Plot 4] Saved: {plot4}")

    # 5. check_rain_full.png: rfh (10-day rain) for 2020-06 to last date
    df_r_sub = df_r[df_r["dt"] >= "2020-06-01"].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df_r_sub["dt"], df_r_sub["rfh"], width=7, color="#2ca02c", alpha=0.7, label="10-day rainfall (rfh)")
    ax.plot(df_r_sub["dt"], df_r_sub["rfh"], color="#1e7b1e", linewidth=0.9, alpha=0.8)
    ax.set_title("Kandal Province 10-Day Rainfall (rfh: Jun 2020 – Sep 2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Rainfall (mm/dekad)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plot5 = results_dir / "check_rain_full.png"
    fig.savefig(plot5, dpi=150)
    plt.close(fig)
    print(f"[Plot 5] Saved: {plot5}")

    print("\nAll verification checks and plots completed successfully.")


if __name__ == "__main__":
    main()

