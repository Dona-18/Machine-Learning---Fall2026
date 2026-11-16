"""Clean dhrw_water_level_2020_2026.csv for Koh_Khel station."""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Find the project root containing data/raw and data/processed."""
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd / "Machine-Learning---Fall2026" / "data" / "raw").exists():
        return cwd / "Machine-Learning---Fall2026"
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    root = get_project_root()
    raw_path = root / "data" / "raw" / "dhrw_water_level_2020_2026.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "kohkhel_daily.csv"

    print("=" * 80)
    print("STEP 2: Cleaning Koh Khel Data (dhrw_water_level_2020_2026.csv)")
    print("=" * 80)
    print(f"Reading from: {raw_path}")

    # 1. Load data
    df = pd.read_csv(raw_path, low_memory=False)

    # 2. Parse date and numeric Koh_Khel
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    parsed_wl = pd.to_numeric(df["Koh_Khel"], errors="coerce")

    # 3. Filter to date and Koh_Khel, rename to WL_raw
    df_clean = pd.DataFrame({
        "date": parsed_date,
        "WL_raw": parsed_wl
    })

    n_initial = len(df_clean)
    df_clean = df_clean.dropna(subset=["date", "WL_raw"])
    n_after_dropna = len(df_clean)

    df_clean = df_clean.sort_values("date")
    n_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["date"]).reset_index(drop=True)
    n_after_dedup = len(df_clean)

    # 4. Print basic checks
    min_date = df_clean["date"].min().strftime("%Y-%m-%d")
    max_date = df_clean["date"].max().strftime("%Y-%m-%d")
    total_rows = len(df_clean)

    print("\n[Check 1] Basic Dimensions & Range:")
    print(f"    - Initial rows: {n_initial}")
    print(f"    - Rows dropped (NaN): {n_initial - n_after_dropna}")
    print(f"    - Duplicate dates dropped: {n_before_dedup - n_after_dedup}")
    print(f"    - Total valid rows: {total_rows}")
    print(f"    - Min date: {min_date}")
    print(f"    - Max date: {max_date}")

    # 5. Rows per year
    print("\n[Check 2] Rows per year:")
    year_counts = df_clean["date"].dt.year.value_counts().sort_index()
    for year, cnt in year_counts.items():
        print(f"    {year}: {cnt:>4} rows")

    # 6. Rows from 2024-07-01 onward
    subset_2024 = df_clean[df_clean["date"] >= "2024-07-01"]
    count_2024 = len(subset_2024)
    end_2024 = subset_2024["date"].max().strftime("%Y-%m-%d")
    start_2024 = subset_2024["date"].min().strftime("%Y-%m-%d")

    print("\n[Check 3] Target Period (2024-07-01 onward):")
    print(f"    - Rows from 2024-07-01 onward: {count_2024} (Expected: 851)")
    print(f"    - Actual range: {start_2024} to {end_2024} (Expected end: 2026-11-11)")
    print(f"    - Match expectation: {count_2024 == 851 and end_2024 == '2026-11-11'}")

    # 7. Check for missing dates / date gaps
    print("\n[Check 4] Missing Date Gaps (gaps between consecutive records > 1 day):")
    df_clean["prev_date"] = df_clean["date"].shift(1)
    df_clean["day_diff"] = (df_clean["date"] - df_clean["prev_date"]).dt.days

    gap_rows = df_clean[df_clean["day_diff"] > 1].copy()
    gap_summary = {}

    if len(gap_rows) == 0:
        print("    No date gaps found. Series is continuous.")
    else:
        for _, row in gap_rows.iterrows():
            gap_start = row["prev_date"].strftime("%Y-%m-%d")
            gap_end = row["date"].strftime("%Y-%m-%d")
            missing_days = int(row["day_diff"] - 1)
            year_key = row["prev_date"].year
            gap_summary[year_key] = gap_summary.get(year_key, 0) + missing_days
            print(f"    - Gap: {gap_start} -> {gap_end} | Missing days: {missing_days}")

        print("\n[Check 5] Summary of Missing Days by Year (DHRW seasonal dry-season off-periods):")
        for yr, m_days in sorted(gap_summary.items()):
            print(f"    Year {yr}: {m_days} missing days")

    # 8. WL_raw min and max and dates
    min_idx = df_clean["WL_raw"].idxmin()
    max_idx = df_clean["WL_raw"].idxmax()
    min_val = df_clean.loc[min_idx, "WL_raw"]
    min_val_date = df_clean.loc[min_idx, "date"].strftime("%Y-%m-%d")
    max_val = df_clean.loc[max_idx, "WL_raw"]
    max_val_date = df_clean.loc[max_idx, "date"].strftime("%Y-%m-%d")

    print("\n[Check 6] WL_raw Extrema:")
    print(f"    - Minimum WL_raw: {min_val:.3f} m on {min_val_date}")
    print(f"    - Maximum WL_raw: {max_val:.3f} m on {max_val_date}")

    # 9. Format output dataframe and save
    out_df = pd.DataFrame({
        "date": df_clean["date"].dt.strftime("%Y-%m-%d"),
        "WL_raw": df_clean["WL_raw"]
    })
    out_df.to_csv(out_path, index=False)

    print(f"\n[Output] Saved cleaned dataset to: {out_path}")
    print(f"         Columns: {list(out_df.columns)}")
    print(f"         Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()

