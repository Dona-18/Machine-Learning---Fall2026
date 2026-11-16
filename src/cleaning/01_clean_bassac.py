"""Clean Bassac_01_ori.csv into daily water level series."""

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
    raw_path = root / "data" / "raw" / "Bassac_01_ori.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "bassac_daily.csv"

    print("=" * 80)
    print("STEP 1: Cleaning Bassac River Data (Bassac_01_ori.csv)")
    print("=" * 80)
    print(f"Reading from: {raw_path}")

    # 1. Parse Date and numeric WL
    df = pd.read_csv(raw_path, low_memory=False)
    parsed_dates = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    parsed_wl = pd.to_numeric(df["WL"], errors="coerce")

    bad_dates = int(parsed_dates.isna().sum())
    bad_wl = int(parsed_wl.isna().sum())
    print(f"\n[Check 1] Bad dates count: {bad_dates}")
    print(f"[Check 1] Bad WL count: {bad_wl}")
    print("\n[Check 1] WL.describe():")
    print(parsed_wl.describe().to_string())

    df["dt"] = parsed_dates
    df["WL"] = parsed_wl

    # 2. Drop bad rows, sort by Date, drop duplicate timestamps
    n_initial = len(df)
    df = df.dropna(subset=["dt", "WL"])
    dropped_bad = n_initial - len(df)

    df = df.sort_values("dt")
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["dt"])
    dropped_duplicates = n_before_dedup - len(df)

    print(f"\n[Check 2] Bad rows dropped: {dropped_bad}")
    print(f"[Check 2] Duplicate timestamps dropped: {dropped_duplicates}")

    # 3. Resample to daily mean and count readings per day (full day = 96 readings)
    df_indexed = df.set_index("dt")
    daily_mean = df_indexed["WL"].resample("D").mean()
    daily_counts = df_indexed["WL"].resample("D").count()

    print("\n[Check 3] Value counts of readings per day (0 = missing calendar day, 96 = full day):")
    counts_vc = daily_counts.value_counts().sort_index()
    for n_read, n_days in counts_vc.items():
        print(f"    {n_read:>2} readings/day : {n_days:>5} days")

    # 4. Extract 07:00 reading for each day
    df_0700 = df[df["dt"].dt.time == pd.to_datetime("07:00").time()].copy()
    df_0700["date_day"] = df_0700["dt"].dt.normalize()
    s_0700 = df_0700.set_index("date_day")["WL"]

    daily_df = pd.DataFrame({
        "date": daily_mean.index.strftime("%Y-%m-%d"),
        "WL": daily_mean.values,
        "WL_0700": daily_mean.index.map(s_0700).values,
        "n_readings": daily_counts.values,
    })

    # 5. Set daily WL to NaN if n_readings < 90
    print("\n[Check 4] Days dropped by '< 90 readings' rule (daily WL set to NaN):")
    low_reading_days = daily_df[(daily_df["n_readings"] < 90) & (daily_df["n_readings"] > 0)]
    for _, row in low_reading_days.iterrows():
        d_str = row["date"]
        cnt = int(row["n_readings"])
        mean_val = row["WL"]
        print(f"    - Date: {d_str} | Readings: {cnt}/96 | Original daily mean WL: {mean_val:.4f} m")

    daily_df.loc[daily_df["n_readings"] < 90, "WL"] = float("nan")

    # 6. Summary metrics
    valid_days = int(daily_df["WL"].notna().sum())
    missing_days = int(daily_df["WL"].isna().sum())
    total_days = len(daily_df)

    valid_subset = daily_df[daily_df["WL"].notna()]
    first_valid = valid_subset["date"].iloc[0]
    last_valid = valid_subset["date"].iloc[-1]

    lowest_row = valid_subset.loc[valid_subset["WL"].idxmin()]
    highest_row = valid_subset.loc[valid_subset["WL"].idxmax()]

    print("\n[Check 5] Final Summary Metrics:")
    print(f"    - Total daily sequence range: {total_days} days ({daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]})")
    print(f"    - Valid days (WL not NaN): {valid_days}")
    print(f"    - Missing days (WL is NaN): {missing_days}")
    print(f"    - First valid date: {first_valid}")
    print(f"    - Last valid date: {last_valid}")
    print(f"    - Lowest daily WL: {lowest_row['WL']:.4f} m on {lowest_row['date']}")
    print(f"    - Highest daily WL: {highest_row['WL']:.4f} m on {highest_row['date']}")

    # 7. Write output
    daily_df.to_csv(out_path, index=False)
    print(f"\n[Output] Saved cleaned dataset to: {out_path}")
    print(f"         Columns: {list(daily_df.columns)}")
    print(f"         Total rows: {len(daily_df)}")


if __name__ == "__main__":
    main()

