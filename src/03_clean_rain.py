"""Clean khm-rainfall-subnat-full.csv for Kandal Province (KH08)."""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Find the project root containing data/raw and data/processed."""
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd / "Machine-Learning---Fall2026" / "data" / "raw").exists():
        return cwd / "Machine-Learning---Fall2026"
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = get_project_root()
    raw_path = root / "data" / "raw" / "khm-rainfall-subnat-full.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "rain_kandal_dekad.csv"

    print("=" * 80)
    print("STEP 3: Cleaning Kandal Rainfall Data (khm-rainfall-subnat-full.csv)")
    print("=" * 80)
    print(f"Reading from: {raw_path}")

    # 1. Load data
    df = pd.read_csv(raw_path, low_memory=False)

    # 2. Before filtering: inspect version for adm_level == 1 and PCODE == 'KH08'
    kandal_all = df[(df["adm_level"] == 1) & (df["PCODE"] == "KH08")].copy()

    print("\n[Check 1] Before Filtering (adm_level == 1, PCODE == 'KH08'):")
    print("    - Value counts of version:")
    version_counts = kandal_all["version"].value_counts()
    for v_name, count in version_counts.items():
        print(f"        * {v_name}: {count:>5} rows")

    print("    - Date ranges per version:")
    for v_name, grp in kandal_all.groupby("version"):
        min_v = grp["date"].min()
        max_v = grp["date"].max()
        print(f"        * {v_name:<7}: {min_v} to {max_v} ({len(grp)} rows)")

    # 3. Filter: adm_level == 1, PCODE == "KH08", version != "prelim"
    filtered = kandal_all[kandal_all["version"] != "prelim"].copy()

    # 4. Parse date and numeric columns
    filtered["date_parsed"] = pd.to_datetime(filtered["date"], errors="coerce")
    for col in ["rfh", "r1h", "r3h", "rfq"]:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    keep_cols = ["date_parsed", "rfh", "r1h", "r3h", "rfq"]
    cleaned = filtered[keep_cols].dropna(subset=["date_parsed"])
    cleaned = cleaned.sort_values("date_parsed").drop_duplicates(subset=["date_parsed"]).reset_index(drop=True)

    # 5. Check min date, max date, row count, days of month, NaNs
    min_date = cleaned["date_parsed"].min().strftime("%Y-%m-%d")
    max_date = cleaned["date_parsed"].max().strftime("%Y-%m-%d")
    row_count = len(cleaned)

    print("\n[Check 2] Filtered Dataset Dimensions & Range:")
    print(f"    - Filtered row count: {row_count}")
    print(f"    - Minimum date: {min_date}")
    print(f"    - Maximum date: {max_date}")

    print("\n[Check 3] Day of Month Value Counts (dekadal days: 1, 11, 21):")
    day_counts = cleaned["date_parsed"].dt.day.value_counts().sort_index()
    for day_num, cnt in day_counts.items():
        print(f"    Day {day_num:>2}: {cnt:>5} rows")

    print("\n[Check 4] NaN Counts per column:")
    nan_counts = cleaned[["rfh", "r1h", "r3h", "rfq"]].isna().sum()
    for c_name, c_nan in nan_counts.items():
        print(f"    - {c_name:<5}: {c_nan:>4} NaNs (only in 1981 accumulation phase)")

    # 6. Check for missing dekads from 2020-06-01 to last date
    print(f"\n[Check 5] Missing Dekads between 2020-06-01 and {max_date}:")
    start_dekad = pd.Timestamp("2020-06-01")
    end_dekad = cleaned["date_parsed"].max()

    expected_dekads = []
    curr = start_dekad
    while curr <= end_dekad:
        for day in [1, 11, 21]:
            d = pd.Timestamp(year=curr.year, month=curr.month, day=day)
            if start_dekad <= d <= end_dekad:
                expected_dekads.append(d)
        if curr.month == 12:
            curr = pd.Timestamp(year=curr.year + 1, month=1, day=1)
        else:
            curr = pd.Timestamp(year=curr.year, month=curr.month + 1, day=1)

    actual_dates = set(cleaned["date_parsed"])
    missing_dekads = [d for d in expected_dekads if d not in actual_dates]

    if not missing_dekads:
        print(f"    No missing dekads found across all {len(expected_dekads)} periods from 2020-06-01 to {max_date}.")
    else:
        print(f"    Found {len(missing_dekads)} missing dekads:")
        for md in missing_dekads:
            print(f"    - {md.strftime('%Y-%m-%d')}")

    # 7. Note regarding prelim rows vs 2026-10-21
    print("\n[Check 6] Status of Preliminary ('prelim') Rows:")
    prelim_rows = kandal_all[kandal_all["version"] == "prelim"]
    print(f"    - Excluded {len(prelim_rows)} prelim rows covering: {prelim_rows['date'].min()} to {prelim_rows['date'].max()}")
    print("    - Dates covered by prelim: 2026-10-01, 2026-10-11, 2026-10-21 (October 2026 dekads).")
    print("    - These were omitted as instructed because they are preliminary non-validated CHIRPS estimates.")

    # 8. Save output
    out_df = pd.DataFrame({
        "date": cleaned["date_parsed"].dt.strftime("%Y-%m-%d"),
        "rfh": cleaned["rfh"],
        "r1h": cleaned["r1h"],
        "r3h": cleaned["r3h"],
        "rfq": cleaned["rfq"],
    })
    out_df.to_csv(out_path, index=False)

    print(f"\n[Output] Saved cleaned dataset to: {out_path}")
    print(f"         Columns: {list(out_df.columns)}")
    print(f"         Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()

