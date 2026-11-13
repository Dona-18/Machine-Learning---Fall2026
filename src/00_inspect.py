"""Inspect the 4 raw datasets for the Bassac River ML project.

Prints:
- Shape (rows, columns)
- Data types (dtypes)
- head(3)
- Missing value counts (NaNs)
- Value ranges & odd value checks (e.g. -999, negative numbers, non-numeric strings)
"""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Find the project root containing data/raw."""
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd / "Machine-Learning---Fall2026" / "data" / "raw").exists():
        return cwd / "Machine-Learning---Fall2026"
    return Path(__file__).resolve().parent.parent


def inspect_file(file_path: Path) -> None:
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print(f"Path: {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        return

    df = pd.read_csv(file_path, low_memory=False)

    # 1. Shape
    print(f"\n[1] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. dtypes
    print("\n[2] Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    - {col}: {dtype}")

    # 3. head(3)
    print("\n[3] head(3):")
    print(df.head(3).to_string())

    # 4. NaN counts
    print("\n[4] NaN counts (missing values):")
    nan_counts = df.isna().sum()
    if nan_counts.sum() == 0:
        print("    No missing (NaN) values found across any columns.")
    else:
        for col, count in nan_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"    - {col}: {count:,} ({pct:.2f}%)")
            else:
                print(f"    - {col}: 0 (0.00%)")

    # 5. Quick sanity & odd value check
    print("\n[5] Value range & anomaly checks:")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            neg_count = (df[col] < 0).sum()
            sentinel_count = (df[col] == -999).sum() + (df[col] == -9999).sum()
            print(f"    - {col}: min = {col_min}, max = {col_max}, negative_count = {neg_count}, sentinels(-999/-9999) = {sentinel_count}")
    else:
        print("    No numeric columns detected.")

    # 6. Text columns that look numeric: show the non-numeric values
    print("\n[6] Odd strings in mostly-numeric text columns:")
    for col in df.select_dtypes(exclude="number").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.5:
            bad = df.loc[converted.isna() & df[col].notna(), col].unique()[:10]
            print(f"    - {col}: {list(bad)}")

    print("\n")


def main() -> None:
    root = get_project_root()
    raw_dir = root / "data" / "raw"

    raw_files = [
        "Bassac_01_ori.csv",
        "dhrw_water_level_2020_2026.csv",
        "glofas_discharge_phnompenh.csv",
        "khm-rainfall-subnat-full.csv",
    ]

    print(f"Project root: {root}")
    print(f"Raw data dir: {raw_dir}\n")

    for fname in raw_files:
        inspect_file(raw_dir / fname)


if __name__ == "__main__":
    main()

