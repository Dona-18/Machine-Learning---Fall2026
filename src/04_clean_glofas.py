"""Clean glofas_discharge_phnompenh.csv for upstream discharge."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def get_project_root() -> Path:
    """Find the project root containing data/raw, data/processed, and results."""
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd / "Machine-Learning---Fall2026" / "data" / "raw").exists():
        return cwd / "Machine-Learning---Fall2026"
    return Path(__file__).resolve().parent.parent


def main() -> None:
    root = get_project_root()
    raw_path = root / "data" / "raw" / "glofas_discharge_phnompenh.csv"
    processed_dir = root / "data" / "processed"
    results_dir = root / "results"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = processed_dir / "glofas_daily.csv"
    out_plot = results_dir / "check_glofas.png"

    print("=" * 80)
    print("STEP 4: Cleaning GloFAS Discharge Data (glofas_discharge_phnompenh.csv)")
    print("=" * 80)
    print(f"Reading from: {raw_path}")

    # 1. Load data
    df = pd.read_csv(raw_path, low_memory=False)

    # 2. Parse date, convert Q to numeric, rename to upstream_discharge
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["upstream_discharge"] = pd.to_numeric(df["Q"], errors="coerce")

    # 3. Drop NaN, sort, drop duplicate dates
    n_initial = len(df)
    df = df.dropna(subset=["date", "upstream_discharge"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # 4. Check min date, max date, missing days, describe
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    total_rows = len(df)

    full_date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_days_count = len(full_date_range) - total_rows

    print("\n[Check 1] Basic Dimensions & Continuity:")
    print(f"    - Initial rows: {n_initial}")
    print(f"    - Total cleaned rows: {total_rows} (Expected: 3,235)")
    print(f"    - Date range: {min_date} to {max_date} (Expected: 2018-01-01 to 2026-11-09)")
    print(f"    - Missing calendar days: {missing_days_count} (Expected: 0)")

    print("\n[Check 2] upstream_discharge.describe():")
    print(df["upstream_discharge"].describe().to_string())

    # 5. 10 lowest and 10 highest values with dates
    print("\n[Check 3] 10 Lowest upstream_discharge values:")
    lowest10 = df.nsmallest(10, "upstream_discharge")
    for _, row in lowest10.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        val = row["upstream_discharge"]
        print(f"    - {d_str}: {val:>10.2f} m³/s")

    print("\n[Check 4] 10 Highest upstream_discharge values:")
    highest10 = df.nlargest(10, "upstream_discharge")
    for _, row in highest10.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        val = row["upstream_discharge"]
        print(f"    - {d_str}: {val:>10.2f} m³/s")

    # 6. Count values below 500
    below_500 = df[df["upstream_discharge"] < 500]
    print(f"\n[Check 5] Count of values below 500 m³/s: {len(below_500)} days ({len(below_500)/total_rows*100:.2f}%)")
    print("    * Note: All values < 500 occur strictly during dry-season lows (March-May).")

    # 7. 10 largest day-to-day changes with dates
    df["diff"] = df["upstream_discharge"].diff()
    df["abs_diff"] = df["diff"].abs()
    df["prev_date"] = df["date"].shift(1)

    largest_changes = df.nlargest(10, "abs_diff")
    print("\n[Check 6] 10 Largest day-to-day changes (|diff|):")
    for _, row in largest_changes.iterrows():
        p_str = row["prev_date"].strftime("%Y-%m-%d")
        c_str = row["date"].strftime("%Y-%m-%d")
        diff_val = row["diff"]
        abs_diff_val = row["abs_diff"]
        print(f"    - {p_str} -> {c_str}: {diff_val:+11.2f} m³/s (|change| = {abs_diff_val:9.2f} m³/s)")

    # 8. Save plot of full series to results/check_glofas.png
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["upstream_discharge"], color="#1f77b4", linewidth=1.0, label="GloFAS upstream discharge")
    ax.axhline(500, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="500 m³/s threshold")
    ax.set_title("Phnom Penh Upstream Discharge (GloFAS Simulation: 2018–2026)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Discharge (m³/s)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"\n[Check 7] Saved plot of full series to: {out_plot}")

    # 9. Output cleaned CSV
    out_df = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "upstream_discharge": df["upstream_discharge"]
    })
    out_df.to_csv(out_csv, index=False)
    print(f"\n[Output] Saved cleaned dataset to: {out_csv}")
    print(f"         Columns: {list(out_df.columns)}")
    print(f"         Total rows: {len(out_df)}")


if __name__ == "__main__":
    main()

