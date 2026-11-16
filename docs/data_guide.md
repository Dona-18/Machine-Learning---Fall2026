# Data Guide: Understanding the Dataset Before Starting ML

This document explains every data file in `data/raw/` and `data/processed/`, what the data represents in the real world, and the exact steps we will take before and during the machine learning pipeline (`src/ml/1_load_data.py` → `src/ml/5_evaluate.py`).

---

## 1. What This Data Is About

The goal of this project is to **predict tomorrow's daily water level (`WL_next`, in metres)** of the **Bassac River** in **Kandal Province, Cambodia (`KH08`)**.

### Physical Context
- **River network:** The Mekong River flows south to **Phnom Penh (Chaktomuk)**, where it splits into the Mekong mainstream and the **Bassac River** (western branch).
- **Travel time:** Water passing Phnom Penh takes **~1–2 days** to travel ~55 km south to **Koh Khel** and **Koh Thum** in Kandal Province.
- **Why two water-level stations are used:**
  - **Koh Thum (`Bassac_01`)** recorded 15-minute readings from **Jun 2020 – May 2024**, then stopped.
  - **Koh Khel (`DHRW`)** (~15 km upstream) has continuous daily readings from **Jul 2024 – Nov 2026** (plus flood-season readings in 2020–2023).
  - Across **645 overlapping days (2020–2023)**, both stations move almost identically (`r = 0.9957`), so `Koh_Khel` can be calibrated onto the `Bassac_01` scale to form a single **2020–2026** timeline.
- **Seasonal behavior:**
  - **Dry season (Dec–May):** Low flow; sea tides push water up and down **twice daily (~0.7 m/day)**.
  - **Rising / Peak / Falling (Jun–Nov):** Monsoon rain across the Mekong basin swells the river to ~4.5–5.6 m before draining back down.

---

## 2. Raw Data Files (`data/raw/`)

Raw files are never edited. Each has a different time resolution, format, or geographic scope.

### 2.1 `data/raw/Bassac_01_ori.csv` — Primary Water Level (2020–2024)
- **Rows × Columns:** `127,606 × 2` (`Date`, `WL`)
- **Date Range:** `2020-06-19 12:00` → `2024-05-26 14:30` (missing values: `0`)
- **Resolution:** Every **15 minutes** (up to 96 readings/day)
- **What it is:** Automated logger water level (metres above sea level, Ha Tien datum) at Koh Thum, Kandal (ITC / IRD).
- **Why it needs cleaning:** 15-minute readings contain twice-daily tidal swings (~0.7 m). Averaging to a **daily mean** removes tidal clock noise so the model learns river hydrology.

### 2.2 `data/raw/dhrw_water_level_2020_2026.csv` — Extended Water Level (2020–2026)
- **Rows × Columns:** `1,498 × 8` (`date`, `Stung_Treng`, `Kratie`, `Kampong_Cham`, `Chaktomuk`, `Neak_Loeung`, `Koh_Khel`, `Prek_Kdam`)
- **Date Range:** `2020-06-17` → `2026-11-11` (missing values: `0`)
- **Resolution:** Daily manual/gauge reading at **07:00 AM**
- **What it is:** Daily river bulletins from Cambodia's Department of Hydrology and River Works (DHRW) across 7 stations.
- **Why it needs cleaning:**
  - Only **`Koh_Khel`** is on the Bassac River in Kandal; the other 6 stations are dropped.
  - Raw `Koh_Khel` is ~1.6 m higher than `Bassac_01` because it uses a different local zero-gauge datum.
  - `2020–2023` only covers flood seasons (Jun–Nov), while `2024-07-01` → `2026-11-11` is continuous year-round.

### 2.3 `data/raw/glofas_discharge_phnompenh.csv` — Upstream River Flow (2018–2026)
- **Rows × Columns:** `3,235 × 2` (`date`, `Q`)
- **Date Range:** `2018-01-01` → `2026-11-09` (missing values: `0`)
- **Resolution:** **Daily**
- **What it is:** Simulated upstream Mekong river discharge (`Q`, in `m³/s`) at Phnom Penh (`11.525 N, 104.925 E`) from the Copernicus GloFAS hydrological model.
- **Why it matters:** Most Bassac water comes from the upper Mekong rather than local Kandal rain, so upstream discharge at Phnom Penh gives a 1–2 day physical lead signal.

---

## 3. Processed Data Files (`data/processed/`)

The cleaning scripts (`src/cleaning/01_clean_bassac.py`, `src/cleaning/02_clean_kohkhel.py`, `src/cleaning/04_clean_glofas.py`) have converted the 3 raw files into three clean daily tables:

| File | Shape | Date Range | Columns | Notes / Nulls |
|---|---|---|---|---|
| `bassac_daily.csv` | `1,438 × 4` | `2020-06-19` → `2024-05-26` | `date`, `WL`, `WL_0700`, `n_readings` | `111` days missing `WL` (sensor gaps); daily mean removes tidal noise |
| `kohkhel_daily.csv` | `1,498 × 2` | `2020-06-17` → `2026-11-11` | `date`, `WL_raw` | `0` nulls; uncalibrated `Koh_Khel` series |
| `glofas_daily.csv` | `3,235 × 2` | `2018-01-01` → `2026-11-09` | `date`, `upstream_discharge` | `0` nulls; `Q` renamed to `upstream_discharge` |

---

## 4. What We Will Do With It (Pre-ML & ML Plan)

The remaining empty scripts in `src/ml/` (`1_load_data.py` through `5_evaluate.py`) will execute the following workflow:

1. **Load & Inspect (`src/ml/1_load_data.py`)**
   - Load the 4 processed CSVs from `data/processed/`.
   - Verify date parsing, sorting, and overlapping date ranges.

2. **Prepare, Calibrate & Align (`src/ml/2_prepare_data.py`)**
   - **Station calibration (leak-free):** Using only `2020–2023` overlapping days, apply `WL_calibrated = 0.7936 * WL_raw - 0.4725` to `Koh_Khel` (`2024-07-01` → `2026-11-11`).
   - **Merge target series:** Combine `Bassac_01` (`2020–2024`) + calibrated `Koh_Khel` (`2024–2026`), leaving the June 2024 gap (`2024-05-27` → `2024-06-30`) un-interpolated.
   - **Merge upstream flow:** Left-join `glofas_daily.csv` (`upstream_discharge`) by `date` (aligned through `2026-11-09`).
   - *(Note: Kandal rainfall was dropped because it only updates every 10 days, has weak correlation with daily water-level change, and ends early on `2026-09-21`.)*

3. **Feature Engineering (`src/ml/3_features.py`)**
   - Build **7 input features (`X`)** using only data from day `t` or earlier:
     - **Water level (5):** `WL`, `WL_lag1`, `WL_lag2`, `WL_lag3`, `WL_diff` (`WL(t) - WL(t-1)`)
     - **Upstream discharge (2):** `upstream_discharge_lag1`, `upstream_discharge_diff` (`Q(t) - Q(t-1)`)
   - Create target **`y = WL_next`** (`WL` shifted by `-1` day, i.e., `t + 1`).
   - Drop rows with `NaN` (at series edges and around the June 2024 gap) so no synthetic interpolation is introduced.

4. **Chronological Split & Model Training (`src/ml/4_train.py`)**
   - **Never shuffle:** Split strictly by date (e.g., Train on `2020–2024`, Test on `2024–2026` or 80/20 chronological split).
   - Train 4 models: **Naive Baseline** (`WL_next = WL`), **Ridge Regression**, **Random Forest**, and **XGBoost**.

5. **Evaluation (`src/ml/5_evaluate.py`)**
   - Compare all models on the test set using **MAE (m)**, **RMSE (m)**, and **R²**.
   - Confirm whether ML models beat the `tomorrow = today` Naive Baseline, and inspect feature importance and Actual vs. Predicted flood-peak plots.
