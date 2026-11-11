# Bassac River Water Level Forecasting

**Project in one sentence:** Predict **tomorrow's** daily water level of the Bassac River in Kandal Province, Cambodia (**2020–2026**) using **past water levels**, **Kandal rainfall**, and upstream river discharge.

---


source code/file and dataset are in Github 

## Part 0 — Project at a glance

| Item | Details |
|---|---|
| **Target (`y`)** | Tomorrow's Bassac River water level (`WL_next` in metres) |
| **Location** | Lower Bassac River, Kandal (`KH08`): Koh Thum (`2020–2024`) + calibrated Koh Khel (`2024–2026`) |
| **Timeline** | **19 Jun 2020 → 11 Nov 2026** (`2,182` total days: `1,331` from `Bassac_01` + `851` from `Koh_Khel`) |
| **Station match** | Across 645 shared days (`2020–2023`), `Bassac_01` and `Koh_Khel` correlate at **`r = 0.9957`** (99.6% match) |
| **Features (`X`)** | **11 inputs:** 5 water-level + 4 Kandal rainfall + 2 upstream discharge features |
| **ML task** | Supervised time-series regression |

**Key purpose**
- Build an end-to-end 2020–2026 ML workflow: collect → calibrate & merge → prevent data leakage → train → evaluate.
- Test whether past water levels, rainfall, and upstream flow beat a simple `tomorrow = today` baseline.

---

## Part 1 — The real-world problem

### 1.1 The river network
- The **Mekong River** flows from China → Laos → Cambodia → Vietnam → sea.
- At **Phnom Penh (Chaktomuk)** it splits into the Mekong mainstream and the **Bassac River** (western branch).
- The Bassac flows south through **Kandal (`KH08`)**, passing **Koh Khel** and **Koh Thum** (~15 km apart) before entering Vietnam.

**River Flow (North → South):**
- Upstream Mekong (Laos / Kratie / Kampong Cham) → Phnom Penh – Chaktomuk Confluence (↔ Tonle Sap Lake)
- Upstream discharge (`upstream_discharge`) is taken at Phnom Penh
- Bassac River flows south: **Koh Khel** (`Jul 2024 – Nov 2026`) → **Koh Thum** (`Jun 2020 – May 2024`) → Chau Doc, Vietnam (Sea)

**Why this matters:** Water passing Phnom Penh reaches Koh Khel and Koh Thum 1–2 days later, giving early warning before a flood wave arrives in Kandal.

### 1.2 Seasonal water level cycle

| Season | Months | River behaviour | Water Level (`Bassac_01` scale) |
|---|---|---|---|
| Dry | Dec – May | Low flow; sea tide pushes water up twice daily (~0.7 m/day) | ~1.3–1.8 m |
| Rising | Jun – Aug | Upstream monsoon rain swells the river | rises rapidly |
| Peak | Sep – Oct | Annual flood peak | ~4.5–5.3 m |
| Falling | Nov – Dec | Rain stops; water drains to the sea | drops |

**Key idea:** Most Bassac water comes from upstream along the Mekong; local Kandal rain adds a smaller boost. Therefore, **past water level** and **upstream flow (`upstream_discharge`)** are the primary predictors.

### 1.3 Why forecasting matters
- **Farming:** Farmers time rice planting and harvest around the annual flood.
- **Early warning:** Predicting water height 1 day ahead helps communities prepare.
- **Changing river:** Upstream dams and climate shifts make historical averages less reliable, so data-driven models help.

---

## Part 2 — The four datasets

### 2.1 Dataset 1: Bassac Water Level 2020–2024 (`Bassac_01_ori.csv`) — TARGET (Part 1)

| Item | Details |
|---|---|
| **Role** | Primary target & historical water level (**19 Jun 2020 → 26 May 2024**) |
| **Location** | Bassac River at Koh Thum, Kandal (`11.05 N, 105.07 E`) |
| **Recorded by** | Institute of Technology of Cambodia (ITC) & IRD / UMR G-EAU (France) |
| **Columns** | `Date` (`dd/mm/yyyy HH:MM`), `WL` (m above sea level, Ha Tien datum) |
| **Frequency** | Every 15 minutes → averaged to **daily mean** (`1,331` valid days, `107` missing days) |
| **Source** | Massuel, S. et al. (2024), DataSuds (IRD), https://doi.org/10.23708/E97HGV |

### 2.2 Dataset 2: DHRW Koh Khel Water Level 2024–2026 (`dhrw_water_level_2020_2026.csv`) — TARGET (Part 2)

| Item | Details |
|---|---|
| **Role** | Extended target & recent water level (**1 Jul 2024 → 11 Nov 2026**) |
| **Station used** | **`Koh_Khel`** only (Bassac River in Kandal, station `033402`). The other 6 river columns are dropped. |
| **Recorded by** | Department of Hydrology and River Works (DHRW), MOWRAM Cambodia (https://dhrw-cam.org/bulletintbl.php) |
| **Frequency** | Daily at 07:00 AM |
| **Rows used** | **`851` continuous days (`2024-07-01` → `2026-11-11`)** after `Bassac_01` ends (`2020–2023` shared rows are used only to calibrate the height scale). |

### 2.3 Dataset 3: Kandal Rainfall (`khm-rainfall-subnat-full.csv`) — FEATURE

| Item | Details |
|---|---|
| **Source** | WFP / HDX (CHIRPS satellite + rain gauges), https://data.humdata.org/dataset/khm-rainfall-subnational |
| **Frequency** | **Dekad** = 10-day period (`1st`, `11th`, `21st` of each month), Jan 1981 → Oct 2026 |
| **Filter used** | `adm_level == 1`, `PCODE == "KH08"` (Kandal Province), `version != "prelim"` |
| **Columns used** | `rfh` (10-day rain mm), `r1h` (1-month rain mm), `r3h` (3-month rain mm), `rfq` (rain % of normal) |

### 2.4 Dataset 4: Upstream Discharge (`glofas_discharge_phnompenh.csv`) — FEATURE

| Item | Details |
|---|---|
| **Column** | Raw `Q` → renamed to **`upstream_discharge`** (cubic metres of water flowing per second) |
| **Meaning** | 1 cubic metre is ~1,000 litres (one large tank), so `4,120` means ~4,120 tanks of water pass Phnom Penh per second. |
| **Location** | Grid cell at Phnom Penh (`11.525 N, 104.925 E`), ~55 km upstream of Koh Khel & Koh Thum |
| **Range & Source** | 1 Jan 2018 → 9 Nov 2026 (daily), Copernicus GloFAS via Open-Meteo API (`r = 0.85` with Bassac WL) |

---

## Part 3 — Extending 2020–2024 to 2026 (Station Calibration)

### 3.1 Why combine `Bassac_01` (2020–2024) and `Koh_Khel` (2024–2026)
1. **`Bassac_01` ends `2024-05-26`**, while DHRW **`Koh_Khel` continues through `2026-11-11`** (`851` continuous days from `2024-07-01`).
2. **Why not use DHRW before May 2024:** `Bassac_01` covers `2020–2024` year-round, whereas DHRW only published flood seasons (`Jun–Nov`) in `2020–2023`.
3. **Why drop the other 6 DHRW stations:** This project predicts only the Bassac River in Kandal (`Koh_Khel`).

### 3.2 What `r` is and why we use it (`r = 0.9957`)
- **What `r` is:** A correlation score from `-1` to `+1` measuring if two stations rise and fall together (`+1.00` = identical movement, `0.00` = no connection).
- **Why we need it:** To prove `Koh_Khel` (~15 km upstream) moves like `Bassac_01` before joining them.
- **Result (`2020–2023`, 645 shared days):** **`r = 0.9957` (99.6% match)** — calculated on raw data before calibration (and unchanged after calibration).

| Metric | BEFORE Calibration (Raw `Koh_Khel` vs `Bassac_01`) | AFTER Calibration (`WL_calibrated` vs `Bassac_01`) |
|---|---|---|
| **Correlation (`r`)** | **`0.9957` (99.6% match)** | **`0.9957` (unchanged)** |
| **Average height gap (MAE)** | **`1.60 m`** (different zero-gauge mark) | **`0.093 m` (9.3 cm)** (aligned to `Bassac_01` scale) |

### 3.3 How `WL_calibrated` adjusts the height scale
- **Why:** Raw `Koh_Khel` is ~1.2–2.1 m higher because it sits upstream on a different zero mark.
- **How:** Using the 645 shared days (`2020–2023`), we scale `Koh_Khel` (`0.7936 × Koh_Khel - 0.4725`) so the 2024–2026 rows sit on the exact `Bassac_01` metre scale.

| Date | Raw `Bassac_01` (Koh Thum) | Raw `Koh_Khel` (DHRW) | Calibrated `Koh_Khel` (`WL_calibrated`) |
|---|---|---|---|
| `2022-06-20` | 2.14 m | 3.40 m | **2.23 m** |
| `2022-08-10` | 3.67 m | 5.36 m | **3.78 m** |
| `2022-10-01` | 5.29 m | 7.39 m | **5.39 m** |
| `2023-09-20` | 4.98 m | 7.00 m | **5.08 m** |

---

## Part 4 — Preparing the data

### 4.1 Daily alignment

| Dataset | Raw frequency | Period used | Preparation step |
|---|---|---|---|
| `Bassac_01_ori.csv` | 15 min | Jun 2020 – May 2024 | Average each day (`resample("D").mean()`) to remove twice-daily tide noise |
| `dhrw_water_level_2020_2026.csv` | Daily (07:00) | Jul 2024 – Nov 2026 | Keep `Koh_Khel` and convert to `WL_calibrated` |
| `khm-rainfall-subnat-full.csv` | 10 days (dekad) | Jun 2020 – Oct 2026 | Divide 10-day rain by days in period, shift 1 dekad, copy across its 10 days (`ffill`) |
| `glofas_discharge_phnompenh.csv` | Daily | Jun 2020 – Nov 2026 | Rename `Q` → `upstream_discharge` |

### 4.2 Preventing data leakage (cheating)
1. **Calibration:** Calculated strictly on `2020–2023` data, never using `2024–2026` data.
2. **Rainfall:** Shifted forward by **1 dekad (`shift(1)`)** before forward-filling so each day only uses completed past 10-day rainfall.
3. **Target:** All 11 features use **today or earlier** to predict **tomorrow (`WL_next`)**.

---

## Part 5 — Features (X) and Target (y)

| # | Column | Role | Meaning | Purpose |
|---|---|---|---|---|
| 1 | `WL` | X | Today's water level (m) | Starting point for tomorrow's level |
| 2 | `WL_lag1` | X | Water level 1 day ago | Yesterday's height |
| 3 | `WL_lag2` | X | Water level 2 days ago | Short-term trend |
| 4 | `WL_lag3` | X | Water level 3 days ago | Short-term trend |
| 5 | `WL_diff` | X | Today minus yesterday (`WL - WL_lag1`) | Daily rate of rise or fall |
| 6 | `rf_daily` | X | Kandal daily rain rate (mm/day, lagged 1 dekad) | Recent local rain intensity |
| 7 | `r1h` | X | Kandal 1-month total rain (mm) | Soil wetness |
| 8 | `r3h` | X | Kandal 3-month total rain (mm) | Monsoon season strength |
| 9 | `rfq` | X | Kandal rain % of normal (`100` = normal) | Wet vs. dry year anomaly |
| 10 | `upstream_discharge_lag1` | X | Yesterday's flow at Phnom Penh (`m³/s`) | Upstream water volume heading to Kandal |
| 11 | `upstream_discharge_diff` | X | Daily change in Phnom Penh flow | Detects incoming upstream flood waves |
| — | **`WL_next`** | **y** | **Tomorrow's Bassac water level (m)** | **Target to predict** |

---

## Part 6 — System Architecture & Machine Learning Workflow

### 6.1 End-to-End Pipeline Architecture

| Stage | Input | Process | Output |
|---|---|---|---|
| **1. Data Ingestion** | 4 raw CSV files | Load 15-min logger, daily DHRW, 10-day rainfall, and daily GloFAS flow | Raw tables in memory |
| **2. Calibration & Alignment** | Raw tables | Daily mean on `Bassac_01`; calibrate `Koh_Khel` (`2024–2026`); shift & forward-fill `KH08` rain | Unified daily timeline (`2020–2026`) |
| **3. Feature Engineering** | Unified daily table | Create 11 lag/difference features (`X`) and 1-day-ahead target (`WL_next`) | Clean modeling table (~2,150 rows) |
| **4. Chronological Split** | Clean modeling table | Split strictly by date into Train (`2020–2024`) and Test (`2024–2026`) | `X_train, y_train` & `X_test, y_test` |
| **5. Model Training** | `X_train, y_train` | Train Naive Baseline, Ridge Regression, Random Forest, and XGBoost | 4 trained models |
| **6. Evaluation & Output** | `X_test, y_test` | Score with MAE, RMSE, R²; rank feature importance; plot Actual vs. Predicted | Final metrics table & forecast charts |

### 6.2 Chronological Train / Test Split (never shuffle)
- **Split B1 (By station period):** Train on **Jun 2020 – May 2024 (`Bassac_01`, ~1,323 days)** → Test on **Jul 2024 – Oct 2026 (Calibrated `Koh_Khel`, ~830 days)**.
- **Split B2 (80/20 by time):** Train on **Jun 2020 – Dec 2024 (~1,500 days)** → Test on **Jan 2025 – Oct 2026 (~660 days)**.
- **Why never shuffle:** Adjacent days are nearly identical; shuffling leaks neighbouring answers into training.

### 6.3 Models compared

| Model | How it works | Why include |
|---|---|---|
| **Naive Baseline** | `WL_next = WL` (`tomorrow = today`) | Benchmark every ML model must beat |
| **Ridge Regression** | Straight-line weighted combination of the 11 features | Captures smooth, continuous water-level trends |
| **Random Forest** | Group of decision trees voting on tomorrow's daily change | Captures non-linear rules safely |
| **XGBoost** | Sequential boosted trees where each tree fixes earlier errors | Strongest gradient-boosted tabular model |

### 6.4 Evaluation metrics

| Metric | Meaning | Goal |
|---|---|---|
| **MAE (m)** | Mean Absolute Error: average daily miss in metres | Lower is better (e.g., `0.03 m` = 3 cm average error) |
| **RMSE (m)** | Root Mean Squared Error: error score that punishes big mistakes harder | Lower is better; checks safety during flood peaks |
| **R²** | Share of water-level movement explained by the model (`0` to `1`) | Close to `1.0` (always compare MAE against Baseline) |

---

## Part 7 — Limitations

1. **Station calibration:** `2024–2026` uses DHRW `Koh_Khel` adjusted to the `Bassac_01` scale (`r = 0.9957`, average gap `0.093 m`).
2. **June 2024 gap:** `Bassac_01` ends `2024-05-26` and continuous DHRW records start `2024-07-01`.
3. **10-day rainfall resolution:** Rainfall is recorded every 10 days (up to `2026-10-21`) and copied across each 10-day period after a 1-dekad shift.
4. **Simulated upstream flow:** `upstream_discharge` comes from the GloFAS hydrological computer model rather than a physical sensor.

---

## Glossary

| Term | Meaning |
|---|---|
| **Water level (`WL`)** | Height of the river surface in metres |
| **Upstream discharge (`upstream_discharge`, Q)** | Volume of water flowing past Phnom Penh per second (`m³/s`) |
| **Correlation (`r`)** | Score from `-1` to `+1` showing how closely two datasets move together (`0.9957` = 99.6% match) |
| **Calibration (`WL_calibrated`)** | Adjusting `Koh_Khel` onto the same metre scale as `Bassac_01` |
| **Dekad** | A 10-day rainfall period |
| **Lag** | A measurement from an earlier day (`lag1` = yesterday) |
| **Leakage** | Accidentally exposing future information during model training |
| **MAE / RMSE** | Average daily error (MAE) and big-error-sensitive error (RMSE) in metres |

---

## References

1. Massuel, S. et al. (2024). *Water Level Data from the Prek System of the Cambodian Mekong Delta (2020–2024)*. DataSuds, IRD. https://doi.org/10.23708/E97HGV
2. Department of Hydrology and River Works (DHRW), MOWRAM Cambodia. *Daily River Water Level Bulletin — Koh Khel Station (2020–2026)*. https://dhrw-cam.org/bulletintbl.php
3. WFP / Humanitarian Data Exchange (HDX). *Cambodia: Rainfall Indicators at Subnational Level (CHIRPS)*. https://data.humdata.org/dataset/khm-rainfall-subnational
4. Copernicus EMS / Open-Meteo. *Global Flood Awareness System (GloFAS) River Discharge API*. https://open-meteo.com/en/docs/flood-api
