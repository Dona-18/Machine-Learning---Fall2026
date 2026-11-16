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

### 1.3 Project Motivation & Why Forecasting Matters
- **Agricultural livelihoods in Kandal:** Communities along the lower Bassac depend on the annual flood pulse for recession rice, irrigation scheduling, and riverbank farming. Knowing tomorrow's water level helps farmers time planting, pumping, and harvesting.
- **Flood early warning & infrastructure safety:** Rapid monsoon rises in Sep–Oct can inundate low-lying communes and roads. Even a **1-day-ahead forecast** gives local authorities and households time to protect livestock, boats, and property.
- **A changing, non-stationary Mekong:** Upstream hydropower dams and shifting monsoon patterns mean historical calendar averages no longer match actual river behavior. A data-driven ML model adapts to real-time water levels and upstream discharge rather than relying on static historical norms.
- **Why ML over a simple rule (`tomorrow = today`):** While river height changes slowly on calm days, during incoming Mekong flood pulses `today's level` lags behind. Combining recent water-level momentum (`WL_diff`) with upstream Phnom Penh discharge (`Q_diff`) lets ML models anticipate sudden rises or drops before they arrive in Kandal.

### 1.4 Where the Datasets Come From & Why Each Was Chosen

| Dataset | Source & Origin | Why We Use It |
|---|---|---|
| **`Bassac_01_ori.csv`** (`2020–2024`) | **ITC (Cambodia) & IRD / UMR G-EAU (France)** via DataSuds open research repository | High-frequency (15-min) research-grade logger at Koh Thum, Kandal. Provides a clean, year-round baseline target after daily averaging removes ~0.7 m tidal swings. |
| **`dhrw_water_level_2020_2026.csv`** (`2020–2026`) | **Department of Hydrology and River Works (DHRW)**, Ministry of Water Resources and Meteorology (MOWRAM), Cambodia | Official government daily river bulletins (`Koh_Khel` station). Needed to extend the timeline past May 2024 through **Nov 2026**; its `2020–2023` flood-season overlap (`r = 0.9957`) lets us calibrate it onto the `Bassac_01` scale. |
| **`glofas_discharge_phnompenh.csv`** (`2018–2026`) | **Copernicus GloFAS** hydrological model accessed via the **Open-Meteo Flood API** | Provides continuous daily Mekong river discharge (`m³/s`) at Phnom Penh (~55 km upstream). Because water takes 1–2 days to reach Kandal, its daily change (`Q_diff`, `r = +0.293` with `WL_next - WL`) acts as an early physical warning signal. |

*(Note: A 4th dataset — WFP/CHIRPS 10-day Kandal rainfall — was initially inspected and later dropped because upstream Mekong flow dominates Bassac water levels, 10-day rainfall showed near-zero correlation with daily water-level changes, and it ended early on `2026-09-21`.)*

---

## Part 2 — The four datasets

All 4 raw datasets live untouched in `data/raw/`. Cleaning scripts (`src/01`–`04`) convert them into aligned daily/dekadal tables in `data/processed/`.

### 2.1 Dataset 1: Bassac Water Level 2020–2024 (`Bassac_01_ori.csv`) — TARGET (Part 1)

| Item | Details |
|---|---|
| **Role** | Primary target & historical water level (**19 Jun 2020 → 26 May 2024**) |
| **Raw size & columns** | `127,606` rows × `2` columns: `Date` (`dd/mm/yyyy HH:MM` string) and `WL` (`float64`, m above sea level, Ha Tien datum) |
| **Location** | Bassac River at Koh Thum, Kandal (`11.05 N, 105.07 E`) |
| **Recorded by** | Institute of Technology of Cambodia (ITC) & IRD / UMR G-EAU (France) |
| **Frequency & aggregation** | Every 15 minutes (up to `96` readings/day) → averaged to **daily mean** (`bassac_daily.csv`: `1,438` calendar days, `1,327` valid days with `WL`, `111` missing days) |
| **Observed range (daily mean)** | Min `0.85 m`, Median `1.98 m`, Mean `2.53 m` (`std = 1.30 m`), Max `5.61 m` |
| **Source** | Massuel, S. et al. (2024), DataSuds (IRD), https://doi.org/10.23708/E97HGV |

**Deep dive:**
- **Why 15-minute readings must be averaged:** The lower Bassac is tidally influenced during the dry season, rising and falling twice a day by `~0.7 m`. Taking the daily mean (`resample("D").mean()`) removes this twice-daily tidal noise so the model learns river hydrology rather than sensor clock timing.
- **Extra diagnostic columns saved in `bassac_daily.csv`:** `WL_0700` (the 07:00 AM reading on each day, `1,329` valid days) and `n_readings` (how many 15-min samples were present, mean `88.74`). `WL_0700` is used in `src/06_check_calibration.py` to measure the tidal bias against DHRW's 07:00 AM readings.

### 2.2 Dataset 2: DHRW Koh Khel Water Level 2024–2026 (`dhrw_water_level_2020_2026.csv`) — TARGET (Part 2)

| Item | Details |
|---|---|
| **Role** | Extended target & recent water level (**1 Jul 2024 → 11 Nov 2026**) |
| **Raw size & columns** | `1,498` rows × `8` columns: `date`, `Stung_Treng`, `Kratie`, `Kampong_Cham`, `Chaktomuk`, `Neak_Loeung`, `Koh_Khel`, `Prek_Kdam` |
| **Station used** | **`Koh_Khel`** only (Bassac River in Kandal, station `033402`, ~15 km upstream of Koh Thum). The other 6 Mekong/Tonle Sap stations are dropped. |
| **Recorded by** | Department of Hydrology and River Works (DHRW), MOWRAM Cambodia (https://dhrw-cam.org/bulletintbl.php) |
| **Frequency & coverage** | Daily manual/gauge reading at **07:00 AM**; flood-season only (`Jun–Nov`) in `2020–2023`, then **continuous year-round** (`851` days) from `2024-07-01` → `2026-11-11` |
| **Observed range (raw `WL_raw`)** | Min `1.23 m`, Median `4.66 m`, Mean `4.50 m` (`std = 1.90 m`), Max `7.88 m` |

**Deep dive:**
- **Why raw `Koh_Khel` is ~1.6 m higher than `Bassac_01`:** Koh Khel sits ~15 km upstream and uses a different local zero-gauge datum. It cannot be appended directly to `Bassac_01` without calibration (otherwise there would be a sudden artificial jump in July 2024).
- **Why `2020–2023` rows are still valuable:** Even though DHRW only published flood-season rows in `2020–2023`, those `645` overlapping days with `Bassac_01` let us fit the linear calibration `0.7936 × Koh_Khel - 0.4725` (`r = 0.9957`) strictly on past data.

### 2.3 Dataset 3: Kandal Rainfall (`khm-rainfall-subnat-full.csv`) — FEATURE

| Item | Details |
|---|---|
| **Role** | Local precipitation & seasonal wetness features for Kandal Province |
| **Raw size & columns** | `364,650` rows × `15` columns (`date`, `adm_level`, `adm_id`, `PCODE`, `n_pixels`, `rfh`, `rfh_avg`, `r1h`, `r1h_avg`, `r3h`, `r3h_avg`, `rfq`, `r1q`, `r3q`, `version`) covering all Cambodian provinces & districts |
| **Filter used** | `adm_level == 1` (province level), `PCODE == "KH08"` (Kandal Province), `version != "prelim"` → **`1,647` dekads** (`1981-01-01` → `2026-09-21`) |
| **Frequency** | **Dekad** = 10-day period starting on the `1st`, `11th`, and `21st` of each month (3 dekads/month; the 3rd dekad has `8–11` days depending on month length) |
| **Columns used & ranges** | `rfh` (10-day rain: `0.71–235.08 mm`, mean `37.94`), `r1h` (1-month rolling rain: `3.04–552.66 mm`, mean `113.78`), `r3h` (3-month rolling rain: `15.44–1,019.61 mm`, mean `341.54`), `rfq` (rainfall anomaly as % of long-term average: `14.47–393.47%`, `100%` = normal) |
| **Source** | WFP / HDX (CHIRPS satellite + rain gauges), https://data.humdata.org/dataset/khm-rainfall-subnational |

**Deep dive:**
- **Why filtering `version != "prelim"` matters:** Preliminary satellite estimates can change when final gauge-corrected CHIRPS data arrives; filtering to `final` ensures consistent numbers.
- **Why dekadal data must be shifted before forward-filling:** A dekad stamped `2024-08-01` summarizes total rain over Aug 1–10. Using it on Aug 2 would leak future rain (Aug 3–10) into that day's prediction. Shifting by 1 dekad guarantees every day only sees completed past 10-day rainfall.

### 2.4 Dataset 4: Upstream Discharge (`glofas_discharge_phnompenh.csv`) — FEATURE

| Item | Details |
|---|---|
| **Role** | Upstream Mekong river flow entering the Bassac branch |
| **Raw size & columns** | `3,235` rows × `2` columns: `date` and `Q` → renamed to **`upstream_discharge`** (`m³/s`, cubic metres per second) |
| **Meaning** | `1 m³/s` = ~1,000 litres/sec (one large water tank), so `4,120 m³/s` means ~4,120 tanks of water pass Phnom Penh every second |
| **Location** | GloFAS grid cell at Phnom Penh (`11.525 N, 104.925 E`), ~55 km upstream of Koh Khel & Koh Thum |
| **Range & stats** | `2018-01-01` → `2026-11-09` (continuous daily, `0` missing values): Min `8.98`, Median `6,682.44`, Mean `13,951.14`, Max `100,229.51 m³/s` |
| **Source** | Copernicus GloFAS hydrological model via Open-Meteo Flood API (`r = 0.85` with Bassac `WL`) |

**Deep dive:**
- **Why upstream discharge is a strong physical predictor:** Most water in the Bassac River comes from the upper Mekong basin (China, Laos, Kratie) rather than local Kandal rain. Because water takes ~1–2 days to travel ~55 km from Phnom Penh to Koh Khel/Koh Thum, changes in `upstream_discharge` give an early physical signal before the water level peaks downstream.
- **Model caveat:** GloFAS `Q` is a hydrological model reanalysis/simulation (not a physical river gauge), which explains occasional near-zero dry-season values (`8.98 m³/s`).

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

The modeling table combines **11 input features (`X`)** from three physical sources to predict **1 target (`y`)**. Every feature uses information available on **day `t` or earlier** (`t - 1`, `t - 2`, `t - 3`) to predict **tomorrow (`t + 1`)**, ensuring zero future leakage.

### 5.1 Summary of all 11 features & target

| # | Column | Group | Formula / Origin | Units | Role in predicting `WL_next` |
|---|---|---|---|---|---|
| 1 | `WL` | Water Level | Water level at day `t` (calibrated if `Koh_Khel`) | m | Strongest baseline anchor; tomorrow rarely jumps far from today |
| 2 | `WL_lag1` | Water Level | `WL` shifted by 1 day (`t - 1`) | m | Yesterday's height; establishes short-term persistence |
| 3 | `WL_lag2` | Water Level | `WL` shifted by 2 days (`t - 2`) | m | 2-day memory; captures whether a rise/fall is accelerating or flattening |
| 4 | `WL_lag3` | Water Level | `WL` shifted by 3 days (`t - 3`) | m | 3-day memory; smooths out 1-day sensor or tide noise |
| 5 | `WL_diff` | Water Level | `WL(t) - WL(t - 1)` | m/day | **Momentum** — positive = river rising, negative = river falling |
| 6 | `rf_daily` | Rainfall (KH08) | `rfh / days_in_dekad`, shifted +1 dekad & forward-filled | mm/day | Recent local rainfall intensity over Kandal |
| 7 | `r1h` | Rainfall (KH08) | 1-month cumulative rain (shifted +1 dekad) | mm | Soil saturation & runoff readiness (wet soil sheds more rain into the river) |
| 8 | `r3h` | Rainfall (KH08) | 3-month cumulative rain (shifted +1 dekad) | mm | Broad monsoon strength (distinguishes early wet season from late flood season) |
| 9 | `rfq` | Rainfall (KH08) | `(rfh / rfh_avg) * 100` (shifted +1 dekad) | % | Wet vs. dry year anomaly (`100%` = historical average for that dekad) |
| 10 | `upstream_discharge_lag1` | Upstream Flow | GloFAS `Q` at Phnom Penh on day `t - 1` | m³/s | Volume of Mekong water already heading south toward Kandal (1–2 day travel time) |
| 11 | `upstream_discharge_diff` | Upstream Flow | `Q(t) - Q(t - 1)` | m³/s/day | Detects incoming flood waves or sudden upstream drops at Phnom Penh |
| — | **`WL_next`** | **Target (`y`)** | **`WL` shifted by -1 day (`t + 1`)** | **m** | **Tomorrow's Bassac water level to forecast** |

### 5.2 Why these three feature groups work together

1. **Autoregressive water-level features (`WL`, `WL_lag1..3`, `WL_diff`):**
   - River height has strong day-to-day inertia (autocorrelation > `0.99`), so `WL` alone gives a strong starting point (`Naive Baseline`: `WL_next = WL`).
   - Adding `WL_diff` and 2–3 day lags lets models tell the difference between a river at `3.5 m` that is **rising rapidly** (`+0.15 m/day`) vs. one at `3.5 m` that is **draining** (`-0.10 m/day`).
2. **Kandal rainfall features (`rf_daily`, `r1h`, `r3h`, `rfq`):**
   - Captures local catchment wetness and monsoon phase across three timescales: **10-day** (`rf_daily`, `rfq`), **1-month** (`r1h`), and **3-month** (`r3h`).
   - Even when `WL` is low in June, high `r1h`/`r3h` signals that monsoon runoff is building up.
3. **Upstream discharge features (`upstream_discharge_lag1`, `upstream_discharge_diff`):**
   - Phnom Penh sits ~55 km upstream (`1–2` days travel time). Using `lag1` and daily `diff` gives the model an early physical warning of an arriving Mekong flood pulse before it fully shows up at Koh Khel or Koh Thum.

### 5.3 Metadata & leakage guards in the modeling table
- **Metadata columns (never passed as `X`):** `date` (used only for chronological splitting & seasonal evaluation) and `station_source` (`"Bassac_01"` vs `"Koh_Khel"`).
- **Row dropping after lag/lead construction:** Creating 3-day lags (`lag3`) and 1-day lead (`WL_next`) produces `NaN` values at the start/end of the series and around the **June 2024 station gap** (`2024-05-27` → `2024-06-30`). Dropping incomplete rows yields clean, leak-free `(X, y)` samples without interpolating across missing days.

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
