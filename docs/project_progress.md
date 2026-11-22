# Project Progress Tracker (Presentation & Milestone Log)

**Project:** Bassac River Daily Water Level Forecasting (`2020–2026`, Kandal Province, Cambodia)  
**Target:** Tomorrow's water level (`WL_next`, metres)

---

## 1. Milestone Checklist

| Phase | Script / Deliverable | Status | Key Outcome |
|---|---|---|---|
| **1. Data Inspection** | `src/cleaning/00_inspect.py` | Done | Audited raw CSVs in `data/raw/` |
| **2. Data Cleaning** | `src/cleaning/01_clean_bassac.py`, `02_clean_kohkhel.py`, `04_clean_glofas.py` | Done | Generated 3 daily tables in `data/processed/` |
| **3. Quality & Calibration Checks** | `src/cleaning/05_check_clean.py`, `06_check_calibration.py` | Done | Verified `Bassac_01` vs. `Koh_Khel` (`r = 0.9957`, `645` shared days); 10 plots saved in `reports/` |
| **4. Documentation & Feature Selection** | `docs/data_guide.md`, `docs/project_progress.md` | Done | Documented schemas; decided to **keep GloFAS** (`Q_diff` `r = +0.293` with `WL_next - WL`) and **drop Kandal rainfall** |
| **5. Load Data** | `src/ml/1_load_data.py` | Done | Loaded `bassac_daily.csv`, `kohkhel_daily.csv`, `glofas_daily.csv` |
| **6. Prepare & Calibrate** | `src/ml/2_prepare_data.py` | Done | Applied `0.7936 * WL_raw - 0.4725` to `Koh_Khel` (`2024–2026`), aligned `2,335` calendar days, joined GloFAS |
| **7. Feature Engineering** | `src/ml/3_features.py`, `tests/test_leakage.py` | Done | Built 7 leak-free features + `WL_next` (`2,108` clean rows); all 4 leakage tests pass |
| **8. Chronological Split & Training** | `src/ml/4_train.py` | Done | Train (`1,315` rows, `2020–2024`) / Test (`793` rows, `2024–2026`); 3-fold `TimeSeriesSplit` + saved models in `models/` |
| **9. Evaluation & Presentation Charts** | `src/ml/5_evaluate.py`, `reports/ml/` | Done | Ridge achieved `MAE = 0.03777 m` (`34.60%` better than Naive `0.05775 m`); charts saved in `reports/ml/` |

---

## 2. Key Technical Decisions (For Presentation Slides)

1. **Removing tidal noise (`Bassac_01`):** Averaged 15-minute logger readings (`127,606` rows) to daily means (`1,438` days) to remove the ~0.7 m twice-daily sea tide.
2. **Extending `2020–2024` to `2026`:** `Bassac_01` ended `2024-05-26`. Calibrated upstream `Koh_Khel` (`r = 0.9957` on `2020–2023` overlap) to align onto `Bassac_01`'s metre scale (`MAE` reduced from `1.60 m` to `0.093 m`).
3. **Why keep GloFAS upstream discharge (`glofas_daily.csv`):** `WL` alone cannot predict tomorrow's direction (`r = -0.036` with `WL_next - WL`), whereas `Q_diff` correlates at `+0.293` due to the 1–2 day Mekong travel time from Phnom Penh.
4. **Why drop Kandal rainfall (`rain_kandal_dekad.csv`):** Weak correlation with daily change (`r1h = -0.010`, `rfq = +0.050`), coarse 10-day resolution, and ends `2026-09-21` (dropping it preserves ~49 days of Oct–Nov 2026 data).
5. **Zero data leakage:** Calibration fit strictly on `2020–2023`, all features use day `t` or earlier, June 2024 gap (`2024-05-27` → `2024-06-30`) is never interpolated, and train/test split is strictly chronological.

---

## 3. Model Results Log (`2024-07-04` → `2026-11-03`, `793` Unseen Test Days)

| Model | MAE (m) | RMSE (m) | R² | Beats Naive Baseline? |
|---|---|---|---|---|
| **Ridge Regression** | **0.03777** | **0.06063** | **0.99869** | **Yes (+34.60% MAE gain)** |
| **XGBoost** | 0.05695 | 0.09221 | 0.99697 | Yes (+1.38% MAE gain) |
| **Naive Baseline (`WL_next = WL`)** | 0.05775 | 0.08458 | 0.99745 | — |
| **Random Forest** | 0.05967 | 0.09381 | 0.99687 | No (-3.31%) |
