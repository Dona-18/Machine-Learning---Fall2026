# Presentation Outline (Group 8 — Bassac River Water Level Forecasting)

1. **Motivation & Data Collection (Dona)**
   - Goal: Predict tomorrow's Bassac River water level (`WL_next`, in metres) in Kandal (`2020–2026`).
   - 3 datasets: `Bassac_01` (Koh Thum, ITC/IRD), `Koh_Khel` (DHRW Cambodia), `GloFAS` (Phnom Penh upstream discharge).

2. **EDA & River Behavior (Kiricheat)**
   - Dry season (`Dec–May`, ~0.7 m twice-daily ocean tides) vs. Monsoon flood (`Jun–Nov`, peaking Sep–Oct).
   - Visuals: `reports/check_bassac_full.png`, `reports/check_glofas.png`.

3. **Data Cleaning & Standardization (Tayhy)**
   - Averaged 15-min `Bassac_01` readings to 24h daily means (`n_readings >= 90`) to remove tidal noise.
   - Filtered DHRW to `Koh_Khel`; dropped 10-day Kandal rainfall (near-zero daily signal, ends early).

4. **Station Calibration & Alignment (Dona)**
   - `2020–2023` overlap (`645` days): `r = 0.9957` → `WL = 0.7936 × Koh_Khel - 0.4725` (gap `1.60 m → 0.093 m`).
   - Visual: `reports/check_overlap_scatter.png`.

5. **Feature Engineering & Leakage Guard (Samin)**
   - 7 past-only features (`WL`, `WL_lag1..3`, `WL_diff`, `upstream_discharge_lag1`, `upstream_discharge_diff`) → `2,108` clean rows.
   - Zero leakage verified by `tests/test_leakage.py` (4/4 tests passing).

6. **Models & Final Results (Dona)**
   - Chronological split: Train `1,315` rows (`2020–2024`) | Test `793` rows (`2024–2026`).
   - **Winner — Ridge Regression:** `MAE = 0.03777 m` (**3.78 cm**), `RMSE = 0.06063 m`, `R² = 0.99869` (**34.60% better than Naive Baseline** `0.05775 m`).
   - Insight: Smooth linear Ridge outperformed tree models (`XGBoost = 0.05695 m`, `RandomForest = 0.05967 m`) because river height changes continuously.
   - Visuals: `reports/ml/model_comparison_mae.png`, `reports/ml/actual_vs_predicted.png`, `reports/ml/feature_importance.png`.
